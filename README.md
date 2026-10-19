# Webcam Face Attendance System with Anti-Spoofing

A smart attendance system that uses a laptop camera, the **MediaPipe Face Mesh** library, and **SQL Server**. It is tuned for instant check-in (under 0.2s) while keeping biometric data private by design and staying available when the network fails.

---

## Highlights

1. **Multi-pose enrollment:**
   - The user is guided through 3 face angles: **look straight**, **turn left**, and **turn right**.
   - The three feature vectors are stored as JSON so a tilted face can still match at check-in.
2. **Blink-to-register:**
   - On the straight-ahead step, the user must **blink** (close the eyes, then open them).
   - This blocks printed photos and photos shown on a phone from being enrolled as a fake student.
3. **AES-256 biometric encryption:**
   - An `ENCRYPTION_KEY` is created automatically and stored in `.env`.
   - Every face vector is encrypted with **AES-256 (Fernet)** before it is written to SQL Server.
   - **Backward compatible:** older plaintext vectors are detected and decrypted without interrupting the system.
4. **Check-in / check-out:**
   - By default the mode follows the clock (before 12:00: `VAO`, from 12:00: `RA`). Those two codes are the values stored in the database.
   - A proctor or teacher can override the mode with **`i`** (check-in) and **`o`** (check-out).
   - The current mode is drawn on the camera window.
5. **Offline queue and background sync:**
   - If SQL Server is unreachable, attendance is written to a local queue, `offline_queue.json`, with the real scan time.
   - The camera window shows `OFFLINE QUEUE: X recs`.
   - A background thread checks every 10 seconds, batch-inserts the queue into SQL Server when the network returns, and then clears the local cache.
6. **Cosine distance and L2 normalization:**
   - Landmarks are converted into real pixel space and L2-normalized into a flat unit vector of length 1434.
   - Matching uses cosine distance `dist = 1.0 - np.dot(...)` with a strict threshold `THRESHOLD = 0.025` (at most 2.5% angular difference, cosine similarity >= 97.5%).
   - This removes error from camera distance, face tilt, and lighting changes.
7. **Noise rejection and cooldown:**
   - A check-in is recorded only after the same face matches for 3 frames in a row.
   - A cooldown blocks duplicate check-ins (default 15 minutes, 900 seconds).

---

## Project layout

```text
├── main.py                 # Main loop: camera, pose enrollment, attendance, and sync
├── delete_student.py       # CLI tool that deletes a student and their attendance history
├── .env                    # SQL Server connection settings and the AES-256 key
├── requirements.txt        # Python dependencies
├── core/
│   ├── anti_fake.py        # L2-normalizes landmark vectors and extracts liveness (blink, head turn)
│   └── compare_face.py     # Demo script that matches a still photo
```

---

## Database design (SQL Server)

Table and column names are lowercase. The names below are the live schema, so they are left as stored.

### 1. Face-template table (`vectormathocsinh`)
```sql
CREATE TABLE vectormathocsinh (
    mahocsinh VARCHAR(50) PRIMARY KEY,
    tenhocsinh NVARCHAR(255) NOT NULL,
    facevector VARCHAR(MAX) NOT NULL -- AES-256 (Fernet) ciphertext of the pose JSON
);
```

### 2. Attendance log (`thongtinchamcong`)
```sql
CREATE TABLE thongtinchamcong (
    id INT IDENTITY(1,1) PRIMARY KEY,
    mahocsinh VARCHAR(50) FOREIGN KEY REFERENCES vectormathocsinh(mahocsinh),
    thoigianquet DATETIME DEFAULT GETDATE(),
    loai_chamcong NVARCHAR(10) DEFAULT 'VAO' -- Stored values: 'VAO' (check-in) or 'RA' (check-out)
);
```

---

## Install and run

### 1. Python environment
Conda with Python `3.10` is recommended:
```bash
# Activate your environment (example: pmcc)
conda activate pmcc

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment variables
Create a `.env` file in the project root with the SQL Server connection:
```env
DB_SERVER=127.0.0.1
DB_PORT=1433
DB_USER=sa
DB_PASSWORD=your_SQL_Server_password
DB_DATABASE=chamcongdatabase
```
*(On the first run, `ENCRYPTION_KEY` is generated and written into `.env` automatically.)*

### 3. Run the attendance app
```bash
python main.py
```

- **Attendance:** stand in front of the camera. The system recognizes the face and writes attendance to SQL Server (or the offline queue if the connection is down) in under 0.2s.
- **Keys:**
  - **`q`** on the camera window **quits**.
  - **`r`** **enrolls a new student** directly.
  - **`i`** sets the mode to **check-in**.
  - **`o`** sets the mode to **check-out**.
  - **`c`** **cancels** the current enrollment.

- **Enrollment:**
  1. Press **`r`** on the camera window.
  2. Enter the student id and student name in the terminal.
  3. Look straight at the camera and **blink** so a photo cannot pass the liveness check.
  4. Turn slightly left, then slightly right, following the on-screen instructions.
  5. When enrollment finishes, the data is encrypted and synced to the database.
  *Enrollment cancels itself if all three angles are not captured within 20 seconds.*

### 4. Delete a student
```bash
python delete_student.py <student_id>
# Example: python delete_student.py hs001
```
