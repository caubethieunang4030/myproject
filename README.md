# Edge AI Biometric Attendance Appliance

Autonomous, on-premise face-attendance system for schools and SMEs. It runs computer vision on a **Raspberry Pi 5** (or a development laptop), stores **encrypted landmark vectors** (not raw photos), resists common **2D photo / phone-screen spoofs**, and keeps checking people in during **network outages** via SQLite WAL + an offline JSON queue.

**Research proposal (full write-up):** [`docs/RESEARCH_PROPOSAL.md`](docs/RESEARCH_PROPOSAL.md)

**GitHub Repository:** https://github.com/caubethieunang4030/myproject.git

---

## Why this project exists

Cloud attendance products often upload biometric data and stop working when Wi-Fi drops. Desktop vision scripts also struggle on small ARM boards (camera lag, CPU overload, easy spoofing). This repository implements a four-technique appliance designed around those constraints:

| Technique | Job in one sentence |
|---|---|
| **Threaded V4L2/MJPG capture** | Always process the *latest* frame without multi-second USB lag |
| **1434-D MediaPipe mesh matching** | Recognize identity from a normalized math vector, not a stored photo |
| **Heuristic liveness** | Reject flat paper/phone attacks using depth, motion, and blink checks |
| **SQLite WAL + offline queue** | Never lose a check-in when the DB/network path fails |

---

## Repository architecture (monorepo)

This is a **monorepo**: runtime code, Pi setup scripts, docs, and sample assets live in one Git repository so hardware knobs, thresholds, and documentation evolve together.

```text
myproject/
├── main.py                  # App entry: camera → mesh → liveness → match → storage
├── delete_student.py        # Privacy / opt-out: delete a student and related logs
├── sync_to_web.py           # (Pi branch) optional SQLite → web API sync helper
├── config.py                # Resolutions, camera backend, cosine threshold, cooldown
├── setup_rpi5.sh            # Bootstrap script for Raspberry Pi OS (Bookworm)
├── requirements.txt         # General / laptop dependency pin set
├── requirements-rpi5.txt    # ARM64-safe pins (NumPy 1.x, MediaPipe band)
├── .env.example             # Safe env template (never commit real secrets)
├── core/
│   ├── camera_stream.py     # ThreadedCamera (V4L2 + MJPG, buffer size 1)
│   ├── utils.py             # normalize_vector, depth liveness, temporal motion, EAR
│   ├── compare_face.py      # Static image comparison utility
│   ├── face_detection.py    # Supporting experiments / helpers
│   ├── facial_landmark.py
│   └── landmark_utils.py
├── docs/
│   ├── RESEARCH_PROPOSAL.md # Capstone research proposal (English)
│   └── images/              # Hardware + landmark figures for the proposal
├── images/                  # Early still-image matching samples
└── README.md                # This file
```

### Why monorepo (and not another layout)?

- **Polyrepo** (split vision / firmware / docs) adds version drift for a solo student project and makes advisor review harder.
- **Multi-package SDK layout** is premature — the deliverable is an integrated appliance, not a published library.
- **Monorepo** keeps camera ingestion, liveness thresholds, DB schema, setup scripts, and the research write-up in **one commit history**, which matches how the system was tuned (see proposal Section 5).

---

## Features

1. **Multi-pose enrollment** — straight / left / right captures stored as JSON vector sets.
2. **Blink-to-register** — enrollment requires a closed→open Eye Aspect Ratio transition to block photo registration.
3. **Runtime liveness** — Z-depth dispersion on rigid landmarks + ICD-normalized temporal motion.
4. **Cosine matching on L2-normalized 1434-D vectors** — default `COSINE_THRESHOLD = 0.025` (≥97.5% similarity) on the optimized path.
5. **Encrypted vectors at rest** — Fernet (**AES-128-CBC + HMAC**); `ENCRYPTION_KEY` auto-created in `.env` on first run.
6. **Offline-first logging** — on write failure, append to `offline_queue.json`; a daemon syncs about every 10 seconds with atomic rename/merge.
7. **Check-in / check-out modes** — auto by time of day, or hotkeys `i` / `o`.
8. **Cooldown** — default 15 minutes to suppress duplicate taps.

> Note: `main` historically used SQL Server (`pymssql`) for lab/internship setups. The Raspberry Pi appliance path (feature branch / Pi deployment) uses **local SQLite WAL** (`database.db`). Both keep the same vector + offline-queue ideas; see `docs/RESEARCH_PROPOSAL.md` §4.4 for the design rationale.

---

## Quick start (development machine)

### 1) Python environment

```bash
# Example: conda env named pmcc, Python 3.10 recommended
conda activate pmcc
pip install -r requirements.txt
```

### 2) Environment file

Copy `.env.example` to `.env` and fill secrets locally:

```env
DB_SERVER=127.0.0.1
DB_PORT=1433
DB_USER=sa
DB_PASSWORD=your_password_here
DB_DATABASE=chamcongdatabase
CAMERA_INDEX=0
```

`ENCRYPTION_KEY` is generated automatically on first launch if missing.

### 3) Run

```bash
python main.py
```

**Hotkeys**

| Key | Action |
|---|---|
| `q` | Quit |
| `r` | Register a new student |
| `i` | Force check-in mode (`VAO`) |
| `o` | Force check-out mode (`RA`) |
| `c` | Cancel an in-progress registration |

**Enrollment flow**

1. Press `r`, enter student ID + name in the terminal.
2. Look straight and **blink** (closed then open).
3. Turn slightly left, then right, as on-screen prompts indicate.
4. Vectors are encrypted and stored. Registration times out after ~20 seconds if incomplete.

### 4) Delete a student (opt-out / purge)

```bash
python delete_student.py <student_id>
# example:
python delete_student.py hs001
```

---

## Raspberry Pi 5 deployment (appliance path)

On Raspberry Pi OS Bookworm:

```bash
chmod +x setup_rpi5.sh
./setup_rpi5.sh
source venv/bin/activate
python main.py
```

Important Pi settings live in `config.py` / environment:

- Capture: 1280×720 @ 30 FPS, MJPG, `CAP_PROP_BUFFERSIZE = 1`
- Inference resize: 640×480
- Camera index via `CAMERA_INDEX` (Logitech Brio 100 often appears as a non-zero index)

Use `requirements-rpi5.txt` on ARM64 to avoid NumPy 2.x / MediaPipe ABI breaks.

Hardware diagram and BOM: see [`docs/RESEARCH_PROPOSAL.md`](docs/RESEARCH_PROPOSAL.md) §4.1 and §6.2.

---

## Database sketch

**Student vectors — `vectormathocsinh`**

| Column | Meaning |
|---|---|
| `mahocsinh` | Student ID (primary key) |
| `tenhocsinh` | Display name |
| `facevector` | Fernet-encrypted JSON of pose vectors |

**Attendance — `thongtinchamcong`**

| Column | Meaning |
|---|---|
| `id` | Row id |
| `mahocsinh` | Student ID |
| `loai_chamcong` | `VAO` / `RA` |
| `thoigianquet` | Timestamp of scan |

---

## Privacy notes

- Store landmark vectors, not photos/videos of check-ins.
- Keep `.env` out of Git (see `.gitignore`).
- Honor opt-out with `delete_student.py`.
- Purge pilot data after the capstone defense as described in the proposal ethics section.

---

## Documentation map

| Document | Contents |
|---|---|
| [`docs/RESEARCH_PROPOSAL.md`](docs/RESEARCH_PROPOSAL.md) | Full research proposal: techniques, architecture, liveness “skeleton,” storage rationale, testing & timeline |
| [`docs/images/`](docs/images/) | Hardware appliance figure + facial landmark liveness figure |

---

## References (short list)

- MediaPipe Face Mesh / perception pipelines (Lugaresi et al., 2019)
- Eye Aspect Ratio blink detection (Soukupová & Čech, 2016)
- SQLite WAL documentation
- PyCA Fernet specification
- Full reference list: [`docs/RESEARCH_PROPOSAL.md`](docs/RESEARCH_PROPOSAL.md) §7
