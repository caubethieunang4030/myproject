# Hệ thống Chấm công & Nhận diện Khuôn mặt Chống Giả mạo (Webcam Face Attendance System)

Hệ thống chấm công thông minh sử dụng camera laptop kết hợp thư viện **MediaPipe Face Mesh** và cơ sở dữ liệu **SQL Server**. Hệ thống được tối ưu hóa cho tốc độ điểm danh tức thời (< 0.2s) đồng thời đảm bảo bảo mật sinh trắc học tuyệt đối (Privacy by Design) và tính sẵn sàng cao trước các sự cố mạng.

---

## 🌟 Tính năng nổi bật

1. **Đăng ký đa góc mặt (Multi-Pose Enrollment):**
   - Hướng dẫn người dùng quét đủ 3 hướng mặt: **Nhìn thẳng**, **Quay trái** và **Quay phải**.
   - Lưu trữ tập hợp 3 vectơ đặc trưng dưới dạng JSON để hỗ trợ so khớp góc nghiêng khi chấm công.
2. **Nháy mắt xác thực (Blink-to-Register):**
   - Ở bước nhìn thẳng khi đăng ký, hệ thống bắt buộc người dùng thực hiện động tác **nháy mắt** (nhắm mắt rồi mở mắt).
   - Ngăn chặn hoàn toàn việc sử dụng ảnh in hoặc ảnh chụp trên điện thoại để đăng ký học sinh giả mạo.
3. **Mã hóa dữ liệu Sinh trắc học (AES-256 Biometric Encryption):**
   - Tự động tạo và lưu trữ khóa mã hóa `ENCRYPTION_KEY` an toàn trong `.env`.
   - Toàn bộ dữ liệu vectơ khuôn mặt được mã hóa bằng thuật toán **AES-256 (Fernet)** trước khi lưu xuống SQL Server.
   - Hỗ trợ **tương thích ngược hoàn hảo**: tự động phát hiện và giải mã dữ liệu cũ chưa mã hóa mà không gây gián đoạn hệ thống.
4. **Phân loại Chấm công Vào/Ra (Check-In/Check-Out):**
   - Mặc định tự động xác định loại chấm công dựa trên thời gian thực hệ thống (Trước 12:00: `VAO`, Từ 12:00: `RA`).
   - Cho phép giám thị/giáo viên thay đổi chế độ chấm công thủ công bằng phím nóng **`i`** (VÀO) và **`o`** (RA).
   - Hiển thị trực quan chế độ chấm công hiện tại ngay trên màn hình camera.
5. **Đồng bộ Ngoại tuyến tự động (Offline Local Queue & Background Sync):**
   - Nếu mất kết nối với SQL Server, hệ thống tự động ghi nhận điểm danh vào hàng đợi cục bộ `offline_queue.json` bảo toàn đúng thời gian quét thực tế.
   - Hiển thị số lượng log đang chờ đồng bộ `OFFLINE QUEUE: X recs` trên màn hình camera.
   - Một luồng nền (background daemon thread) chạy song song quét mỗi 10 giây, tự động đồng bộ hàng loạt (batch insert) lên SQL Server khi có mạng lại và tự dọn dẹp bộ nhớ đệm local.
6. **Thuật toán so khớp Cosine Distance & Chuẩn hóa L2:**
   - Landmark được chuyển về không gian pixel thực tế và L2-normalize thành vectơ đơn vị 1D kích thước 1434.
   - So khớp bằng khoảng cách Cosine `dist = 1.0 - np.dot(...)` với ngưỡng nghiêm ngặt `THRESHOLD = 0.025` (độ lệch góc tối đa 2.5%, tương đương độ tương đồng CosSim >= 97.5%).
   - Triệt tiêu hoàn toàn sai lệch do khoảng cách camera xa/gần, góc nghiêng khuôn mặt, và biến động ánh sáng.
7. **Chống nhiễu & Cooldown:**
   - Chỉ ghi nhận chấm công khi khớp khuôn mặt ổn định trong 3 khung hình liên tiếp.
   - Cơ chế cooldown chống ghi trùng lặp chấm công (mặc định 15 phút - 900 giây).

---

## 🛠️ Cấu trúc dự án

```text
├── main.py                 # Luồng chính: hiển thị cam, đăng ký góc mặt, điểm danh & đồng bộ
├── delete_student.py       # Công cụ CLI nhanh để xóa dữ liệu học sinh/lịch sử khỏi DB
├── .env                    # Lưu cấu hình kết nối SQL Server & Khóa mã hóa AES-256
├── requirements.txt        # Danh sách các thư viện Python cần thiết
├── core/
│   ├── anti_fake.py        # Chuẩn hóa L2 vector landmark & trích xuất tính năng liveness (chớp mắt, xoay đầu)
│   └── compare_face.py     # Script so khớp khuôn mặt tĩnh demo từ ảnh
```

---

## 💾 Thiết kế Cơ sở dữ liệu (SQL Server)

Hệ thống sử dụng cơ sở dữ liệu SQL Server với cấu trúc bảng viết thường toàn bộ:

### 1. Bảng lưu trữ mẫu khuôn mặt (`vectormathocsinh`)
```sql
CREATE TABLE vectormathocsinh (
    mahocsinh VARCHAR(50) PRIMARY KEY,
    tenhocsinh NVARCHAR(255) NOT NULL,
    facevector VARCHAR(MAX) NOT NULL -- Chứa chuỗi mã hóa AES-256 (Fernet) của dữ liệu góc mặt JSON
);
```

### 2. Bảng nhật ký điểm danh (`thongtinchamcong`)
```sql
CREATE TABLE thongtinchamcong (
    id INT IDENTITY(1,1) PRIMARY KEY,
    mahocsinh VARCHAR(50) FOREIGN KEY REFERENCES vectormathocsinh(mahocsinh),
    thoigianquet DATETIME DEFAULT GETDATE(),
    loai_chamcong NVARCHAR(10) DEFAULT 'VAO' -- Giá trị nhận diện: 'VAO' hoặc 'RA'
);
```

---

## 🚀 Hướng dẫn cài đặt & Vận hành

### 1. Cài đặt môi trường Python
Khuyến nghị sử dụng Conda với Python `3.10`:
```bash
# Kích hoạt môi trường (ví dụ môi trường pmcc của bạn)
conda activate pmcc

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Cấu hình biến môi trường
Tạo file `.env` tại thư mục gốc của dự án với các thông tin kết nối tới SQL Server:
```env
DB_SERVER=127.0.0.1
DB_PORT=1433
DB_USER=sa
DB_PASSWORD=Mật_khẩu_SQL_Server_của_bạn
DB_DATABASE=chamcongdatabase
```
*(Lưu ý: Khóa `ENCRYPTION_KEY` sẽ tự động được sinh ra và chèn vào file `.env` khi chạy chương trình lần đầu).*

### 3. Chạy ứng dụng Chấm công
Khởi động ứng dụng bằng lệnh:
```bash
python main.py
```

- **Chấm công:** Chỉ cần đứng trước camera, hệ thống sẽ tự động nhận diện và ghi nhận chấm công vào SQL Server (hoặc hàng đợi offline nếu mất kết nối) trong chưa đầy 0.2s.
- **Phím điều khiển:**
  - Nhấn phím **`q`** trên màn hình camera để **THOÁT**.
  - Nhấn phím **`r`** để **ĐĂNG KÝ HỌC SINH MỚI** trực tiếp.
  - Nhấn phím **`i`** để chọn chế độ chấm công **VÀO (Check-In)**.
  - Nhấn phím **`o`** để chọn chế độ chấm công **RA (Check-Out)**.
  - Nhấn phím **`c`** để **HỦY** phiên đăng ký hiện tại.

- **Quy trình Đăng ký mới:**
  1. Nhấn phím **`r`** trên camera.
  2. Nhập `Mã học sinh` và `Tên học sinh` trên terminal.
  3. Nhìn thẳng vào camera và **nháy mắt** để xác thực liveness chống giả mạo bằng ảnh.
  4. Quay đầu nhẹ sang trái rồi quay đầu nhẹ sang phải theo hướng dẫn hiển thị trên camera.
  5. Đăng ký hoàn thành, dữ liệu sẽ tự động được mã hóa và đồng bộ lên database.
  *Phiên đăng ký sẽ tự động hủy nếu sau 20 giây không quét đủ các góc mặt.*

### 4. Xóa thông tin học sinh
Để xóa dữ liệu học sinh và lịch sử chấm công liên quan:
```bash
python delete_student.py <ma_hoc_sinh>
# Ví dụ: python delete_student.py hs001
```
