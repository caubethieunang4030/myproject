# Hệ thống Chấm công & Nhận diện Khuôn mặt Chống Giả mạo (Webcam Face Recognition System)

Hệ thống chấm công thông minh sử dụng camera laptop kết hợp thư viện **MediaPipe Face Mesh** và cơ sở dữ liệu **SQL Server**. Hệ thống được tối ưu hóa cho tốc độ điểm danh tức thời (< 0.2s) đồng thời bảo mật tuyệt đối trước các hình thức giả mạo bằng ảnh tĩnh/màn hình thông qua cơ chế quét đa góc và nháy mắt xác thực liveness khi đăng ký.

---

## 🌟 Tính năng nổi bật

1. **Đăng ký đa góc mặt (Multi-Pose Enrollment):**
   - Hướng dẫn người dùng quét đủ 3 hướng mặt: **Nhìn thẳng**, **Quay trái** và **Quay phải**.
   - Lưu trữ tập hợp 3 vectơ đặc trưng dưới dạng JSON trong SQL Server để hỗ trợ so khớp góc nghiêng khi chấm công.
2. **Nháy mắt xác thực (Blink-to-Register):**
   - Ở bước nhìn thẳng khi đăng ký, hệ thống bắt buộc người dùng thực hiện động tác **nháy mắt** (nhắm mắt rồi mở mắt).
   - Ngăn chặn hoàn toàn việc kẻ gian sử dụng ảnh in hoặc ảnh chụp trên điện thoại để đăng ký học sinh giả mạo.
3. **Chấm công tự động, tức thời (Instant Attendance Check-in):**
   - Loại bỏ các yêu cầu chuyển động phức tạp (nháy mắt/há miệng) khi chấm công hằng ngày.
   - Nhận diện khuôn mặt ngay khi xuất hiện trước ống kính camera thông qua so khớp song song 3 góc mặt mẫu.
   - Tốc độ nhận dạng và ghi nhận chấm công siêu tốc (dưới 0.2 giây).
4. **Chuẩn hóa vectơ độc lập tỷ lệ (Aspect Ratio Invariance):**
   - Landmark được chuyển về không gian pixel thực tế của camera để loại bỏ hoàn toàn sự méo dạng do tỷ lệ khung hình camera (ví dụ camera 16:9 vs 4:3) hoặc khoảng cách xa gần.
5. **Chống nhiễu & Cooldown:**
   - Chỉ ghi nhận chấm công khi khớp khuôn mặt ổn định trong 3 khung hình liên tiếp.
   - Cơ chế cooldown chống ghi trùng lặp chấm công (mặc định 15 phút).

---

## 🛠️ Cấu trúc dự án

```text
├── main.py                 # File chạy chính: luồng camera, đăng ký & điểm danh
├── delete_student.py       # Công cụ CLI nhanh để xóa dữ liệu học sinh khỏi DB
├── .env                    # Lưu trữ cấu hình kết nối SQL Server (IP, Port, User, Pass)
├── requirements.txt        # Danh sách các thư viện Python cần thiết
├── core/
│   ├── anti_fake.py        # Thư viện chuẩn hóa vectơ landmarks & trích xuất liveness (mắt, miệng, yaw, pitch)
│   ├── compare_face.py     # Script kiểm tra/so khớp khuôn mặt tĩnh từ ảnh mẫu
│   └── ...                 # Các file tiện ích bổ trợ nhận diện khác
```

---

## 💾 Thiết kế Cơ sở dữ liệu (SQL Server)

Hệ thống sử dụng cơ sở dữ liệu SQL Server với hai bảng chính:

### 1. Bảng lưu trữ mẫu khuôn mặt (`vectormathocsinh`)
Dùng để lưu thông tin định danh học sinh và chuỗi vectơ đặc trưng của 3 góc mặt.
```sql
CREATE TABLE vectormathocsinh (
    mahocsinh NVARCHAR(50) PRIMARY KEY,
    tenhocsinh NVARCHAR(255) NOT NULL,
    facevector NVARCHAR(MAX) NOT NULL -- Chứa JSON 3 góc mặt {"straight": [...], "left": [...], "right": [...]}
);
```

### 2. Bảng nhật ký điểm danh (`thongtinchamcong`)
Dùng để ghi lại lịch sử điểm danh của học sinh theo thời gian thực.
```sql
CREATE TABLE thongtinchamcong (
    id INT IDENTITY(1,1) PRIMARY KEY,
    mahocsinh NVARCHAR(50) FOREIGN KEY REFERENCES vectormathocsinh(mahocsinh),
    thoigian DATETIME DEFAULT GETDATE()
);
```

---

## 🚀 Hướng dẫn cài đặt & Chạy ứng dụng

### 1. Cài đặt môi trường Python
Yêu cầu Python từ `3.8` trở lên (Khuyến nghị sử dụng Conda để quản lý thư viện dễ dàng):
```bash
# Kích hoạt môi trường (ví dụ môi trường pmcc của bạn)
conda activate pmcc

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Cấu hình biến môi trường
Tạo file `.env` tại thư mục gốc của dự án với các thông tin kết nối tới SQL Server của bạn:
```env
DB_SERVER=127.0.0.1
DB_PORT=1433
DB_USER=sa
DB_PASSWORD=Mật_khẩu_SQL_Server_của_bạn
DB_DATABASE=chamcongdatabase
```

### 3. Chạy ứng dụng Chấm công
Khởi động ứng dụng bằng lệnh:
```bash
python main.py
```

- **Chấm công:** Chỉ cần đứng trước camera, hệ thống sẽ tự động quét khuôn mặt, tính toán khoảng cách khớp (Min Dist) và tự động thực hiện ghi log điểm danh vào SQL Server nếu Min Dist < `0.13`.
- **Đăng ký học sinh mới:**
  1. Nhấn phím **`r`** trên màn hình camera.
  2. Nhập `Mã học sinh` và `Tên học sinh` trên terminal.
  3. Làm theo hướng dẫn trên khung camera:
     - **Bước 1/3 (Nhìn thẳng & Nháy mắt):** Nhìn thẳng vào camera và **nháy mắt** (nhắm mắt và mở ra) để vượt qua bộ lọc chống giả mạo bằng ảnh tĩnh.
     - **Bước 2/3 (Quay trái):** Quay nhẹ đầu sang trái (`yaw < -0.15`).
     - **Bước 3/3 (Quay phải):** Quay nhẹ đầu sang phải (`yaw > 0.15`).
  4. Hệ thống sẽ lưu mẫu 3 góc mặt vào cơ sở dữ liệu và tự động tải lại bộ nhớ RAM để sẵn sàng chấm công.
  - *Lưu ý: Nếu quá 20 giây mà chưa hoàn tất quét 3 góc mặt, hệ thống sẽ tự động hủy phiên đăng ký hiện tại và quay về màn hình nhập thông tin học sinh.*
- **Hủy phiên đăng ký:** Nhấn phím **`c`** trên cửa sổ camera để hủy luồng đăng ký bất kỳ lúc nào.
- **Thoát chương trình:** Nhấn phím **`q`** trên cửa sổ camera.

### 4. Xóa thông tin học sinh
Khi muốn xóa sạch dữ liệu khuôn mặt của một học sinh khỏi hệ thống, chạy script CLI:
```bash
python delete_student.py
```
Nhập mã học sinh (ví dụ `hs001`) để tiến hành xóa bản ghi trên cơ sở dữ liệu.
