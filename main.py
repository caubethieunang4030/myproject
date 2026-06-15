import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp
import numpy as np
import os
import time
import csv
from datetime import datetime
import json
import pymssql
from dotenv import load_dotenv
import random
# pyrefly: ignore [missing-import]
from core.anti_fake import normalize_vector, extract_liveness_features
# Tải cấu hình từ file .env
load_dotenv()

# Định nghĩa màu sắc ANSI phục vụ hiển thị terminal đẹp mắt
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Khởi tạo Face Mesh từ mediapipe chuyên biệt cho VIDEO stream (static_image_mode=False)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Cấu hình kết nối SQL Server (đọc từ file .env)
db_config = {
    'server': os.getenv('DB_SERVER', '127.0.0.1'),
    'port': int(os.getenv('DB_PORT', 1433)),
    'user': os.getenv('DB_USER', 'sa'),
    'password': os.getenv('DB_PASSWORD', 'DuyAnhMs2026!'),
    'database': os.getenv('DB_DATABASE', 'chamcongdatabase')
}

THRESHOLD = 0.25  # Ngưỡng tối ưu cho mảng vector đã được chuẩn hóa theo khoảng cách mắt (eye-distance normalized)

# Quản lý cooldown ghi log chấm công (15 phút = 900 giây)
LOG_COOLDOWN_SECONDS = 900
last_logged_time = {} # Lưu {name: timestamp}

def load_database():
    """
    Tải cơ sở dữ liệu khuôn mặt từ bảng vectormathocsinh của SQL Server.
    Cấu trúc trả về: {mahocsinh: {"name": tenhocsinh, "vectors": [np_array, ...]}}
    """
    database = {}
    conn = None
    try:
        conn = pymssql.connect(
            server=db_config['server'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        cursor = conn.cursor()
        
        # câu lệnh sql viết thường toàn bộ tên bảng và tên cột
        sql_query = "select mahocsinh, tenhocsinh, facevector from vectormathocsinh"
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        
        for row in rows:
            ma_hs, ten_hs, facevector_str = row
            try:
                data = json.loads(facevector_str)
                vectors = []
                if isinstance(data, dict):
                    # Thiết kế mới: chứa các góc xoay đầu khác nhau
                    for key in ["straight", "left", "right"]:
                        if key in data:
                            vec = np.array(data[key], dtype=np.float32)
                            if vec.size == 1434:
                                vectors.append(normalize_vector(vec.reshape(478, 3)))
                else:
                    # Thiết kế cũ: chỉ chứa 1 vector thẳng
                    vec = np.array(data, dtype=np.float32)
                    if vec.size == 1434:
                        vectors.append(normalize_vector(vec.reshape(478, 3)))
                
                if vectors:
                    database[ma_hs] = {
                        "name": ten_hs,
                        "vectors": vectors
                    }
            except Exception as ex:
                print(f"❌ lỗi khi phân giải vector cho học sinh {ten_hs} ({ma_hs}): {ex}")
                
    except Exception as e:
        print(f"❌ lỗi khi kết nối database để tải vector: {e}")
    finally:
        if conn:
            conn.close()
    return database

def luu_vector_hoc_sinh(ma_hs, ten_hs, mang_vector, overwrite=False):
    """
    Lưu thông tin học sinh và mảng vector mẫu vào bảng vectormathocsinh của SQL Server.
    Nếu overwrite=True, tiến hành cập nhật bản ghi nếu mã học sinh đã tồn tại.
    """
    conn = None
    try:
        conn = pymssql.connect(
            server=db_config['server'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        cursor = conn.cursor()
        
        # chuyển mảng vector số thực thành chuỗi text json sạch để lưu vào nvarchar(max)
        chuoi_vector = json.dumps(mang_vector)
        
        # Kiểm tra xem mã học sinh đã tồn tại trong DB chưa
        cursor.execute("select count(*) from vectormathocsinh where mahocsinh = %s", (ma_hs,))
        exists = cursor.fetchone()[0] > 0
        
        if exists:
            if not overwrite:
                print(f"❌ lỗi: Mã học sinh '{ma_hs}' đã tồn tại trong database.")
                return False
            
            # Câu lệnh cập nhật viết thường toàn bộ tên bảng và tên cột
            sql_query = """
                update vectormathocsinh 
                set tenhocsinh = %s, facevector = %s 
                where mahocsinh = %s
            """
            cursor.execute(sql_query, (ten_hs, chuoi_vector, ma_hs))
            conn.commit()
            print(f"🚀 đã cập nhật thành công vector cho học sinh: {ten_hs}")
            return True
        else:
            # Câu lệnh chèn viết thường toàn bộ tên bảng và tên cột
            sql_query = """
                insert into vectormathocsinh (mahocsinh, tenhocsinh, facevector) 
                values (%s, %s, %s)
            """
            cursor.execute(sql_query, (ma_hs, ten_hs, chuoi_vector))
            conn.commit()
            print(f"🚀 đã lưu thành công vector cho học sinh: {ten_hs}")
            return True
    except Exception as e:
        print(f"❌ lỗi khi lưu vector: {e}")
        return False
    finally:
        if conn:
            conn.close()

def ghi_nhan_cham_cong(ma_hs):
    """
    Ghi nhận check-in vào bảng thongtinchamcong của SQL Server.
    """
    conn = None
    try:
        conn = pymssql.connect(
            server=db_config['server'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        cursor = conn.cursor()
        
        # câu lệnh sql viết thường toàn bộ tên bảng và tên cột
        sql_query = "insert into thongtinchamcong (mahocsinh) values (%s)"
        
        cursor.execute(sql_query, (ma_hs,))
        conn.commit()
        print(f"✅ điểm danh thành công cho mã học sinh: {ma_hs}")
        return True
    except Exception as e:
        print(f"❌ lỗi khi ghi nhận chấm công: {e}")
        return False
    finally:
        if conn:
            conn.close()

def main():
    print(f"\n{CYAN}=================================================={RESET}")
    print(f"{CYAN}{BOLD}    HỆ THỐNG CHẤM CÔNG WEBCAM DEMO     {RESET}")
    print(f"{CYAN}=================================================={RESET}")
    
    # Load database khuôn mặt từ SQL Server
    print(f"[*] Đang tải cơ sở dữ liệu khuôn mặt...")
    database = load_database()
    print(f"{GREEN}[OK] Đã tải thành công {len(database)} khuôn mặt.{RESET}")
    
    # Mở camera của laptop
    print(f"\n[*] Đang khởi động camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"{RED}[LỖI] Không thể mở camera laptop. Vui lòng kiểm tra quyền truy cập camera.{RESET}")
        return
        
    print(f"{GREEN}[OK] Camera đã sẵn sàng!{RESET}")
    print(f"{YELLOW}>>> HƯỚNG DẪN ĐIỀU KHIỂN CAM:{RESET}")
    print(f"  - Nhấn phím {BOLD}'q'{RESET} trên cửa sổ camera để {BOLD}THOÁT{RESET}.")
    print(f"  - Nhấn phím {BOLD}'r'{RESET} trên cửa sổ camera để {BOLD}ĐĂNG KÝ KHUÔN MẶT MỚI{RESET} trực tiếp.\n")
    
    # Biến tạm lưu vector khuôn mặt đang được quét để đăng ký khi nhấn phím 'r'
    current_target_vec = None
    
    # Biến theo dõi xác thực liên tiếp chống nhiễu
    last_detected_name = None
    consecutive_count = 0
    
    # Quản lý hiển thị kết quả xác thực / đăng ký
    liveness_status = "idle"     # Trạng thái: "idle", "approved", "rejected"
    liveness_result_time = 0.0   # Lưu thời điểm hiển thị kết quả
    liveness_result_msg = ""     # Thông báo kết quả để hiển thị lên màn hình
    
    # Quản lý đăng ký học sinh mới bằng cách xoay đầu
    register_mode = False
    register_step = "idle"       # "idle", "straight", "left", "right", "complete"
    register_user_id = ""
    register_user_name = ""
    register_overwrite = False
    register_vectors = {}
    register_stable_frames = 0
    register_start_time = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"{RED}[LỖI] Không nhận được khung hình từ camera.{RESET}")
            break
            
        current_time = time.time()
        
        # Kiểm tra timeout 10 giây trong chế độ đăng ký góc mặt
        if register_mode and (current_time - register_start_time > 10.0):
            print(f"\n{RED}[HẾT GIỜ] Đã quá 10 giây mà chưa hoàn tất quét 3 góc mặt. Quay lại luồng nhập thông tin.{RESET}")
            liveness_status = "rejected"
            liveness_result_time = current_time
            liveness_result_msg = "Qua thoi gian (10s)!"
            
            register_mode = False
            register_step = "idle"
            
            # Tự động reprompt yêu cầu nhập lại thông tin học sinh
            print(f"\n{YELLOW}[ĐĂNG KÝ LẠI] Vui lòng nhập lại thông tin học sinh...{RESET}")
            ma_hs = input(f"{YELLOW} Nhập mã học sinh (ví dụ: hs001): {RESET}").strip()
            name_input = input(f"{YELLOW} Nhập tên học sinh: {RESET}").strip()
            if ma_hs and name_input:
                overwrite = False
                if ma_hs in database:
                    existing_name = database[ma_hs]["name"]
                    confirm = input(f"{YELLOW}⚠️ Mã học sinh '{ma_hs}' đã tồn tại (Tên: {existing_name}). Bạn có muốn cập nhật/ghi đè? (y/n): {RESET}").strip().lower()
                    if confirm == 'y':
                        overwrite = True
                    else:
                        print(f"{RED}[HỦY BỎ] Hủy đăng ký để tránh ghi đè dữ liệu.{RESET}\n")
                        continue
                
                # Khởi động lại luồng quét và reset lại thời gian bắt đầu quét
                print(f"{YELLOW}[*] Bắt đầu quét góc mặt. Vui lòng nhìn thẳng vào camera...{RESET}")
                register_mode = True
                register_step = "straight"
                register_user_id = ma_hs
                register_user_name = name_input
                register_overwrite = overwrite
                register_vectors = {}
                register_stable_frames = 0
                register_start_time = time.time()
            else:
                print(f"{RED}[HUỶ ĐĂNG KÝ] Thiếu thông tin mã học sinh hoặc tên học sinh. Huỷ bỏ.{RESET}\n")
            continue
            
        # Lật ngang khung hình để giống hiệu ứng soi gương
        frame = cv2.flip(frame, 1)
        
        # --- ĐOẠN CODE XỬ LÝ THIẾU SÁNG ---
        # Tăng độ tương phản (Alpha) và độ sáng (Beta)
        alpha = 1.3  # Hệ số tương phản (1.0 - 3.0) giúp làm rõ nét các đường biên
        beta = 40    # Giá trị độ sáng cộng thêm (0 - 100) giúp kích sáng phòng tối
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        h, w, _ = frame.shape
        
        # Tối ưu hóa hiệu năng: Chuyển ảnh sang BGR2RGB và đặt flag writable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        current_target_vec = None
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 1. Trích xuất vector 1434 chiều từ 478 landmarks
                target_vec = np.array([[v.x, v.y, v.z] for v in face_landmarks.landmark])
                current_target_vec = target_vec  # Lưu lại vector hiện tại phục vụ đăng ký
                
                # Tính toán bounding box từ các điểm landmarks để vẽ khung quanh khuôn mặt
                x_coords = [lm.x for lm in face_landmarks.landmark]
                y_coords = [lm.y for lm in face_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
                
                padding_x = int((x_max - x_min) * 0.1)
                padding_y = int((y_max - y_min) * 0.1)
                x1 = max(0, x_min - padding_x)
                y1 = max(0, y_min - padding_y)
                x2 = min(w, x_max + padding_x)
                y2 = min(h, y_max + padding_y)
                
                # 2. So sánh và tìm khuôn mặt khớp nhất trong database
                best_match_id = "Unknown"
                best_match_name = "Unknown"
                min_dist = float('inf')
                
                # Chuẩn hóa target_vec trước khi so sánh
                normalized_target_vec = normalize_vector(target_vec)
                
                if register_mode:
                    # --- CHẾ ĐỘ ĐĂNG KÝ (THU THẬP NHIỀU GÓC MẶT) ---
                    features = extract_liveness_features(face_landmarks)
                    _, _, _, yaw, pitch = features
                    
                    color = (255, 191, 0)  # Màu xanh lam sáng (Cyan)
                    
                    if register_step == "straight":
                        if abs(yaw) < 0.08:
                            register_stable_frames += 1
                            if register_stable_frames >= 10:
                                register_vectors["straight"] = target_vec.flatten().tolist()
                                register_step = "left"
                                register_stable_frames = 0
                                print(f"{GREEN}[OK] Da chup va luu goc THANG cho hoc sinh: {register_user_name}{RESET}")
                        else:
                            register_stable_frames = 0
                        
                        label = f"[1/3] DANG KY: NHIN THANG ({register_stable_frames}/10)"
                        cv2.putText(frame, "NHIN THANG VAO CAMERA", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                        
                    elif register_step == "left":
                        if yaw < -0.15:
                            register_stable_frames += 1
                            if register_stable_frames >= 5:
                                register_vectors["left"] = target_vec.flatten().tolist()
                                register_step = "right"
                                register_stable_frames = 0
                                print(f"{GREEN}[OK] Da chup va luu goc TRAI cho hoc sinh: {register_user_name}{RESET}")
                        else:
                            register_stable_frames = 0
                        
                        label = f"[2/3] DANG KY: QUAY TRAI ({register_stable_frames}/5)"
                        cv2.putText(frame, "QUAY DAU SANG TRAI (<-)", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                        
                    elif register_step == "right":
                        if yaw > 0.15:
                            register_stable_frames += 1
                            if register_stable_frames >= 5:
                                register_vectors["right"] = target_vec.flatten().tolist()
                                register_step = "complete"
                                register_stable_frames = 0
                                print(f"{GREEN}[OK] Da chup va luu goc PHAI cho hoc sinh: {register_user_name}{RESET}")
                        else:
                            register_stable_frames = 0
                        
                        label = f"[3/3] DANG KY: QUAY PHAI ({register_stable_frames}/5)"
                        cv2.putText(frame, "QUAY DAU SANG PHAI (->)", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                    
                    if register_step == "complete":
                        print(f"[*] Dang luu vector 3 goc cua {register_user_name} vao SQL Server...")
                        if luu_vector_hoc_sinh(register_user_id, register_user_name, register_vectors, overwrite=register_overwrite):
                            database = load_database()
                            print(f"{GREEN}[THÀNH CÔNG] Đăng ký thành công học sinh: {register_user_name}{RESET}\n")
                            liveness_status = "approved"
                            liveness_result_time = time.time()
                            liveness_result_msg = f"Chao {register_user_name}! (Dang ky thanh cong)"
                        else:
                            print(f"{RED}[LỖI] Đăng ký thất bại do không luu duoc vao database.{RESET}\n")
                            liveness_status = "rejected"
                            liveness_result_time = time.time()
                            liveness_result_msg = "Dang ky that bai!"
                            
                        register_mode = False
                        register_step = "idle"
                        color = (0, 255, 0) if liveness_status == "approved" else (0, 0, 255)
                        label = liveness_result_msg
                    
                    # Vẽ bounding box và nhãn đăng ký
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(frame, (x1, y1 - 25), (x2, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                else:
                    # --- CHẾ ĐỘ CHẤM CÔNG THƯỜNG (NHẬN DIỆN MỌI GÓC MẶT VÀ CHẤM CÔNG TỨC THỜI) ---
                    best_match_id = "Unknown"
                    best_match_name = "Unknown"
                    min_dist = float('inf')
                    
                    # Chuẩn hóa target_vec trước khi so sánh
                    normalized_target_vec = normalize_vector(target_vec)
                    
                    for ma_hs, info in database.items():
                        # Lặp qua tất cả các vector đã lưu cho học sinh này
                        for saved_vec in info["vectors"]:
                            dist = np.mean(np.linalg.norm(normalized_target_vec - saved_vec, axis=1))
                            if dist < min_dist:
                                min_dist = dist
                                if dist < THRESHOLD:
                                    best_match_id = ma_hs
                                    best_match_name = info["name"]
                    
                    # 3. Xử lý hiển thị UI và Ghi nhật ký chấm công tức thì
                    current_time = time.time()
                    
                    # Kiểm tra và xử lý trạng thái hiển thị kết quả cũ
                    if liveness_status in ["approved", "rejected"]:
                        if current_time - liveness_result_time >= 2.0:
                            liveness_status = "idle"
                    
                    if liveness_status == "approved":
                        color = (0, 255, 0)  # Xanh lá (BGR)
                        label = liveness_result_msg
                    elif liveness_status == "rejected":
                        color = (0, 0, 255)  # Đỏ (BGR)
                        label = liveness_result_msg
                    else:
                        # Trạng thái rảnh (idle) - thực hiện nhận diện và chấm công nhanh
                        if best_match_id != "Unknown":
                            # Cập nhật đếm liên tiếp chống nhiễu
                            if best_match_id == last_detected_name:
                                consecutive_count += 1
                            else:
                                last_detected_name = best_match_id
                                consecutive_count = 1
                            
                            if consecutive_count >= 3:
                                in_cooldown = (best_match_id in last_logged_time) and (current_time - last_logged_time[best_match_id] <= LOG_COOLDOWN_SECONDS)
                                if in_cooldown:
                                    # Trong thời gian cooldown chỉ hiện chào thông thường
                                    color = (0, 255, 0)
                                    label = f"Chao {best_match_name}"
                                else:
                                    # Điểm danh TỨC THÌ
                                    ghi_nhan_cham_cong(best_match_id)
                                    last_logged_time[best_match_id] = current_time
                                    liveness_status = "approved"
                                    liveness_result_time = current_time
                                    liveness_result_msg = f"Chao {best_match_name}! (Thanh cong)"
                                    print(f"{GREEN}[OK] Diem danh thanh cong cho {best_match_name}!{RESET}")
                                    
                                    color = (0, 255, 0)
                                    label = liveness_result_msg
                            else:
                                color = (0, 165, 255)
                                label = f"Nhan dang... {best_match_name} ({consecutive_count}/3)"
                        else:
                            last_detected_name = None
                            consecutive_count = 0
                            color = (0, 0, 255)
                            label = "[X] - Unknown"
                    
                    # Vẽ bounding box và nhãn
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(frame, (x1, y1 - 25), (x2, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # Không phát hiện khuôn mặt nào
            last_detected_name = None
            consecutive_count = 0
            if register_mode:
                register_stable_frames = 0
                cv2.putText(frame, "KHONG TIM THAY KHUON MAT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "Nhan 'c' de HUY DANG KY", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        # 4. Hiển thị màn hình camera
        cv2.imshow('Face Recognition Demo', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # 5. Xử lý sự kiện nhấn phím
        # THOÁT khi nhấn phím 'q'
        if key == ord('q'):
            break
            
        # ĐĂNG KÝ khuôn mặt mới trực tiếp khi nhấn phím 'r'
        elif key == ord('r'):
            if current_target_vec is not None:
                print(f"\n{YELLOW}[ĐĂNG KÝ] Phát hiện lệnh đăng ký khuôn mặt.{RESET}")
                ma_hs = input(f"{YELLOW} Nhập mã học sinh (ví dụ: hs001): {RESET}").strip()
                name_input = input(f"{YELLOW} Nhập tên học sinh: {RESET}").strip()
                if ma_hs and name_input:
                    overwrite = False
                    if ma_hs in database:
                        existing_name = database[ma_hs]["name"]
                        confirm = input(f"{YELLOW}⚠️ Mã học sinh '{ma_hs}' đã tồn tại (Tên: {existing_name}). Bạn có muốn cập nhật/ghi đè? (y/n): {RESET}").strip().lower()
                        if confirm == 'y':
                            overwrite = True
                        else:
                            print(f"{RED}[HỦY BỎ] Hủy đăng ký để tránh ghi đè dữ liệu.{RESET}\n")
                            continue
                            
                    # Bắt đầu chế độ đăng ký góc mặt tự động
                    print(f"{YELLOW}[*] Bắt đầu quét góc mặt. Vui lòng nhìn thẳng vào camera...{RESET}")
                    register_mode = True
                    register_step = "straight"
                    register_user_id = ma_hs
                    register_user_name = name_input
                    register_overwrite = overwrite
                    register_vectors = {}
                    register_stable_frames = 0
                    register_start_time = time.time()
                else:
                    print(f"{RED}[HUỶ ĐĂNG KÝ] Thiếu thông tin mã học sinh hoặc tên học sinh. Huỷ bỏ.{RESET}\n")
            else:
                print(f"{RED}[CẢNH BÁO] Không phát hiện khuôn mặt nào trước camera để đăng ký!{RESET}")
                
        # Nhấn phím 'c' để hủy đăng ký khi đang trong chế độ đăng ký
        elif key == ord('c'):
            if register_mode:
                register_mode = False
                register_step = "idle"
                print(f"{RED}[HỦY BỎ] Đã hủy đăng ký học sinh.{RESET}\n")
            
    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{CYAN}=================================================={RESET}")
    print(f"{CYAN}       ĐÃ ĐÓNG CAMERA & KẾT THÚC DỰ ÁN DEMO       {RESET}")
    print(f"{CYAN}=================================================={RESET}\n")

if __name__ == "__main__":
    main()

