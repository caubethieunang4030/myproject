import cv2
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
import threading
from cryptography.fernet import Fernet
import config
from core.utils import normalize_vector, extract_liveness_features, check_depth_liveness, temporal_motion_tracker, batch_cosine_distance
from core.camera_stream import ThreadedCamera
load_dotenv()

GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Khởi tạo Face Mesh từ mediapipe chuyên biệt cho VIDEO stream (static_image_mode=False)
try:
    import mediapipe.solutions.face_mesh as mp_face_mesh
except (AttributeError, ModuleNotFoundError, ImportError):
    try:
        mp_face_mesh = mp.solutions.face_mesh
    except (AttributeError, ModuleNotFoundError, ImportError):
        raise RuntimeError(
            f"\n{RED}❌ LỖI KHÔNG TÌM THẤY MEDIAPIPE SOLUTIONS:{RESET}\n"
            f"Môi trường venv hiện tại đang dùng Python 3.13 với MediaPipe 0.10.35+ (bị bỏ module `solutions`).\n"
            f"👉 {YELLOW}Trên Mac, vui lòng kích hoạt môi trường Conda Python 3.10 có sẵn bằng lệnh:{RESET}\n"
            f"   {CYAN}conda activate pmcc{RESET}\n"
            f"   {CYAN}python main.py{RESET}\n"
        )

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

db_password = os.getenv('DB_PASSWORD')
if not db_password:
    raise ValueError("LỖI: Biến môi trường DB_PASSWORD chưa được cấu hình trong file .env!")

db_config = {
    'server': os.getenv('DB_SERVER', '127.0.0.1'),
    'port': int(os.getenv('DB_PORT', 1433)),
    'user': os.getenv('DB_USER', 'sa'),
    'password': db_password,
    'database': os.getenv('DB_DATABASE', 'chamcongdatabase')
}

# Hàm khởi tạo và lấy công cụ mã hóa AES-256 (Fernet)
def get_encryption_suite():
    key = os.getenv("ENCRYPTION_KEY")
    if not key:
        new_key = Fernet.generate_key().decode()
        env_path = '.env'
        try:
            with open(env_path, 'a') as f:
                f.write(f"\nENCRYPTION_KEY={new_key}\n")
            os.environ["ENCRYPTION_KEY"] = new_key
            key = new_key
            print(f"🔑 Đã tự động tạo khóa mã hóa sinh trắc học và lưu vào .env")
        except Exception as e:
            raise RuntimeError(f"Không thể khởi tạo và ghi ENCRYPTION_KEY vào .env: {e}")
    return Fernet(key.encode() if isinstance(key, str) else key)

cipher_suite = get_encryption_suite()

# Ngưỡng khoảng cách Cosine Distance (độ lệch tối đa 2.5%, tương đương tương đồng CosSim >= 97.5%)
THRESHOLD = 0.025  

# Cấu hình hàng đợi ngoại tuyến (Offline Queue)
OFFLINE_QUEUE_FILE = "offline_queue.json"
offline_lock = threading.Lock()

def upgrade_db_schema():
    """
    Tự động nâng cấp bảng thongtinchamcong để thêm cột loai_chamcong nếu chưa tồn tại.
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
        cursor.execute("""
            IF NOT EXISTS (
                SELECT * FROM sys.columns 
                WHERE object_id = OBJECT_ID('thongtinchamcong') AND name = 'loai_chamcong'
            )
            BEGIN
                ALTER TABLE thongtinchamcong ADD loai_chamcong NVARCHAR(10) DEFAULT 'VAO';
            END
        """)
        conn.commit()
        print(f"🚀 {GREEN}[OK] Đã kiểm tra và đồng bộ cấu trúc bảng thongtinchamcong (cột loai_chamcong).{RESET}")
    except Exception as e:
        print(f"⚠️ {YELLOW}[CẢNH BÁO] Không thể nâng cấp tự động schema DB: {e}.{RESET}")
    finally:
        if conn:
            conn.close()

# Quản lý cooldown ghi log chấm công (15 phút = 900 giây)
LOG_COOLDOWN_SECONDS = 900
last_logged_time = {} # Lưu {name: timestamp}

def load_database(w=640, h=480):
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
                # Giải mã AES-256 dữ liệu sinh trắc học
                try:
                    decrypted_str = cipher_suite.decrypt(facevector_str.encode()).decode()
                    data = json.loads(decrypted_str)
                except Exception:
                    # Tương thích ngược: nếu giải mã lỗi, đọc thẳng dạng plain text JSON
                    print(f"⚠️ {RED}[SECURITY] HS {ma_hs}: Đọc vector dạng plaintext - Có thể dữ liệu cũ chưa mã hóa hoặc đã bị sửa đổi!{RESET}")
                    data = json.loads(facevector_str)
                
                vectors = []
                if isinstance(data, dict):
                    # Thiết kế mới: chứa các góc xoay đầu khác nhau
                    for key in ["straight", "left", "right"]:
                        if key in data:
                            vec = np.array(data[key], dtype=np.float32)
                            if vec.size == 1434:
                                vectors.append(normalize_vector(vec.reshape(478, 3), w, h))
                else:
                    # Thiết kế cũ: chỉ chứa 1 vector thẳng
                    vec = np.array(data, dtype=np.float32)
                    if vec.size == 1434:
                        vectors.append(normalize_vector(vec.reshape(478, 3), w, h))
                
                if vectors:
                    database[ma_hs] = {
                        "name": ten_hs,
                        "vectors": vectors
                    }
            except Exception as ex:
                print(f"lỗi khi phân giải vector cho học sinh {ten_hs} ({ma_hs}): {ex}")
                
    except Exception as e:
        print(f"lỗi khi kết nối database để tải vector: {e}")
    finally:
        if conn:
            conn.close()
    return database

def build_db_matrix(database):
    """
    Biến đổi dict database thành ma trận NumPy (N, 1434) để tăng tốc độ so sánh vector
    trên Raspberry Pi 5 bằng phép nhân ma trận hàng loạt (Batch Matrix Multiplication).
    """
    matrix_rows = []
    mapping = []
    for ma_hs, info in database.items():
        name = info["name"]
        for vec in info["vectors"]:
            matrix_rows.append(vec.flatten())
            mapping.append((ma_hs, name))
    if matrix_rows:
        return np.array(matrix_rows, dtype=np.float32), mapping
    return None, []

def luu_vector_hoc_sinh(ma_hs, ten_hs, mang_vector, overwrite=False):
    """
    Lưu thông tin học sinh và mảng vector mẫu đã mã hóa AES-256 vào SQL Server.
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
        
        # chuyển mảng vector số thực thành chuỗi text json sạch và mã hóa bảo mật sinh trắc
        chuoi_vector = json.dumps(mang_vector)
        chuoi_vector_encrypted = cipher_suite.encrypt(chuoi_vector.encode()).decode()
        
        # Kiểm tra xem mã học sinh đã tồn tại trong DB chưa
        cursor.execute("select count(*) from vectormathocsinh where mahocsinh = %s", (ma_hs,))
        exists = cursor.fetchone()[0] > 0
        
        if exists:
            if not overwrite:
                print(f"lỗi: Mã học sinh '{ma_hs}' đã tồn tại trong database.")
                return False
            
            # Câu lệnh cập nhật viết thường toàn bộ tên bảng và tên cột
            sql_query = """
                update vectormathocsinh 
                set tenhocsinh = %s, facevector = %s 
                where mahocsinh = %s
            """
            cursor.execute(sql_query, (ten_hs, chuoi_vector_encrypted, ma_hs))
            conn.commit()
            print(f"🚀 đã cập nhật thành công vector cho học sinh: {ten_hs}")
            return True
        else:
            # Câu lệnh chèn viết thường toàn bộ tên bảng và tên cột
            sql_query = """
                insert into vectormathocsinh (mahocsinh, tenhocsinh, facevector) 
                values (%s, %s, %s)
            """
            cursor.execute(sql_query, (ma_hs, ten_hs, chuoi_vector_encrypted))
            conn.commit()
            print(f"🚀 đã lưu thành công vector cho học sinh: {ten_hs}")
            return True
    except Exception as e:
        print(f"lỗi khi lưu vector: {e}")
        return False
    finally:
        if conn:
            conn.close()

def save_offline_log(ma_hs, loai_chamcong):
    """
    Lưu log điểm danh ngoại tuyến vào file JSON cục bộ khi mất kết nối DB.
    """
    record = {
        "mahocsinh": ma_hs,
        "loai_chamcong": loai_chamcong,
        "thoigian": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with offline_lock:
        data = []
        if os.path.exists(OFFLINE_QUEUE_FILE):
            try:
                with open(OFFLINE_QUEUE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                data = []
        data.append(record)
        try:
            with open(OFFLINE_QUEUE_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"⚠️ {YELLOW}[OFFLINE] Đã lưu điểm danh ngoại tuyến cho {ma_hs} ({loai_chamcong}) vào file local.{RESET}")
        except Exception as e:
            print(f"❌ Không thể lưu file offline: {e}")

def sync_offline_queue():
    """
    Hàm luồng chạy nền tự động đồng bộ hàng đợi ngoại tuyến lên SQL Server khi có mạng trở lại.
    Tránh race condition bằng cách đổi tên file sang file tạm trong lock trước khi đồng bộ.
    """
    SYNC_TEMP_FILE = "offline_queue_sync.json"
    while True:
        time.sleep(10) # Kiểm tra mỗi 10 giây
        
        if not os.path.exists(OFFLINE_QUEUE_FILE) and not os.path.exists(SYNC_TEMP_FILE):
            continue
            
        has_records = False
        with offline_lock:
            try:
                # Nếu file tạm cũ vẫn tồn tại từ phiên chạy trước (ví dụ do chương trình crash đột ngột), merge lại vào file chính
                if os.path.exists(SYNC_TEMP_FILE):
                    s_rec, o_rec = [], []
                    try:
                        with open(SYNC_TEMP_FILE, 'r', encoding='utf-8') as sf:
                            s_rec = json.load(sf)
                    except Exception:
                        s_rec = []
                    try:
                        with open(OFFLINE_QUEUE_FILE, 'r', encoding='utf-8') as of:
                            o_rec = json.load(of)
                    except Exception:
                        o_rec = []
                    
                    merged = o_rec + s_rec
                    if merged:
                        with open(OFFLINE_QUEUE_FILE, 'w', encoding='utf-8') as of:
                            json.dump(merged, of, ensure_ascii=False, indent=4)
                    
                    try:
                        os.remove(SYNC_TEMP_FILE)
                    except Exception:
                        pass
                
                if os.path.exists(OFFLINE_QUEUE_FILE):
                    os.rename(OFFLINE_QUEUE_FILE, SYNC_TEMP_FILE)
                    has_records = True
            except Exception as e:
                print(f"Lỗi khi đổi tên file tạm đồng bộ: {e}")
                has_records = False
                
        if not has_records:
            continue
            
        # Đọc dữ liệu từ file tạm (không cần lock vì file tạm này không bị thread chính ghi đè)
        records = []
        try:
            with open(SYNC_TEMP_FILE, 'r', encoding='utf-8') as f:
                records = json.load(f)
        except Exception:
            try:
                os.remove(SYNC_TEMP_FILE)
            except Exception:
                pass
            continue
            
        if not records:
            try:
                os.remove(SYNC_TEMP_FILE)
            except Exception:
                pass
            continue
            
        # Thử kết nối tới SQL Server
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
            print(f"📡 {GREEN}[ĐỒNG BỘ] Đã khôi phục kết nối database. Đang đồng bộ {len(records)} log ngoại tuyến...{RESET}")
            
            success_count = 0
            for record in records:
                sql_query = "insert into thongtinchamcong (mahocsinh, loai_chamcong, thoigianquet) values (%s, %s, %s)"
                cursor.execute(sql_query, (record["mahocsinh"], record["loai_chamcong"], record["thoigian"]))
                success_count += 1
                
            conn.commit()
            print(f"🚀 {GREEN}[ĐỒNG BỘ] Đồng bộ thành công {success_count}/{len(records)} bản ghi.{RESET}")
            
            # Xóa file tạm sau khi đồng bộ thành công
            try:
                os.remove(SYNC_TEMP_FILE)
            except Exception:
                pass
        except Exception as e:
            # Gặp lỗi DB, merge ngược dữ liệu từ file tạm vào file chính dưới lock để bảo toàn dữ liệu
            with offline_lock:
                try:
                    main_records = []
                    if os.path.exists(OFFLINE_QUEUE_FILE):
                        try:
                            with open(OFFLINE_QUEUE_FILE, 'r', encoding='utf-8') as f:
                                main_records = json.load(f)
                        except Exception:
                            main_records = []
                    
                    merged_records = records + main_records
                    with open(OFFLINE_QUEUE_FILE, 'w', encoding='utf-8') as f:
                        json.dump(merged_records, f, ensure_ascii=False, indent=4)
                    
                    os.remove(SYNC_TEMP_FILE)
                except Exception as ex:
                    print(f"Lỗi khi hoàn tác merge records ngoại tuyến: {ex}")
        finally:
            if conn:
                conn.close()

def ghi_nhan_cham_cong(ma_hs, loai_chamcong="VAO"):
    """
    Ghi nhận check-in/out vào bảng thongtinchamcong của SQL Server.
    Nếu mất kết nối, tự động lưu ngoại tuyến.
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
        
        # Câu lệnh SQL viết thường toàn bộ tên bảng và tên cột, chèn thêm loai_chamcong
        sql_query = "insert into thongtinchamcong (mahocsinh, loai_chamcong) values (%s, %s)"
        cursor.execute(sql_query, (ma_hs, loai_chamcong))
        conn.commit()
        print(f"✅ {GREEN}Điểm danh ({loai_chamcong}) thành công cho học sinh: {ma_hs}{RESET}")
        return True
    except Exception as e:
        print(f"⚠️ {RED}Lỗi khi kết nối SQL Server: {e}. Tiến hành lưu log ngoại tuyến...{RESET}")
        save_offline_log(ma_hs, loai_chamcong)
        return False
    finally:
        if conn:
            conn.close()

def main():
    print(f"\n{CYAN}=================================================={RESET}")
    print(f"{CYAN}{BOLD}    HỆ THỐNG CHẤM CÔNG WEBCAM DEMO     {RESET}")
    print(f"{CYAN}=================================================={RESET}")
    
    # Tự động đồng bộ nâng cấp schema database
    upgrade_db_schema()
    
    # Khởi động luồng chạy nền đồng bộ chấm công ngoại tuyến
    sync_thread = threading.Thread(target=sync_offline_queue, daemon=True)
    sync_thread.start()
    
    # Mở camera đa luồng chuyên biệt cho Raspberry Pi 5 & Logitech Brio 100
    cam_index = int(os.getenv('CAMERA_INDEX', '1'))
    print(f"\n[*] Đang khởi động ThreadedCamera (Index={cam_index} - Logitech Brio 100)...")
    cam_stream = ThreadedCamera(
        src=cam_index,
        width=config.DISPLAY_WIDTH,
        height=config.DISPLAY_HEIGHT,
        fps=config.DISPLAY_FPS
    ).start()
    
    print(f"{GREEN}[OK] ThreadedCamera đã sẵn sàng!{RESET}")
    
    # Chờ camera khởi tạo và đọc thử khung hình (chờ tối đa 3.0 giây)
    print(f"[*] Đang chờ luồng hình ảnh từ camera...")
    ret, frame_init = False, None
    for _ in range(30):
        ret, frame_init = cam_stream.read()
        if ret and frame_init is not None:
            break
        time.sleep(0.1)
        
    if not ret or frame_init is None:
        print(f"{RED}[LỖI] Không thể đọc khung hình từ camera.{RESET}")
        print(f"👉 {YELLOW}Các nguyên nhân có thể xảy ra:{RESET}")
        print(f"   1. Chưa cấp quyền Camera cho Terminal/Python trong System Settings > Privacy & Security > Camera.")
        print(f"   2. Camera Index = {cam_index} đang bị chiếm dụng bởi ứng dụng khác (Zoom, Teams, Photo Booth...).")
        print(f"   3. Thử đổi CAMERA_INDEX=0 trong .env để chuyển sang FaceTime HD Camera hoặc cổng USB khác.")
        cam_stream.stop()
        return
    h_disp, w_disp, _ = frame_init.shape
    
    # Load cấu hình tối ưu hóa di động từ config.py / .env
    mobile_opt = os.getenv('MOBILE_OPTIMIZATION', 'True').lower() in ('true', '1', 'yes')
    process_w = config.PROCESS_WIDTH if mobile_opt else w_disp
    process_h = config.PROCESS_HEIGHT if mobile_opt else h_disp
    target_fps = int(os.getenv('TARGET_FPS', '30'))
    headless_mode = os.getenv('HEADLESS_MODE', 'False').lower() in ('true', '1', 'yes')
    
    # Nếu tối ưu hóa di động, dùng kích thước xử lý để đồng bộ hóa chuẩn hóa vector
    db_w = process_w if mobile_opt else w_disp
    db_h = process_h if mobile_opt else h_disp
    
    # Load database khuôn mặt từ SQL Server và dựng ma trận so khớp hàng loạt
    print(f"[*] Đang tải cơ sở dữ liệu khuôn mặt...")
    database = load_database(db_w, db_h)
    db_matrix, db_mapping = build_db_matrix(database)
    print(f"{GREEN}[OK] Đã tải thành công {len(database)} học sinh ({len(db_mapping)} vectors).{RESET}")
    
    print(f"{YELLOW}>>> HƯỚNG DẪN ĐIỀU KHIỂN CAM:{RESET}")
    if not headless_mode:
        print(f"  - Nhấn phím {BOLD}'q'{RESET} trên cửa sổ camera để {BOLD}THOÁT{RESET}.")
        print(f"  - Nhấn phím {BOLD}'r'{RESET} trên cửa sổ camera để {BOLD}ĐĂNG KÝ KHUÔN MẶT MỚI{RESET} trực tiếp.")
        print(f"  - Nhấn phím {BOLD}'i'{RESET} để chọn chế độ chấm công {BOLD}VÀO (Check-In){RESET}.")
        print(f"  - Nhấn phím {BOLD}'o'{RESET} để chọn chế độ chấm công {BOLD}RA (Check-Out){RESET}.\n")
    else:
        print(f"  - Đang chạy ở chế độ KHÔNG MÀN HÌNH (Headless). Nhấn {BOLD}Ctrl+C{RESET} trong terminal để dừng.\n")
    
    # Biến tạm lưu vector khuôn mặt đang được quét để đăng ký khi nhấn phím 'r'
    current_target_vec = None
    
    # Biến theo dõi xác thực liên tiếp chống nhiễu
    last_detected_name = None
    consecutive_count = 0
    
    # Quản lý hiển thị kết quả xác thực / đăng ký
    liveness_status = "idle"     
    liveness_result_time = 0.0 
    liveness_result_msg = ""    
    
    # Quản lý đăng ký học sinh mới bằng cách xoay đầu
    register_mode = False
    register_step = "idle"  
    register_user_id = ""
    register_user_name = ""
    register_overwrite = False
    register_vectors = {}
    register_stable_frames = 0
    register_start_time = 0.0
    register_blink_state = "waiting" # "waiting", "closed", "verified"
    
    # Chế độ chấm công hiện tại (mặc định tự động theo giờ hệ thống)
    current_hour = datetime.now().hour
    active_mode = "VAO" if current_hour < 12 else "RA"
    manual_override = False
    
    register_prompt_active = False
    
    # Bộ theo dõi chuyển động thời gian chống giả mạo sinh học
    motion_tracker = temporal_motion_tracker()
    
    # Biến tối ưu hóa tần suất và lưu trữ hình ảnh hiển thị di động
    last_process_time = 0.0
    cached_face_drawings = []

    def prompt_registration_input():
        nonlocal register_mode, register_step, register_user_id, register_user_name
        nonlocal register_overwrite, register_vectors, register_stable_frames
        nonlocal register_start_time, register_blink_state, register_prompt_active
        nonlocal database
        
        try:
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
                        register_prompt_active = False
                        return
                
                print(f"{YELLOW}[*] Bắt đầu quét góc mặt. Vui lòng nhìn thẳng vào camera...{RESET}")
                register_user_id = ma_hs
                register_user_name = name_input
                register_overwrite = overwrite
                register_vectors = {}
                register_stable_frames = 0
                register_start_time = time.time()
                register_blink_state = "waiting"
                register_step = "straight"
                register_mode = True
            else:
                print(f"{RED}[HUỶ ĐĂNG KÝ] Thiếu thông tin mã học sinh hoặc tên học sinh. Huỷ bỏ.{RESET}\n")
        except Exception as e:
            print(f"{RED}[LỖI] Đăng ký lỗi: {e}{RESET}")
        finally:
            register_prompt_active = False
            if not register_mode:
                register_step = "idle"

    try:
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0.0

        while True:
            ret, frame = cam_stream.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
                
            current_time = time.time()
            
            # Tự động cập nhật chế độ chấm công theo thời gian thực (nếu không có override thủ công)
            if not manual_override:
                active_mode = "VAO" if datetime.now().hour < 12 else "RA"
            
            # Kiểm tra timeout 20 giây trong chế độ đăng ký góc mặt
            if register_mode and (current_time - register_start_time > 20.0):
                print(f"\n{RED}[HẾT GIỜ] Đã quá 20 giây mà chưa hoàn tất quét 3 góc mặt. Quay lại luồng nhập thông tin.{RESET}")
                liveness_status = "rejected"
                liveness_result_time = current_time
                liveness_result_msg = "Qua thoi gian (20s)!"
                
                register_mode = False
                register_step = "idle"
                
                register_prompt_active = True
                threading.Thread(target=prompt_registration_input, daemon=True).start()
                continue
                
            # Lật ngang khung hình để giống hiệu ứng soi gương
            frame = cv2.flip(frame, 1)
            
            # --- ĐOẠN CODE XỬ LÝ THIẾU SÁNG ---
            # Tăng độ tương phản (Alpha) và độ sáng (Beta)
            alpha = 1.3  # Hệ số tương phản (1.0 - 3.0) giúp làm rõ nét các đường biên
            beta = 40    # Giá trị độ sáng cộng thêm (0 - 100) giúp kích sáng phòng tối
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            
            h_disp, w_disp, _ = frame.shape
            
            # Giới hạn tần suất xử lý AI nếu tối ưu di động và không ở chế độ đăng ký
            should_process_face = True
            elapsed = current_time - last_process_time
            if mobile_opt and not register_mode:
                min_interval = 1.0 / target_fps
                if elapsed < min_interval:
                    should_process_face = False
                    
            if should_process_face:
                # Tải kích thước xử lý tùy vào cấu hình
                if mobile_opt:
                    h, w = process_h, process_w
                    frame_proc = cv2.resize(frame, (w, h))
                else:
                    h, w = h_disp, w_disp
                    frame_proc = frame
                    
                # Tối ưu hóa hiệu năng: Chuyển ảnh sang BGR2RGB và đặt flag writable = False
                rgb_frame = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = face_mesh.process(rgb_frame)
                rgb_frame.flags.writeable = True
                
                current_target_vec = None
                cached_face_drawings = []  # Xóa cache vẽ cũ
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # 1. Trích xuất vector 1434 chiều từ 478 landmarks
                        target_vec = np.array([[v.x, v.y, v.z] for v in face_landmarks.landmark])
                        current_target_vec = target_vec  # Lưu lại vector hiện tại phục vụ đăng ký
                        
                        # Tính toán bounding box từ các điểm landmarks để vẽ khung quanh khuôn mặt (theo kích thước hiển thị hiển thị)
                        x_coords = [lm.x for lm in face_landmarks.landmark]
                        y_coords = [lm.y for lm in face_landmarks.landmark]
                        
                        x_min, x_max = int(min(x_coords) * w_disp), int(max(x_coords) * w_disp)
                        y_min, y_max = int(min(y_coords) * h_disp), int(max(y_coords) * h_disp)
                        
                        padding_x = int((x_max - x_min) * 0.1)
                        padding_y = int((y_max - y_min) * 0.1)
                        x1 = max(0, x_min - padding_x)
                        y1 = max(0, y_min - padding_y)
                        x2 = min(w_disp, x_max + padding_x)
                        y2 = min(h_disp, y_max + padding_y)
                        
                        # 2. So sánh và tìm khuôn mặt khớp nhất trong database (dựa trên w, h xử lý)
                        best_match_id = "Unknown"
                        best_match_name = "Unknown"
                        min_dist = float('inf')
                        
                        # Chuẩn hóa target_vec trước khi so sánh, khử sai lệch tỷ lệ camera
                        normalized_target_vec = normalize_vector(target_vec, w, h)
                        
                        # Phân tích liveness (Độ sâu 3D & Chuyển động sinh học) cho chế độ chấm công thường
                        if not register_mode:
                            depth_ok, z_std = check_depth_liveness(face_landmarks)
                            motion_ok, avg_motion = motion_tracker.update_and_check(face_landmarks)
                        else:
                            depth_ok, z_std = True, 0.0
                            motion_ok, avg_motion = True, 0.0
                        
                        if register_mode:
                            # --- CHẾ ĐỘ ĐĂNG KÝ (THU THẬP NHIỀU GÓC MẶT) ---
                            features = extract_liveness_features(face_landmarks)
                            eye_l, eye_r, mouth_openness, yaw, pitch = features
                            
                            color = (255, 191, 0)  # Màu xanh lam sáng (Cyan)
                            
                            if register_step == "straight":
                                if abs(yaw) < 0.08:
                                    # State machine nháy mắt chống giả mạo bằng ảnh tĩnh
                                    if register_blink_state == "waiting":
                                        if eye_l < 0.15 and eye_r < 0.15:
                                            register_blink_state = "closed"
                                            print(f"{YELLOW}[*] Đã nhắm mắt. Vui lòng mở mắt để hoàn tất xác thực...{RESET}")
                                        label = "[1/3] DANG KY: NHAY MAT DE XAC MINH"
                                    elif register_blink_state == "closed":
                                        if eye_l > 0.22 and eye_r > 0.22:
                                            register_blink_state = "verified"
                                            register_stable_frames = 0
                                            print(f"{GREEN}[OK] Xác thực liveness thành công!{RESET}")
                                        label = "[1/3] DANG KY: MO MAT DE TIEP TUC"
                                    elif register_blink_state == "verified":
                                        register_stable_frames += 1
                                        if register_stable_frames >= 5:
                                            register_vectors["straight"] = target_vec.flatten().tolist()
                                            register_step = "left"
                                            register_stable_frames = 0
                                            print(f"{GREEN}[OK] Đã chụp và lưu góc THẲNG cho học sinh: {register_user_name}{RESET}")
                                        label = f"[1/3] DANG KY: NHIN THANG ({register_stable_frames}/5)"
                                else:
                                    if register_blink_state != "verified":
                                        register_blink_state = "waiting"
                                    register_stable_frames = 0
                                    label = "[1/3] DANG KY: NHIN THANG"
                                
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
                            
                            if register_step == "complete":
                                print(f"[*] Dang luu vector 3 goc cua {register_user_name} vao SQL Server...")
                                if luu_vector_hoc_sinh(register_user_id, register_user_name, register_vectors, overwrite=register_overwrite):
                                    database = load_database(db_w, db_h)
                                    db_matrix, db_mapping = build_db_matrix(database)
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
                            
                            # Vẽ ngay lên frame
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.rectangle(frame, (x1, y1 - 25), (x2, y1), color, -1)
                            cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                            
                            # Lưu vào cache
                            cached_face_drawings.append(("box", x1, y1, x2, y2, color, label))
                            
                            # Hướng dẫn vẽ thêm phụ trợ chế độ đăng ký
                            if register_step == "straight" and register_blink_state == "waiting":
                                cv2.putText(frame, "NHAY MAT DE XAC THUC LIVENESS", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                            elif register_step == "straight" and register_blink_state == "closed":
                                cv2.putText(frame, "MO MAT DE HOAN TAT", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                            elif register_step == "straight" and register_blink_state == "verified":
                                cv2.putText(frame, "GIU NGUYEN HUONG NHIN THANG", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                            elif register_step == "straight":
                                cv2.putText(frame, "NHIN THANG VAO CAMERA", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                            elif register_step == "left":
                                cv2.putText(frame, "QUAY DAU SANG TRAI (<-)", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                            elif register_step == "right":
                                cv2.putText(frame, "QUAY DAU SANG PHAI (->)", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                                
                        else:
                            # --- CHẾ ĐỘ CHẤM CÔNG THƯỜNG (SO KHỚP HÀNG LOẠT VỚI BATCH MATRIX MULTIPLICATION) ---
                            best_match_id = "Unknown"
                            best_match_name = "Unknown"
                            min_dist = float('inf')
                            
                            if db_matrix is not None and len(db_matrix) > 0:
                                distances = batch_cosine_distance(normalized_target_vec.flatten(), db_matrix)
                                min_idx = np.argmin(distances)
                                min_dist = distances[min_idx]
                                if min_dist < THRESHOLD:
                                    best_match_id, best_match_name = db_mapping[min_idx]
                                    
                            print(f"[DEBUG] Khop: {best_match_name} (ID: {best_match_id}) | Cos Dist: {min_dist:.4f} | Nguong THRESHOLD: {THRESHOLD}")
                            
                            if liveness_status in ["approved", "rejected"]:
                                if current_time - liveness_result_time >= 2.0:
                                    liveness_status = "idle"
                            
                            if liveness_status == "approved":
                                color = (0, 255, 0)
                                label = liveness_result_msg
                            elif liveness_status == "rejected":
                                color = (0, 0, 255)
                                label = liveness_result_msg
                            else:
                                if best_match_id != "Unknown":
                                    if best_match_id == last_detected_name:
                                        consecutive_count += 1
                                    else:
                                        last_detected_name = best_match_id
                                        consecutive_count = 1
                                        motion_tracker.buffer.clear()
                                    
                                    if consecutive_count >= 3:
                                        if not depth_ok or not motion_ok:
                                            liveness_status = "rejected"
                                            liveness_result_time = current_time
                                            liveness_result_msg = "[X] - GIA MAO"
                                            color = (0, 0, 255)
                                            label = liveness_result_msg
                                            print(f"{RED}[CANH BAO] Phat hien gia mao cho {best_match_name}! Depth OK: {depth_ok} (z_std: {z_std:.4f}), Motion OK: {motion_ok} (avg_motion: {avg_motion:.4f}){RESET}")
                                            consecutive_count = 0
                                            motion_tracker.buffer.clear()
                                        else:
                                            in_cooldown = (best_match_id in last_logged_time) and (current_time - last_logged_time[best_match_id] <= LOG_COOLDOWN_SECONDS)
                                            if in_cooldown:
                                                color = (0, 255, 0)
                                                label = f"Chao {best_match_name}"
                                            else:
                                                ghi_nhan_cham_cong(best_match_id, active_mode)
                                                last_logged_time[best_match_id] = current_time
                                                liveness_status = "approved"
                                                liveness_result_time = current_time
                                                liveness_result_msg = f"Chao {best_match_name}! ({active_mode} OK)"
                                                print(f"{GREEN}[OK] Diem danh ({active_mode}) thanh cong cho {best_match_name}!{RESET}")
                                                consecutive_count = 0
                                                motion_tracker.buffer.clear()
                                                color = (0, 255, 0)
                                                label = liveness_result_msg
                                    else:
                                        color = (0, 165, 255)
                                        label = f"Nhan dang... {best_match_name} ({consecutive_count}/3)"
                                else:
                                    last_detected_name = None
                                    consecutive_count = 0
                                    motion_tracker.buffer.clear()
                                    color = (0, 0, 255)
                                    label = "[X] - Unknown"
                            
                            # Vẽ lên frame
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.rectangle(frame, (x1, y1 - 25), (x2, y1), color, -1)
                            cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                            
                            # Lưu vào cache
                            cached_face_drawings.append(("box", x1, y1, x2, y2, color, label))
                else:
                    # Không phát hiện khuôn mặt nào
                    last_detected_name = None
                    consecutive_count = 0
                    motion_tracker.buffer.clear()
                    if register_mode:
                        register_stable_frames = 0
                        cv2.putText(frame, "KHONG TIM THAY KHUON MAT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, "Nhan 'c' de HUY DANG KY", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cached_face_drawings.append(("text", "KHONG TIM THAY KHUON MAT", (30, 50), 0.6, (0, 0, 255)))
                        cached_face_drawings.append(("text", "Nhan 'c' de HUY DANG KY", (30, 80), 0.5, (0, 0, 255)))
                
                last_process_time = current_time
            else:
                # Vẽ lại các bounding box từ cached vẽ trên khung hình mới khi bỏ qua AI processing
                for drawing in cached_face_drawings:
                    if drawing[0] == "box":
                        _, x1, y1, x2, y2, color, label = drawing
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.rectangle(frame, (x1, y1 - 25), (x2, y1), color, -1)
                        cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    elif drawing[0] == "text":
                        _, text, pos, scale, color = drawing
                        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            
            # Vẽ trạng thái chế độ chấm công hiện tại & chỉ số FPS thực tế
            mode_text = f"CHE DO: {'VAO (Check-In)' if active_mode == 'VAO' else 'RA (Check-Out)'}"
            cv2.putText(frame, mode_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Tính toán và hiển thị FPS thực tế cho Raspberry Pi 5
            fps_counter += 1
            if current_time - fps_start_time >= 1.0:
                current_fps = fps_counter / (current_time - fps_start_time)
                fps_counter = 0
                fps_start_time = current_time
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (w_disp - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Vẽ số lượng điểm danh đang lưu ngoại tuyến nếu có
            if os.path.exists(OFFLINE_QUEUE_FILE):
                try:
                    with open(OFFLINE_QUEUE_FILE, 'r') as f:
                        offline_count = len(json.load(f))
                    if offline_count > 0:
                        cv2.putText(frame, f"OFFLINE QUEUE: {offline_count} recs", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                except Exception:
                    pass
                    
            # Vẽ overlay thông báo nếu đang chờ nhập liệu terminal không đồng bộ
            if register_prompt_active:
                overlay = frame.copy()
                cv2.rectangle(overlay, (30, h_disp // 2 - 40), (w_disp - 30, h_disp // 2 + 40), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                cv2.putText(frame, "NHAP THONG TIN HS TREN TERMINAL CONSOLE...", (45, h_disp // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            
            # 4. Hiển thị màn hình camera (nếu không ở chế độ headless)
            key = 0xFF
            if not headless_mode:
                cv2.imshow('Face Recognition Demo', frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                # Tránh sử dụng 100% CPU khi chạy không có màn hình
                time.sleep(0.05)
            
            # 5. Xử lý sự kiện nhấn phím
            # THOÁT khi nhấn phím 'q' hoặc kết thúc
            if key == ord('q'):
                break
                
            # ĐĂNG KÝ khuôn mặt mới trực tiếp khi nhấn phím 'r'
            elif key == ord('r'):
                if register_mode or register_prompt_active:
                    print(f"⚠️ {YELLOW}[CẢNH BÁO] Hệ thống đang trong tiến trình đăng ký!{RESET}")
                elif current_target_vec is not None:
                    register_prompt_active = True
                    threading.Thread(target=prompt_registration_input, daemon=True).start()
                else:
                    print(f"⚠️ {RED}[CẢNH BÁO] Không phát hiện khuôn mặt nào trước camera để đăng ký!{RESET}")
                    
            # Phím nóng chuyển sang chế độ chấm công VÀO
            elif key == ord('i'):
                active_mode = "VAO"
                manual_override = True
                print(f"🔄 {CYAN}[CHẾ ĐỘ] Đã chuyển sang chấm công: VÀO (Check-In) [Override]{RESET}")
                
            # Phím nóng chuyển sang chế độ chấm công RA
            elif key == ord('o'):
                active_mode = "RA"
                manual_override = True
                print(f"🔄 {CYAN}[CHẾ ĐỘ] Đã chuyển sang chấm công: RA (Check-Out) [Override]{RESET}")
                
            # Nhấn phím 'c' để hủy đăng ký khi đang trong chế độ đăng ký
            elif key == ord('c'):
                if register_mode:
                    register_mode = False
                    register_step = "idle"
                    print(f"{RED}[HỦY BỎ] Đã hủy đăng ký học sinh.{RESET}\n")
    except KeyboardInterrupt:
        print(f"\n{YELLOW}[HỆ THỐNG] Nhận lệnh tắt từ người dùng (Ctrl+C). Đang đóng ứng dụng...{RESET}")

            
    # Giải phóng tài nguyên
    cam_stream.stop()
    cv2.destroyAllWindows()
    print(f"\n{CYAN}=================================================={RESET}")
    print(f"{CYAN}       ĐÃ ĐÓNG CAMERA & KẾT THÚC DỰ ÁN DEMO       {RESET}")
    print(f"{CYAN}=================================================={RESET}\n")

if __name__ == "__main__":
    main()

