import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from datetime import datetime

# 1. Khởi tạo Mediapipe Face Mesh (Tận dụng GPU Apple M1 Pro qua Metal)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1, 
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Cấu hình đường dẫn
DB_PATH = "employee_db"
LOG_EXCEL = "cham_cong.xlsx"

# Đảm bảo thư mục cơ sở dữ liệu tồn tại
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

def get_face_vector(image_path):
    """Trích xuất tọa độ 478 điểm landmark từ ảnh"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Chuyển màu sang RGB cho Mediapipe
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)
    
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0]
        # Trả về mảng numpy (478, 3)
        return np.array([[v.x, v.y, v.z] for v in lm.landmark])
    return None

def identify_person(target_vec, threshold=0.06):
    """So sánh vector mới với toàn bộ database trong employee_db"""
    # Lấy danh sách file hợp lệ (bỏ qua file ẩn hệ thống như .DS_Store)
    files = [f for f in os.listdir(DB_PATH) if not f.startswith('.')]
    
    if not files:
        return "Empty_DB", float('inf')
    
    best_match = "Unknown"
    min_dist = float('inf')

    for file in files:
        file_path = os.path.join(DB_PATH, file)
        try:
            # Load dữ liệu mẫu và reshape về đúng định dạng
            saved_vec = np.loadtxt(file_path, delimiter=',')
            saved_vec = saved_vec.reshape(478, 3)
            
            # Tính khoảng cách Euclidean trung bình giữa các điểm
            dist = np.mean(np.linalg.norm(target_vec - saved_vec, axis=1))
            
            if dist < min_dist:
                min_dist = dist
                if dist < threshold:
                    # Lấy tên file làm tên nhân viên (bỏ đuôi file)
                    best_match = os.path.splitext(file)[0]
        except Exception as e:
            print(f"Lỗi đọc file {file}: {e}")
                
    return best_match, min_dist

def save_to_excel(name_info, score):
    """Lưu dữ liệu vào file Excel"""
    # Tách mã nhân viên và tên nếu đặt tên file kiểu NV01_SonTung
    parts = name_info.split('_')
    msnv = parts[0] if len(parts) > 1 else "N/A"
    ten = parts[1] if len(parts) > 1 else name_info
    
    new_record = {
        'Ngày Giờ': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'MSNV': [msnv],
        'Tên Nhân Viên': [ten],
        'Sai lệch': [round(score, 5)]
    }
    df_new = pd.DataFrame(new_record)
    
    try:
        if not os.path.exists(LOG_EXCEL):
            df_new.to_excel(LOG_EXCEL, index=False)
        else:
            df_old = pd.read_excel(LOG_EXCEL)
            df_final = pd.concat([df_old, df_new], ignore_index=True)
            df_final.to_excel(LOG_EXCEL, index=False)
        return True
    except Exception as e:
        print(f"Lỗi ghi Excel: {e}")
        return False

if __name__ == "__main__":
    # Tên các tệp ảnh của bạn
    img_mau = "anhsontungso1.jpeg"
    img_test = "anhsontungso2.jpeg"
    
    # --- TỰ ĐỘNG ĐĂNG KÝ NẾU DATABASE TRỐNG ---
    if not os.listdir(DB_PATH):
        print(f"[*] Database trống. Đang lấy {img_mau} làm dữ liệu gốc...")
        v_mau = get_face_vector(img_mau)
        if v_mau is not None:
            np.savetxt(os.path.join(DB_PATH, "NV01_SonTung.txt"), v_mau.flatten(), delimiter=',')
            print("✅ Đã tạo file mẫu thành công.")
        else:
            print(f"❌ Không tìm thấy file {img_mau} để làm mẫu.")

    # --- TIẾN HÀNH NHẬN DIỆN ---
    print(f"[*] Đang quét khuôn mặt: {img_test}")
    current_vec = get_face_vector(img_test)

    if current_vec is not None:
        name_id, score = identify_person(current_vec)
        
        if name_id not in ["Unknown", "Empty_DB"]:
            print(f"✅ NHẬN DIỆN THÀNH CÔNG: {name_id}")
            print(f"📊 Độ sai lệch: {score:.5f}")
            if save_to_excel(name_id, score):
                print(f"💾 Đã ghi nhận vào file {LOG_EXCEL}")
        else:
            msg = "DATABASE CHƯA CÓ DỮ LIỆU" if name_id == "Empty_DB" else "KHÔNG KHỚP NHÂN VIÊN"
            print(f"❌ {msg} (Sai lệch thấp nhất: {score:.5f})")
    else:
        print(f"⚠️ Không tìm thấy khuôn mặt trong file {img_test}.")