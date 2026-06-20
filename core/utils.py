import numpy as np

def normalize_vector(vec, w=640, h=480):
    """
    Chuẩn hóa vector landmarks khuôn mặt để đạt được Translation, Scale & Aspect Ratio Invariance.
    Dịch chuyển tâm về gốc tọa độ (0,0,0) và chuẩn hóa L2 về vectơ đơn vị (Unit Vector) phẳng 1434 chiều trong không gian pixel.
    """
    vec_pixel = vec.copy()
    vec_pixel[:, 0] *= w
    vec_pixel[:, 1] *= h
    vec_pixel[:, 2] *= w  # Trục Z tỷ lệ theo chiều rộng để giữ nguyên tỷ lệ không gian 3D
    
    centered = vec_pixel - np.mean(vec_pixel, axis=0)
    flat = centered.flatten()
    norm = np.linalg.norm(flat)
    if norm == 0:
        return flat
    return flat / norm

def extract_liveness_features(face_landmarks):
    """
    Trích xuất độ mở mắt, miệng và hướng đầu (yaw, pitch) ở dạng 2D phục vụ nháy mắt đăng ký.
    """
    coords = np.array([[v.x, v.y] for v in face_landmarks.landmark])
    dist = lambda p1, p2: np.linalg.norm(coords[p1] - coords[p2])
    
    # Khoảng cách giữa 2 khóe mắt trong (inner corners) để làm chuẩn tỉ lệ
    eye_dist = dist(133, 362)
    if eye_dist == 0:
        return [0.0] * 5
        
    # Độ mở mắt và miệng
    eye_l = dist(159, 145) / eye_dist
    eye_r = dist(386, 374) / eye_dist
    mouth_openness = dist(13, 14) / eye_dist
    
    # Ước lượng hướng đầu:
    # 1. Yaw (Xoay đầu trái/phải): Dùng khoảng cách khóe mắt ngoài 33 và 263
    dx = dist(33, 263)
    if dx == 0:
        yaw = 0.0
    else:
        mid_x = (coords[33][0] + coords[263][0]) / 2.0
        yaw = (coords[4][0] - mid_x) / dx
        
    # 2. Pitch (Cúi/ngửa đầu): Dùng đỉnh trán 10 và cằm 152
    dy = dist(10, 152)
    if dy == 0:
        pitch = 0.0
    else:
        mid_y = (coords[10][1] + coords[152][1]) / 2.0
        pitch = (coords[4][1] - mid_y) / dy
        
    return [eye_l, eye_r, mouth_openness, yaw, pitch]

def check_depth_liveness(face_landmarks, threshold=0.015):
    """
    Kiểm tra độ sâu 3D (Lớp 1): Trích xuất độ lệch chuẩn (Standard Deviation)
    trên trục Z của các điểm mốc cứng trên hộp sọ. Ảnh 2D hoặc màn hình sẽ có độ lệch chuẩn cực nhỏ.
    """
    # 10: Trán, 152: Cằm, 4: Đầu mũi, 133: Khóe mắt trong trái, 362: Khóe mắt trong phải
    rigid_ids = [10, 152, 4, 133, 362]
    z_coords = np.array([face_landmarks.landmark[idx].z for idx in rigid_ids])
    z_std = np.std(z_coords)
    return z_std > threshold, float(z_std)

class temporal_motion_tracker:
    """
    Bộ đệm chuyển động thời gian (Lớp 2): Quét liên tiếp 3 khung hình,
    tính chuyển động trung bình của các điểm landmark chính đã được chuẩn hóa qua ICD.
    Khử nhiễu cảm biến camera yếu và phát hiện chuyển động sinh học hợp lệ.
    """
    def __init__(self, buffer_size=3, noise_threshold=0.002, human_threshold=0.015):
        self.buffer_size = buffer_size
        self.noise_threshold = noise_threshold
        self.human_threshold = human_threshold
        self.buffer = []

    def update_and_check(self, face_landmarks):
        # 4: Đầu mũi, 133: Khóe mắt trái, 362: Khóe mắt phải, 13: Môi trên, 14: Môi dưới
        key_ids = [4, 133, 362, 13, 14]
        coords = np.array([[face_landmarks.landmark[idx].x, 
                            face_landmarks.landmark[idx].y, 
                            face_landmarks.landmark[idx].z] for idx in key_ids])
        
        # Khoảng cách giữa 2 khóe mắt trong (Inner Canthus Distance)
        icd = np.linalg.norm(coords[1] - coords[2])
        if icd == 0:
            icd = 1.0
            
        # Chuẩn hóa tọa độ 3D theo tỷ lệ ICD
        normalized_coords = coords / icd
        
        self.buffer.append(normalized_coords)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            
        if len(self.buffer) < self.buffer_size:
            return False, 0.0
            
        total_motion = 0.0
        for i in range(1, len(self.buffer)):
            diff = np.linalg.norm(self.buffer[i] - self.buffer[i-1], axis=1)
            total_motion += np.mean(diff)
            
        avg_motion = total_motion / (len(self.buffer) - 1)
        
        # Kiểm tra điều kiện chuyển động sinh học và khử nhiễu
        if avg_motion < self.noise_threshold:
            return False, float(avg_motion)
        elif avg_motion > self.human_threshold:
            return True, float(avg_motion)
        else:
            return False, float(avg_motion)
