import numpy as np
def normalize_vector(vec):
    """
    Chuẩn hóa vector landmarks khuôn mặt để đạt được Translation & Scale Invariance.
    Dịch chuyển tâm về gốc tọa độ (0,0,0) và chia cho khoảng cách giữa hai mắt (inner corners).
    """
    centered = vec - np.mean(vec, axis=0)
    # Khoảng cách giữa 2 khóe mắt trong (landmark 133 và 362)
    eye_dist = np.linalg.norm(vec[133] - vec[362])
    if eye_dist == 0:
        return centered
    return centered / eye_dist

# Cấu hình liveness detection (xác thực chống giả mạo bằng ảnh tĩnh)
LIVENESS_DURATION = 2.0        # Thời gian bắt buộc kiểm tra chuyển động (giây)
LIVENESS_THRESHOLD = 0.05      # Ngưỡng biến thiên tối thiểu để tính là có chuyển động (nháy mắt hoặc mở miệng)

def normalize_vector(vec):
    """
    Chuẩn hóa vector landmarks khuôn mặt để đạt được Translation & Scale Invariance.
    Dịch chuyển tâm về gốc tọa độ (0,0,0) và chia cho khoảng cách giữa hai mắt (inner corners).
    """
    centered = vec - np.mean(vec, axis=0)
    # Khoảng cách giữa 2 khóe mắt trong (landmark 133 và 362)
    eye_dist = np.linalg.norm(vec[133] - vec[362])
    if eye_dist == 0:
        return centered
    return centered / eye_dist

# Cấu hình liveness detection (xác thực chống giả mạo bằng ảnh tĩnh)
LIVENESS_DURATION = 2.0        # Thời gian bắt buộc kiểm tra chuyển động (giây)
LIVENESS_THRESHOLD = 0.05      # Ngưỡng biến thiên tối thiểu để tính là có chuyển động (nháy mắt hoặc mở miệng)

def extract_liveness_features(face_landmarks):
    """
    Trích xuất độ mở mắt, miệng và hướng đầu (yaw, pitch) ở dạng 2D.
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
