import numpy as np

def normalize_vector(vec, w=640, h=480):
    """
    Normalize face-landmark vectors so they are invariant to translation, scale, and aspect ratio.
    Move the center to the origin (0,0,0) and L2-normalize into a flat 1434-D unit vector in pixel space.
    """
    vec_pixel = vec.copy()
    vec_pixel[:, 0] *= w
    vec_pixel[:, 1] *= h
    vec_pixel[:, 2] *= w  # Scale the Z axis by the width so the 3D proportions stay intact
    
    centered = vec_pixel - np.mean(vec_pixel, axis=0)
    flat = centered.flatten()
    norm = np.linalg.norm(flat)
    if norm == 0:
        return flat
    return flat / norm

def extract_liveness_features(face_landmarks):
    """
    Extract eye opening, mouth opening, and head direction (yaw, pitch) in 2D for the enrollment blink.
    """
    coords = np.array([[v.x, v.y] for v in face_landmarks.landmark])
    dist = lambda p1, p2: np.linalg.norm(coords[p1] - coords[p2])
    
    # Distance between the two inner eye corners, used as the scale reference
    eye_dist = dist(133, 362)
    if eye_dist == 0:
        return [0.0] * 5
        
    # Eye opening and mouth opening
    eye_l = dist(159, 145) / eye_dist
    eye_r = dist(386, 374) / eye_dist
    mouth_openness = dist(13, 14) / eye_dist
    
    # Estimate head direction:
    # 1. Yaw (turn left/right): use the distance between outer eye corners 33 and 263
    dx = dist(33, 263)
    if dx == 0:
        yaw = 0.0
    else:
        mid_x = (coords[33][0] + coords[263][0]) / 2.0
        yaw = (coords[4][0] - mid_x) / dx
        
    # 2. Pitch (tilt down/up): use forehead point 10 and chin point 152
    dy = dist(10, 152)
    if dy == 0:
        pitch = 0.0
    else:
        mid_y = (coords[10][1] + coords[152][1]) / 2.0
        pitch = (coords[4][1] - mid_y) / dy
        
    return [eye_l, eye_r, mouth_openness, yaw, pitch]

def check_depth_liveness(face_landmarks, threshold=0.015):
    """
    3D depth check (layer 1): take the standard deviation
    of the Z axis on rigid skull landmarks. A flat photo or a screen has a very small standard deviation.
    """
    # 10: forehead, 152: chin, 4: nose tip, 133: left inner eye corner, 362: right inner eye corner
    rigid_ids = [10, 152, 4, 133, 362]
    z_coords = np.array([face_landmarks.landmark[idx].z for idx in rigid_ids])
    z_std = np.std(z_coords)
    return z_std > threshold, float(z_std)

class temporal_motion_tracker:
    """
    Temporal motion buffer (layer 2): scan 3 frames in a row,
    and average the motion of the key landmarks after they are normalized by the inner-canthus distance.
    Reject noise from a weak camera sensor and accept only plausible biological motion.
    """
    def __init__(self, buffer_size=3, noise_threshold=0.002, human_threshold=0.015):
        self.buffer_size = buffer_size
        self.noise_threshold = noise_threshold
        self.human_threshold = human_threshold
        self.buffer = []

    def update_and_check(self, face_landmarks):
        # 4: nose tip, 133: left eye corner, 362: right eye corner, 13: upper lip, 14: lower lip
        key_ids = [4, 133, 362, 13, 14]
        coords = np.array([[face_landmarks.landmark[idx].x, 
                            face_landmarks.landmark[idx].y, 
                            face_landmarks.landmark[idx].z] for idx in key_ids])
        
        # Distance between the two inner eye corners (inner canthus distance)
        icd = np.linalg.norm(coords[1] - coords[2])
        if icd == 0:
            icd = 1.0
            
        # Normalize 3D coordinates by the inner-canthus distance
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
        
        # Check the biological-motion condition and reject noise
        if avg_motion < self.noise_threshold:
            return False, float(avg_motion)
        elif avg_motion > self.human_threshold:
            return True, float(avg_motion)
        else:
            return False, float(avg_motion)

def batch_cosine_distance(query_vec, candidate_matrix):
    """
    Compute cosine distance in batch between one query vector (1434,)
    and an L2-normalized candidate_matrix (N, 1434).
    Return an (N,) array of cosine distances: 1.0 - np.dot(candidate_matrix, query_vec).
    """
    if candidate_matrix is None or len(candidate_matrix) == 0:
        return np.array([])
    # Matrix times a unit vector: (N, 1434) dot (1434,) -> (N,)
    cos_sim = np.dot(candidate_matrix, query_vec)
    # Cosine distance = 1 - cos_sim (clipped to [0, 2])
    distances = 1.0 - cos_sim
    return np.clip(distances, 0.0, 2.0)

