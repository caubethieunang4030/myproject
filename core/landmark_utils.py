import mediapipe as mp
import json
import numpy as np

def extract_and_save_landmarks(image_rgb, output_json_path="face_vector.json"):
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            face_vector = []
            for landmark in face_landmarks.landmark:
                face_vector.extend([landmark.x, landmark.y, landmark.z])
            
        
            with open(output_json_path, "w") as f:
                json.dump(face_vector, f)
            
            print(f"--- Đã trích xuất {len(face_vector)//3} điểm và lưu vào {output_json_path} ---")
            return face_vector
        else:
            print("Không tìm thấy khuôn mặt.")
            return None