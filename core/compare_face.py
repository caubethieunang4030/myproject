import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime


mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1, 
    refine_landmarks=True,
    min_detection_confidence=0.5
)


DB_PATH = "employee_db"


if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

def get_face_vector(image_path):

    img = cv2.imread(image_path)
    if img is None:
        return None

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)
    
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0]
        
        return np.array([[v.x, v.y, v.z] for v in lm.landmark])
    return None

def identify_person(target_vec, threshold=0.06):

    
    files = [f for f in os.listdir(DB_PATH) if not f.startswith('.')]
    
    if not files:
        return "Empty_DB", float('inf')
    
    best_match = "Unknown"
    min_dist = float('inf')

    for file in files:
        file_path = os.path.join(DB_PATH, file)
        try:

            saved_vec = np.loadtxt(file_path, delimiter=',')
            saved_vec = saved_vec.reshape(478, 3)
            

            dist = np.mean(np.linalg.norm(target_vec - saved_vec, axis=1))
            
            if dist < min_dist:
                min_dist = dist
                if dist < threshold:

                    best_match = os.path.splitext(file)[0]
        except Exception as e:
            print(f"Failed to read file {file}: {e}")
                
    return best_match, min_dist

if __name__ == "__main__":
  
    img_mau = "son_tung_photo_1.jpeg"
    img_test = "son_tung_photo_2.jpeg"
    
    
    if not os.listdir(DB_PATH):
        print(f"[*] Database is empty. Using {img_mau} as the reference sample...")
        v_mau = get_face_vector(img_mau)
        if v_mau is not None:
            np.savetxt(os.path.join(DB_PATH, "NV01_SonTung.txt"), v_mau.flatten(), delimiter=',')
            print("Reference file created.")
        else:
            print(f"Could not find file {img_mau} to use as the reference.")


    print(f"[*] Scanning face: {img_test}")
    current_vec = get_face_vector(img_test)

    if current_vec is not None:
        name_id, score = identify_person(current_vec)
        
        if name_id not in ["Unknown", "Empty_DB"]:
            print(f"MATCH FOUND: {name_id}")
            print(f"Distance: {score:.5f}")
        else:
            msg = "DATABASE HAS NO DATA" if name_id == "Empty_DB" else "NO EMPLOYEE MATCH"
            print(f"{msg} (Lowest distance: {score:.5f})")
    else:
        print(f"No face found in file {img_test}.")