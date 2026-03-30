import cv2
import numpy as np
import face_recognition
import time
from utils import load_and_encode_known_faces, record_attendance
from config import IMAGE_LIB_DIR, INPUT_IMAGE_PATH, MATCH_THRESHOLD
print("test")
def main():
    known_encodings, known_names = load_and_encode_known_faces(IMAGE_LIB_DIR)

    input_img = cv2.imread(INPUT_IMAGE_PATH)
    if input_img is None:
        print(f"Error: {INPUT_IMAGE_PATH} not found")
        return

    start_time = time.time()

    rgb_input = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_input)
    face_encodings = face_recognition.face_encodings(rgb_input, face_locations)

    for encoding, location in zip(face_encodings, face_locations):
        distances = face_recognition.face_distance(known_encodings, encoding)
        
        if len(distances) > 0:
            best_match_idx = np.argmin(distances)
            
            if distances[best_match_idx] < MATCH_THRESHOLD:
                name = known_names[best_match_idx]
                dist = distances[best_match_idx]
                
                y1, x2, y2, x1 = location
                cv2.rectangle(input_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(input_img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                record_attendance(name, dist)

    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Execution Time: {execution_time:.4f}s")

    cv2.imshow('Result', input_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()