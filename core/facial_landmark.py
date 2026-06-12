import cv2
import mediapipe as mp
import json

input_image = "output.jpg"
output_image = "output_landmarks.jpg"
vector_file = "facial_vector.txt"

image = cv2.imread(input_image)
if image is None:
    raise FileNotFoundError(f"Không đọc được ảnh: {input_image}")

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
) as face_mesh:
    
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

 
            face_vector = []
            for landmark in face_landmarks.landmark:
                face_vector.extend([landmark.x, landmark.y, landmark.z])

            
            with open(vector_file, "w") as f:
                f.write(",".join(map(str, face_vector)))
            
            print(f"--- Đã lưu vector ({len(face_vector)} con số) vào file {vector_file} ---")
    else:
        print("Không phát hiện khuôn mặt nào.")


cv2.imwrite(output_image, image)
print(f"Đã lưu ảnh kết quả: {output_image}")
cv2.imshow("Facial Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()