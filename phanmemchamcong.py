import cv2
import face_recognition
import os
from datetime import datetime

def load_and_encode_known_faces(directory_path, allowed_extensions=('.jpg', '.png', '.jpeg')):
    encoding_list = []
    employee_names = []
    
    if not os.path.exists(directory_path):
        return [], []
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(allowed_extensions):
            path = os.path.join(directory_path, filename)
            image = cv2.imread(path)
            
            if image is not None:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_image)
                if encodings:
                    encoding_list.append(encodings[0])
                    employee_names.append(os.path.splitext(filename)[0])
    
    return encoding_list, employee_names

def record_attendance(name, distance, log_file='attendance.csv'):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    accuracy = round((1 - distance) * 100, 2)

    if not os.path.exists(log_file):
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('Name,Date,Time,Accuracy\n')

    with open(log_file, 'r', encoding='utf-8') as f:
        existing_data = f.readlines()
    
    already_recorded = any(name in line and date_str in line for line in existing_data)
    
    if not already_recorded:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f'{name},{date_str},{time_str},{accuracy}%\n')
        return True
    return False