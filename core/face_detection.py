import sys
import numpy as np
import face_recognition
from PIL import Image, ImageDraw
import cv2

def detect_faces(image_path: str) -> None:
    try:
        # 1. Read the image with OpenCV (a strong array library)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print("Image not found!")
            return

        # 2. Convert to RGB and force a proper uint8 type
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Make a brand-new physical copy in RAM
        img = np.array(img_rgb, dtype='uint8', copy=True)
        img = np.ascontiguousarray(img)

        print(f"--- Debug Mac M1 ---")
        print(f"Shape: {img.shape}, Dtype: {img.dtype}")
        # Check whether the array is empty
        if img.size == 0:
            print("Image array is empty!")
            return

        # 3. EXPERIMENT: if RGB still fails, uncomment the line below to run on a grayscale image
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Call dlib through face_recognition
        face_locations = face_recognition.face_locations(img, model="hog")
        
        print(f"Success! Face count: {len(face_locations)}")

        # 4. Draw (use PIL so the preview looks clean)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        for top, right, bottom, left in face_locations:
            draw.rectangle([(left, top), (right, bottom)], outline="green", width=5)
        crop_img = img[top:bottom, left:right]
        #crop_img.save("output.jpg")
        cv2.imwrite("output.jpg",crop_img)
        #cv2.imshow("cropped", crop_img)
        #cv2.waitKey(0)
     

    except Exception as e:
        print(f"An error occurred: {e}")
if __name__ == "__main__":
    # Make sure this path matches the folder open in VS Code
    image_path = "images/train/0a0d7a87378422e3.jpg"
    detect_faces(image_path)

