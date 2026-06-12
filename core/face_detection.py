import sys
import numpy as np
import face_recognition
from PIL import Image, ImageDraw
import cv2

def detect_faces(image_path: str) -> None:
    try:
        # 1. Đọc ảnh bằng OpenCV (thư viện cực mạnh về mảng)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print("Khong tim thay anh!")
            return

        # 2. Chuyển sang RGB và ĐẢM BẢO ép kiểu uint8 chuẩn
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Tạo một bản sao vật lý mới hoàn toàn trong RAM
        img = np.array(img_rgb, dtype='uint8', copy=True)
        img = np.ascontiguousarray(img)

        print(f"--- Debug Mac M1 ---")
        print(f"Shape: {img.shape}, Dtype: {img.dtype}")
        # Kiểm tra xem mảng có bị rỗng không
        if img.size == 0:
            print("Mang anh bi rong!")
            return

        # 3. THỬ NGHIỆM: Nếu vẫn lỗi RGB, hãy bỏ comment dòng dưới để chạy bằng ảnh xám
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Gọi dlib qua face_recognition
        face_locations = face_recognition.face_locations(img, model="hog")
        
        print(f"Thanh cong! So mat: {len(face_locations)}")

        # 4. Vẽ (Dùng PIL để hiển thị cho đẹp)
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
        print(f"Co loi xay ra: {e}")
if __name__ == "__main__":
    # Đảm bảo đường dẫn này đúng với thư mục trong VS Code của bạn
    image_path = "images/train/0a0d7a87378422e3.jpg"
    detect_faces(image_path)

