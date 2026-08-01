import cv2
import threading
import time
import os
import config

class ThreadedCamera:
    """
    Tối ưu hóa luồng đọc webcam cho Raspberry Pi 5 & Logitech Brio 100.
    Chạy luồng nền (background thread) liên tục lấy khung hình mới nhất,
    loại bỏ hoàn toàn bộ đệm (buffer latency) gây trễ webcam trên OpenCV Linux.
    """
    def __init__(self, src=config.CAMERA_INDEX, width=config.DISPLAY_WIDTH, height=config.DISPLAY_HEIGHT, fps=config.DISPLAY_FPS):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap = None
        self.frame = None
        self.ret = False
        self.stopped = False
        self.lock = threading.Lock()
        
        self._init_camera()
        
    def _init_camera(self):
        # Thử khởi tạo camera với backend thích hợp (V4L2 trên Linux)
        print(f"🎥 Đang khởi tạo Camera (index={self.src}, backend={config.CAMERA_BACKEND})...")
        self.cap = cv2.VideoCapture(self.src, config.CAMERA_BACKEND)
        
        # Nếu mở thất bại, tự động tìm kiếm các cổng /dev/video* khả dụng
        if not self.cap.isOpened():
            print(f"⚠️ Không thể mở camera index {self.src}. Đang quét thiết bị camera khả dụng...")
            for idx in range(0, 4):
                if idx == self.src:
                    continue
                temp_cap = cv2.VideoCapture(idx, config.CAMERA_BACKEND)
                if temp_cap.isOpened():
                    print(f"✅ Đã tìm thấy camera khả dụng tại index {idx}!")
                    self.src = idx
                    self.cap = temp_cap
                    break
                    
        if not self.cap.isOpened():
            raise RuntimeError("❌ KHÔNG TÌM THẤY WEBCAM! Vui lòng kiểm tra kết nối Logitech Brio 100.")

        # Cấu hình codec MJPG tối ưu cho Logitech Brio 100 trên Raspberry Pi 5
        self.cap.set(cv2.CAP_PROP_FOURCC, config.CAMERA_FOURCC)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Giảm thiểu size bộ đệm V4L2 xuống 1 khung hình để chống trễ tích tụ
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Chờ camera khởi động và đọc thử khung hình (thử lại trong tối đa 3 giây)
        start_t = time.time()
        while time.time() - start_t < 3.0:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.ret = ret
                    self.frame = frame.copy()
                print(f"✅ Camera Index {self.src} khởi động thành công!")
                break
            time.sleep(0.1)
            
        if not self.ret or self.frame is None:
            print(f"⚠️ Cảnh báo: Camera index {self.src} không trả về khung hình. Đang thử quét các camera khác...")
            for alt_idx in [0, 1, 2, 3]:
                if alt_idx == self.src:
                    continue
                temp_cap = cv2.VideoCapture(alt_idx, config.CAMERA_BACKEND)
                if temp_cap.isOpened():
                    for _ in range(10):
                        r, f = temp_cap.read()
                        if r and f is not None:
                            print(f"✅ Đã kết nối thành công sang Camera Index {alt_idx}!")
                            self.src = alt_idx
                            self.cap.release()
                            self.cap = temp_cap
                            with self.lock:
                                self.ret = r
                                self.frame = f.copy()
                            break
                        time.sleep(0.1)
                    if self.ret:
                        break

    def start(self):
        """Khởi chạy luồng đọc camera ngầm."""
        self.stopped = False
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        return self

    def _update_loop(self):
        """Vòng lặp ngầm liên tục cào khung hình mới nhất."""
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(0.01)
                continue
                
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self):
        """Lấy khung hình mới nhất theo cách an toàn đa luồng."""
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def is_running(self):
        return self.cap is not None and self.cap.isOpened() and not self.stopped

    def stop(self):
        """Dừng luồng và giải phóng tài nguyên camera."""
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("🎥 Đã giải phóng camera an toàn.")
