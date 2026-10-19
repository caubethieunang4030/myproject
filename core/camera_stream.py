import cv2
import threading
import time
import os
import config

class ThreadedCamera:
    """
    Optimized webcam reader for a Raspberry Pi 5 and a Logitech Brio 100.
    A background thread keeps grabbing the newest frame,
    which drops the buffer so OpenCV on Linux does not lag behind the webcam.
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
        # Try to open the camera with the right backend (V4L2 on Linux)
        print(f"🎥 Starting camera (index={self.src}, backend={config.CAMERA_BACKEND})...")
        self.cap = cv2.VideoCapture(self.src, config.CAMERA_BACKEND)
        
        # If that fails, scan for an available /dev/video* device
        if not self.cap.isOpened():
            print(f"⚠️ Could not open camera index {self.src}. Scanning for an available camera...")
            for idx in range(0, 4):
                if idx == self.src:
                    continue
                temp_cap = cv2.VideoCapture(idx, config.CAMERA_BACKEND)
                if temp_cap.isOpened():
                    print(f"✅ Found an available camera at index {idx}!")
                    self.src = idx
                    self.cap = temp_cap
                    break
                    
        if not self.cap.isOpened():
            raise RuntimeError("❌ NO WEBCAM FOUND! Check that the Logitech Brio 100 is connected.")

        # Set the MJPG codec that works best for a Logitech Brio 100 on a Raspberry Pi 5
        self.cap.set(cv2.CAP_PROP_FOURCC, config.CAMERA_FOURCC)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Shrink the V4L2 buffer to 1 frame so latency does not pile up
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Wait for the camera to start and try to read a frame (up to 3 seconds)
        start_t = time.time()
        while time.time() - start_t < 3.0:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.ret = ret
                    self.frame = frame.copy()
                print(f"✅ Camera Index {self.src} started successfully!")
                break
            time.sleep(0.1)
            
        if not self.ret or self.frame is None:
            print(f"⚠️ Warning: camera index {self.src} is not returning frames. Trying the other cameras...")
            for alt_idx in [0, 1, 2, 3]:
                if alt_idx == self.src:
                    continue
                temp_cap = cv2.VideoCapture(alt_idx, config.CAMERA_BACKEND)
                if temp_cap.isOpened():
                    for _ in range(10):
                        r, f = temp_cap.read()
                        if r and f is not None:
                            print(f"✅ Switched successfully to camera index {alt_idx}!")
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
        """Start the background camera-read thread."""
        self.stopped = False
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        return self

    def _update_loop(self):
        """Background loop that keeps grabbing the newest frame."""
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
        """Return the newest frame in a thread-safe way."""
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def is_running(self):
        return self.cap is not None and self.cap.isOpened() and not self.stopped

    def stop(self):
        """Stop the thread and release the camera."""
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("🎥 Camera released safely.")
