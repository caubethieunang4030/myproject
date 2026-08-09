import os
import cv2

# Legacy Settings
IMAGE_LIB_DIR = "images"
INPUT_IMAGE_PATH = "sontungmtp.jpg"
ATTENDANCE_FILE = "attendance.csv"
MATCH_THRESHOLD = 0.5

# Hardware Camera Settings (tailored for Logitech Brio 100 & Raspberry Pi 5)
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))

# Standard V4L2 backend on Linux/Raspberry Pi OS; fallback to CAP_ANY on other platforms
IS_LINUX = hasattr(os, "uname") and os.uname().sysname == "Linux"
CAMERA_BACKEND = cv2.CAP_V4L2 if IS_LINUX else cv2.CAP_ANY

# MJPG fourcc encoding for Logitech Brio 100 over USB (prevents YUYV bandwidth bottleneck)
CAMERA_FOURCC = cv2.VideoWriter_fourcc(*'MJPG')

# Display Resolution (Stream capture resolution)
DISPLAY_WIDTH = int(os.getenv("DISPLAY_WIDTH", 1280))
DISPLAY_HEIGHT = int(os.getenv("DISPLAY_HEIGHT", 720))
DISPLAY_FPS = int(os.getenv("DISPLAY_FPS", 30))

# Downscaled resolution for MediaPipe Face Mesh processing (reduces CPU load on Pi 5)
PROCESS_WIDTH = int(os.getenv("PROCESS_WIDTH", 640))
PROCESS_HEIGHT = int(os.getenv("PROCESS_HEIGHT", 480))

# Threshold for face recognition (Cosine Distance <= 0.025 corresponds to >= 97.5% similarity)
COSINE_THRESHOLD = float(os.getenv("COSINE_THRESHOLD", 0.025))

# Attendance logging cooldown (seconds)
LOG_COOLDOWN_SECONDS = int(os.getenv("LOG_COOLDOWN_SECONDS", 900))
OFFLINE_QUEUE_FILE = "offline_queue.json"

# Web Sync Settings
DB_PATH = os.getenv("DB_PATH", "database.db")
WEB_API_URL = os.getenv("WEB_API_URL", "http://localhost:8080/api/attendance/sync-pi5")