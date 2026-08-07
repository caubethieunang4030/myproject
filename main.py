import csv
from datetime import datetime
import json
import os
import random
import sqlite3
import threading
import time
import urllib.request
from cryptography.fernet import Fernet
import cv2
from dotenv import load_dotenv
import numpy as np

# Ép OpenCV/Qt sử dụng backend XCB và font hệ thống
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import mediapipe as mp

import config
from core.camera_stream import ThreadedCamera
from core.utils import (
    batch_cosine_distance,
    check_depth_liveness,
    extract_liveness_features,
    normalize_vector,
    temporal_motion_tracker,
)

load_dotenv()

GREEN = '\033[92m'
RED = '\033[91m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'


# Wrapper tương thích MediaPipe 1.0+ (Tasks API)
class FaceMeshTaskWrapper:

  def __init__(self):
    model_path = 'face_landmarker.task'
    if not os.path.exists(model_path):
      print(
          f'{YELLOW}📥 Đang tải file model Face Landmarker cho MediaPipe'
          f' 1.0+...{RESET}'
      )
      url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
      urllib.request.urlretrieve(url, model_path)
      print(f'{GREEN}✅ Đã tải xong model!{RESET}')

    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
    )
    self.landmarker = FaceLandmarker.create_from_options(options)

  def process(self, rgb_image):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = self.landmarker.detect(mp_image)

    # Chuẩn hóa wrapper cho kết quả trả về
    class WrapResult:

      def __init__(self, face_landmarks_list):
        self.multi_face_landmarks = face_landmarks_list

    return WrapResult(result.face_landmarks)


# --- 1. KHỞI TẠO MEDIAPIPE FACE MESH TƯƠNG THÍCH MỌI PHIÊN BẢN ---
face_mesh = None

try:
  import mediapipe.python.solutions.face_mesh as mp_fm

  face_mesh = mp_fm.FaceMesh(
      static_image_mode=False,
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5,
  )
except Exception:
  pass

if face_mesh is None:
  try:
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
      face_mesh = mp.solutions.face_mesh.FaceMesh(
          static_image_mode=False,
          max_num_faces=1,
          refine_landmarks=True,
          min_detection_confidence=0.5,
          min_tracking_confidence=0.5,
      )
  except Exception:
    pass

if face_mesh is None:
  try:
    face_mesh = FaceMeshTaskWrapper()
    print(
        f'{GREEN}🚀 Khởi tạo thành công MediaPipe 1.0+ (Tasks API - Face'
        f' Landmarker)!{RESET}'
    )
  except Exception as e:
    print(f'⚠️ {RED}[LỖI] Không thể khởi tạo MediaPipe Face Mesh: {e}{RESET}')

# --- 2. CẤU HÌNH CƠ SỞ DỮ LIỆU SQLITE ---
DB_PATH = os.getenv('DB_PATH', 'database.db')


def get_db_connection():
  conn = sqlite3.connect(DB_PATH)
  conn.execute('PRAGMA journal_mode=WAL;')
  return conn


def get_encryption_suite():
  key = os.getenv('ENCRYPTION_KEY')
  if not key:
    new_key = Fernet.generate_key().decode()
    env_path = '.env'
    try:
      with open(env_path, 'a') as f:
        f.write(f'\nENCRYPTION_KEY={new_key}\n')
      os.environ['ENCRYPTION_KEY'] = new_key
      key = new_key
      print(f'🔑 Đã tự động tạo khóa mã hóa sinh trắc học và lưu vào .env')
    except Exception as e:
      raise RuntimeError(
          f'Không thể khởi tạo và ghi ENCRYPTION_KEY vào .env: {e}'
      )
  return Fernet(key.encode() if isinstance(key, str) else key)


cipher_suite = get_encryption_suite()
THRESHOLD = 0.025
OFFLINE_QUEUE_FILE = 'offline_queue.json'
offline_lock = threading.Lock()


def upgrade_db_schema():
  conn = None
  try:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS vectormathocsinh (
                mahocsinh TEXT PRIMARY KEY,
                tenhocsinh TEXT NOT NULL,
                facevector TEXT NOT NULL
            );
        """)
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS thongtinchamcong (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mahocsinh TEXT NOT NULL,
                loai_chamcong TEXT DEFAULT 'VAO',
                thoigianquet DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
    conn.commit()
    print(
        f'🚀 {GREEN}[OK] Đã kiểm tra và đồng bộ cấu trúc cơ sở dữ liệu SQLite'
        f' ({DB_PATH}).{RESET}'
    )
  except Exception as e:
    print(f'⚠️ {YELLOW}[CẢNH BÁO] Không thể khởi tạo schema SQLite: {e}.{RESET}')
  finally:
    if conn:
      conn.close()


LOG_COOLDOWN_SECONDS = 900
last_logged_time = {}


def load_database(w=640, h=480):
  database = {}
  conn = None
  try:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT mahocsinh, tenhocsinh, facevector FROM vectormathocsinh'
    )
    rows = cursor.fetchall()

    for row in rows:
      ma_hs, ten_hs, facevector_str = row
      try:
        try:
          decrypted_str = cipher_suite.decrypt(
              facevector_str.encode()
          ).decode()
          data = json.loads(decrypted_str)
        except Exception:
          data = json.loads(facevector_str)

        vectors = []
        if isinstance(data, dict):
          for key in ['straight', 'left', 'right']:
            if key in data:
              vec = np.array(data[key], dtype=np.float32)
              if vec.size == 1434:
                vectors.append(normalize_vector(vec.reshape(478, 3), w, h))
        else:
          vec = np.array(data, dtype=np.float32)
          if vec.size == 1434:
            vectors.append(normalize_vector(vec.reshape(478, 3), w, h))

        if vectors:
          database[ma_hs] = {'name': ten_hs, 'vectors': vectors}
      except Exception as ex:
        print(
            f'lỗi khi phân giải vector cho học sinh {ten_hs} ({ma_hs}): {ex}'
        )

  except Exception as e:
    print(f'lỗi khi kết nối SQLite để tải vector: {e}')
  finally:
    if conn:
      conn.close()
  return database


def build_db_matrix(database):
  matrix_rows = []
  mapping = []
  for ma_hs, info in database.items():
    name = info['name']
    for vec in info['vectors']:
      matrix_rows.append(vec.flatten())
      mapping.append((ma_hs, name))
  if matrix_rows:
    return np.array(matrix_rows, dtype=np.float32), mapping
  return None, []


def luu_vector_hoc_sinh(ma_hs, ten_hs, mang_vector, overwrite=False):
  conn = None
  try:
    conn = get_db_connection()
    cursor = conn.cursor()
    chuoi_vector = json.dumps(mang_vector)
    chuoi_vector_encrypted = cipher_suite.encrypt(
        chuoi_vector.encode()
    ).decode()

    cursor.execute(
        'SELECT COUNT(*) FROM vectormathocsinh WHERE mahocsinh = ?', (ma_hs,)
    )
    exists = cursor.fetchone()[0] > 0

    if exists:
      if not overwrite:
        print(f"lỗi: Mã học sinh '{ma_hs}' đã tồn tại trong database.")
        return False

      sql_query = """
                UPDATE vectormathocsinh 
                SET tenhocsinh = ?, facevector = ? 
                WHERE mahocsinh = ?
            """
      cursor.execute(sql_query, (ten_hs, chuoi_vector_encrypted, ma_hs))
      conn.commit()
      print(f'🚀 Đã cập nhật thành công vector cho học sinh: {ten_hs}')
      return True
    else:
      sql_query = """
                INSERT INTO vectormathocsinh (mahocsinh, tenhocsinh, facevector) 
                VALUES (?, ?, ?)
            """
      cursor.execute(sql_query, (ma_hs, ten_hs, chuoi_vector_encrypted))
      conn.commit()
      print(f'🚀 Đã lưu thành công vector cho học sinh: {ten_hs}')
      return True
  except Exception as e:
    print(f'lỗi khi lưu vector: {e}')
    return False
  finally:
    if conn:
      conn.close()


def save_offline_log(ma_hs, loai_chamcong):
  record = {
      'mahocsinh': ma_hs,
      'loai_chamcong': loai_chamcong,
      'thoigian': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
  }
  with offline_lock:
    data = []
    if os.path.exists(OFFLINE_QUEUE_FILE):
      try:
        with open(OFFLINE_QUEUE_FILE, 'r', encoding='utf-8') as f:
          data = json.load(f)
      except Exception:
        data = []
    data.append(record)
    try:
      with open(OFFLINE_QUEUE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
      print(
          f'⚠️ {YELLOW}[OFFLINE] Đã lưu điểm danh ngoại tuyến cho {ma_hs}'
          f' ({loai_chamcong}) vào file local.{RESET}'
      )
    except Exception as e:
      print(f'❌ Không thể lưu file offline: {e}')


def sync_offline_queue():
  SYNC_TEMP_FILE = 'offline_queue_sync.json'
  while True:
    time.sleep(10)

    if not os.path.exists(OFFLINE_QUEUE_FILE) and not os.path.exists(
        SYNC_TEMP_FILE
    ):
      continue

    has_records = False
    with offline_lock:
      try:
        if os.path.exists(SYNC_TEMP_FILE):
          s_rec, o_rec = [], []
          try:
            with open(SYNC_TEMP_FILE, 'r', encoding='utf-8') as sf:
              s_rec = json.load(sf)
          except Exception:
            s_rec = []
          try:
            with open(OFFLINE_QUEUE_FILE, 'r', encoding='utf-8') as of:
              o_rec = json.load(of)
          except Exception:
            o_rec = []

          merged = o_rec + s_rec
          if merged:
            with open(OFFLINE_QUEUE_FILE, 'w', encoding='utf-8') as of:
              json.dump(merged, of, ensure_ascii=False, indent=4)

          try:
            os.remove(SYNC_TEMP_FILE)
          except Exception:
            pass

        if os.path.exists(OFFLINE_QUEUE_FILE):
          os.rename(OFFLINE_QUEUE_FILE, SYNC_TEMP_FILE)
          has_records = True
      except Exception as e:
        print(f'Lỗi khi đổi tên file tạm đồng bộ: {e}')
        has_records = False

    if not has_records:
      continue

    records = []
    try:
      with open(SYNC_TEMP_FILE, 'r', encoding='utf-8') as f:
        records = json.load(f)
    except Exception:
      try:
        os.remove(SYNC_TEMP_FILE)
      except Exception:
        pass
      continue

    if not records:
      try:
        os.remove(SYNC_TEMP_FILE)
      except Exception:
        pass
      continue

    conn = None
    try:
      conn = get_db_connection()
      cursor = conn.cursor()

      success_count = 0
      for record in records:
        sql_query = """
                    INSERT INTO thongtinchamcong (mahocsinh, loai_chamcong, thoigianquet) 
                    VALUES (?, ?, ?)
                """
        cursor.execute(
            sql_query,
            (record['mahocsinh'], record['loai_chamcong'], record['thoigian']),
        )
        success_count += 1

      conn.commit()
      print(
          f'🚀 {GREEN}[ĐỒNG BỘ] Đồng bộ thành công {success_count}/{len(records)}'
          f' bản ghi vào SQLite.{RESET}'
      )

      try:
        os.remove(SYNC_TEMP_FILE)
      except Exception:
        pass
    except Exception as e:
      with offline_lock:
        try:
          main_records = []
          if os.path.exists(OFFLINE_QUEUE_FILE):
            try:
              with open(OFFLINE_QUEUE_FILE, 'r', encoding='utf-8') as f:
                main_records = json.load(f)
            except Exception:
              main_records = []

          merged_records = records + main_records
          with open(OFFLINE_QUEUE_FILE, 'w', encoding='utf-8') as f:
            json.dump(merged_records, f, ensure_ascii=False, indent=4)

          os.remove(SYNC_TEMP_FILE)
        except Exception as ex:
          print(f'Lỗi khi hoàn tác merge records ngoại tuyến: {ex}')
    finally:
      if conn:
        conn.close()


def ghi_nhan_cham_cong(ma_hs, loai_chamcong='VAO'):
  conn = None
  try:
    conn = get_db_connection()
    cursor = conn.cursor()
    sql_query = """
            INSERT INTO thongtinchamcong (mahocsinh, loai_chamcong, thoigianquet) 
            VALUES (?, ?, ?)
        """
    thoigian_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute(sql_query, (ma_hs, loai_chamcong, thoigian_now))
    conn.commit()
    print(
        f'✅ {GREEN}Điểm danh ({loai_chamcong}) thành công cho học sinh:'
        f' {ma_hs}{RESET}'
    )
    return True
  except Exception as e:
    print(
        f'⚠️ {RED}Lỗi khi ghi nhận SQLite: {e}. Tiến hành lưu log'
        f' ngoại tuyến...{RESET}'
    )
    save_offline_log(ma_hs, loai_chamcong)
    return False
  finally:
    if conn:
      conn.close()


# Adapter hỗ trợ đối tượng landmark cho hàm liveness bên ngoài
class LandmarkAdapter:

  def __init__(self, raw_landmark_list):
    self.landmark = raw_landmark_list


def main():
  print(f'\n{CYAN}=================================================={RESET}')
  print(f'{CYAN}{BOLD}    HỆ THỐNG CHẤM CÔNG WEBCAM DEMO     {RESET}')
  print(f'{CYAN}=================================================={RESET}')

  if face_mesh is None:
    print(
        f'{RED}[LỖI NGHIÊM TRỌNG] Không thể khởi tạo MediaPipe FaceMesh.{RESET}'
    )
    return

  upgrade_db_schema()

  sync_thread = threading.Thread(target=sync_offline_queue, daemon=True)
  sync_thread.start()

  cam_index = int(os.getenv('CAMERA_INDEX', '0'))
  print(
      f'\n[*] Đang khởi động ThreadedCamera (Index={cam_index} - Camera)...'
  )
  cam_stream = ThreadedCamera(
      src=cam_index,
      width=config.DISPLAY_WIDTH,
      height=config.DISPLAY_HEIGHT,
      fps=config.DISPLAY_FPS,
  ).start()

  print(f'{GREEN}[OK] ThreadedCamera đã sẵn sàng!{RESET}')

  print(f'[*] Đang chờ luồng hình ảnh từ camera...')
  ret, frame_init = False, None
  for _ in range(30):
    ret, frame_init = cam_stream.read()
    if ret and frame_init is not None:
      break
    time.sleep(0.1)

  if not ret or frame_init is None:
    print(f'{RED}[LỖI] Không thể đọc khung hình từ camera.{RESET}')
    cam_stream.stop()
    return
  h_disp, w_disp, _ = frame_init.shape

  mobile_opt = os.getenv('MOBILE_OPTIMIZATION', 'True').lower() in (
      'true',
      '1',
      'yes',
  )
  process_w = config.PROCESS_WIDTH if mobile_opt else w_disp
  process_h = config.PROCESS_HEIGHT if mobile_opt else h_disp
  target_fps = int(os.getenv('TARGET_FPS', '30'))
  headless_mode = os.getenv('HEADLESS_MODE', 'False').lower() in (
      'true',
      '1',
      'yes',
  )

  db_w = process_w if mobile_opt else w_disp
  db_h = process_h if mobile_opt else h_disp

  print(f'[*] Đang tải cơ sở dữ liệu khuôn mặt...')
  database = load_database(db_w, db_h)
  db_matrix, db_mapping = build_db_matrix(database)
  print(
      f'{GREEN}[OK] Đã tải thành công {len(database)} học sinh'
      f' ({len(db_mapping)} vectors).{RESET}'
  )

  print(f'{YELLOW}>>> HƯỚNG DẪN ĐIỀU KHIỂN CAM:{RESET}')
  if not headless_mode:
    print(
        f"  - Nhấn phím {BOLD}'q'{RESET} trên cửa sổ camera để"
        f' {BOLD}THOÁT{RESET}.'
    )
    print(
        f"  - Nhấn phím {BOLD}'r'{RESET} trên cửa sổ camera để {BOLD}ĐĂNG KÝ"
        f' KHUÔN MẶT MỚI{RESET} trực tiếp.'
    )
    print(
        f"  - Nhấn phím {BOLD}'i'{RESET} để chọn chế độ chấm công {BOLD}VÀO"
        f' (Check-In){RESET}.'
    )
    print(
        f"  - Nhấn phím {BOLD}'o'{RESET} để chọn chế độ chấm công {BOLD}RA"
        f' (Check-Out){RESET}.\n'
    )
  else:
    print(
        '  - Đang chạy ở chế độ KHÔNG MÀN HÌNH (Headless). Nhấn'
        f' {BOLD}Ctrl+C{RESET} trong terminal để dừng.\n'
    )

  current_target_vec = None
  last_detected_name = None
  consecutive_count = 0

  liveness_status = 'idle'
  liveness_result_time = 0.0
  liveness_result_msg = ''

  register_mode = False
  register_step = 'idle'
  register_user_id = ''
  register_user_name = ''
  register_overwrite = False
  register_vectors = {}
  register_stable_frames = 0
  register_start_time = 0.0
  register_blink_state = 'waiting'

  current_hour = datetime.now().hour
  active_mode = 'VAO' if current_hour < 12 else 'RA'
  manual_override = False

  register_prompt_active = False

  motion_tracker = temporal_motion_tracker()

  last_process_time = 0.0
  cached_face_drawings = []

  def prompt_registration_input():
    nonlocal register_mode, register_step, register_user_id, register_user_name
    nonlocal register_overwrite, register_vectors, register_stable_frames
    nonlocal register_start_time, register_blink_state, register_prompt_active
    nonlocal database

    try:
      print(f'\n{YELLOW}[ĐĂNG KÝ] Phát hiện lệnh đăng ký khuôn mặt.{RESET}')
      ma_hs = (
          input(f'{YELLOW} Nhập mã học sinh (ví dụ: hs001): {RESET}').strip()
      )
      name_input = input(f'{YELLOW} Nhập tên học sinh: {RESET}').strip()
      if ma_hs and name_input:
        overwrite = False
        if ma_hs in database:
          existing_name = database[ma_hs]['name']
          confirm = (
              input(
                  f"⚠️ Mã học sinh '{ma_hs}' đã tồn tại (Tên:"
                  f' {existing_name}). Bạn có muốn cập nhật/ghi đè? (y/n):'
                  f' {RESET}'
              )
              .strip()
              .lower()
          )
          if confirm == 'y':
            overwrite = True
          else:
            print(
                f'{RED}[HỦY BỎ] Hủy đăng ký để tránh ghi đè dữ liệu.{RESET}\n'
            )
            register_prompt_active = False
            return

        print(
            f'{YELLOW}[*] Bắt đầu quét góc mặt. Vui lòng nhìn thẳng vào'
            f' camera...{RESET}'
        )
        register_user_id = ma_hs
        register_user_name = name_input
        register_overwrite = overwrite
        register_vectors = {}
        register_stable_frames = 0
        register_start_time = time.time()
        register_blink_state = 'waiting'
        register_step = 'straight'
        register_mode = True
      else:
        print(
            f'{RED}[HUỶ ĐĂNG KÝ] Thiếu thông tin mã học sinh hoặc tên học'
            f' sinh. Huỷ bỏ.{RESET}\n'
        )
    except Exception as e:
      print(f'{RED}[LỖI] Đăng ký lỗi: {e}{RESET}')
    finally:
      register_prompt_active = False
      if not register_mode:
        register_step = 'idle'

  try:
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0

    while True:
      ret, frame = cam_stream.read()
      if not ret or frame is None:
        time.sleep(0.005)
        continue

      current_time = time.time()

      if not manual_override:
        active_mode = 'VAO' if datetime.now().hour < 12 else 'RA'

      if register_mode and (current_time - register_start_time > 20.0):
        print(
            f'\n{RED}[HẾT GIỜ] Đã quá 20 giây mà chưa hoàn tất quét 3 góc mặt.'
            f' Quay lại luồng nhập thông tin.{RESET}'
        )
        liveness_status = 'rejected'
        liveness_result_time = current_time
        liveness_result_msg = 'Qua thoi gian (20s)!'

        register_mode = False
        register_step = 'idle'

        register_prompt_active = True
        threading.Thread(target=prompt_registration_input, daemon=True).start()
        continue

      frame = cv2.flip(frame, 1)

      alpha = 1.3
      beta = 40
      frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

      h_disp, w_disp, _ = frame.shape

      should_process_face = True
      elapsed = current_time - last_process_time
      if mobile_opt and not register_mode:
        min_interval = 1.0 / target_fps
        if elapsed < min_interval:
          should_process_face = False

      if should_process_face:
        if mobile_opt:
          h, w = process_h, process_w
          frame_proc = cv2.resize(frame, (w, h))
        else:
          h, w = h_disp, w_disp
          frame_proc = frame

        rgb_frame = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        current_target_vec = None
        cached_face_drawings = []

        if results.multi_face_landmarks:
          for face_landmarks in results.multi_face_landmarks:
            # Tự động trích xuất danh sách landmark bất kể kiểu dữ liệu
            landmarks_list = (
                face_landmarks.landmark
                if hasattr(face_landmarks, 'landmark')
                else face_landmarks
            )
            adapted_landmarks = (
                face_landmarks
                if hasattr(face_landmarks, 'landmark')
                else LandmarkAdapter(landmarks_list)
            )

            target_vec = np.array([[v.x, v.y, v.z] for v in landmarks_list])
            current_target_vec = target_vec

            x_coords = [lm.x for lm in landmarks_list]
            y_coords = [lm.y for lm in landmarks_list]

            x_min, x_max = int(min(x_coords) * w_disp), int(
                max(x_coords) * w_disp
            )
            y_min, y_max = int(min(y_coords) * h_disp), int(
                max(y_coords) * h_disp
            )

            padding_x = int((x_max - x_min) * 0.1)
            padding_y = int((y_max - y_min) * 0.1)
            x1 = max(0, x_min - padding_x)
            y1 = max(0, y_min - padding_y)
            x2 = min(w_disp, x_max + padding_x)
            y2 = min(h_disp, y_max + padding_y)

            best_match_id = 'Unknown'
            best_match_name = 'Unknown'
            min_dist = float('inf')

            normalized_target_vec = normalize_vector(target_vec, w, h)

            if not register_mode:
              depth_ok, z_std = check_depth_liveness(adapted_landmarks)
              motion_ok, avg_motion = motion_tracker.update_and_check(
                  adapted_landmarks
              )
            else:
              depth_ok, z_std = True, 0.0
              motion_ok, avg_motion = True, 0.0

            if register_mode:
              features = extract_liveness_features(adapted_landmarks)
              eye_l, eye_r, mouth_openness, yaw, pitch = features

              color = (255, 191, 0)

              if register_step == 'straight':
                if abs(yaw) < 0.08:
                  if register_blink_state == 'waiting':
                    if eye_l < 0.15 and eye_r < 0.15:
                      register_blink_state = 'closed'
                      print(
                          f'{YELLOW}[*] Đã nhắm mắt. Vui lòng mở mắt để hoàn'
                          f' tất xác thực...{RESET}'
                      )
                    label = '[1/3] DANG KY: NHAY MAT DE XAC MINH'
                  elif register_blink_state == 'closed':
                    if eye_l > 0.22 and eye_r > 0.22:
                      register_blink_state = 'verified'
                      register_stable_frames = 0
                      print(f'{GREEN}[OK] Xác thực liveness thành công!{RESET}')
                    label = '[1/3] DANG KY: MO MAT DE TIEP TUC'
                  elif register_blink_state == 'verified':
                    register_stable_frames += 1
                    if register_stable_frames >= 5:
                      register_vectors['straight'] = (
                          target_vec.flatten().tolist()
                      )
                      register_step = 'left'
                      register_stable_frames = 0
                      print(
                          f'{GREEN}[OK] Đã chụp và lưu góc THẲNG cho học sinh:'
                          f' {register_user_name}{RESET}'
                      )
                    label = (
                        '[1/3] DANG KY: NHIN THANG'
                        f' ({register_stable_frames}/5)'
                    )
                else:
                  if register_blink_state != 'verified':
                    register_blink_state = 'waiting'
                  register_stable_frames = 0
                  label = '[1/3] DANG KY: NHIN THANG'

              elif register_step == 'left':
                if yaw < -0.15:
                  register_stable_frames += 1
                  if register_stable_frames >= 5:
                    register_vectors['left'] = target_vec.flatten().tolist()
                    register_step = 'right'
                    register_stable_frames = 0
                    print(
                        f'{GREEN}[OK] Da chup va luu goc TRAI cho hoc sinh:'
                        f' {register_user_name}{RESET}'
                    )
                else:
                  register_stable_frames = 0

                label = f'[2/3] DANG KY: QUAY TRAI ({register_stable_frames}/5)'

              elif register_step == 'right':
                if yaw > 0.15:
                  register_stable_frames += 1
                  if register_stable_frames >= 5:
                    register_vectors['right'] = target_vec.flatten().tolist()
                    register_step = 'complete'
                    register_stable_frames = 0
                    print(
                        f'{GREEN}[OK] Da chup va luu goc PHAI cho hoc sinh:'
                        f' {register_user_name}{RESET}'
                    )
                else:
                  register_stable_frames = 0

                label = f'[3/3] DANG KY: QUAY PHAI ({register_stable_frames}/5)'

              if register_step == 'complete':
                print(
                    f'[*] Dang luu vector 3 goc cua {register_user_name} vao'
                    ' SQLite database...'
                )
                if luu_vector_hoc_sinh(
                    register_user_id,
                    register_user_name,
                    register_vectors,
                    overwrite=register_overwrite,
                ):
                  database = load_database(db_w, db_h)
                  db_matrix, db_mapping = build_db_matrix(database)
                  print(
                      f'{GREEN}[THÀNH CÔNG] Đăng ký thành công học sinh:'
                      f' {register_user_name}{RESET}\n'
                  )
                  liveness_status = 'approved'
                  liveness_result_time = time.time()
                  liveness_result_msg = (
                      f'Chao {register_user_name}! (Dang ky thanh cong)'
                  )
                else:
                  print(
                      f'{RED}[LỖI] Đăng ký thất bại do không luu duoc vao'
                      f' database.{RESET}\n'
                  )
                  liveness_status = 'rejected'
                  liveness_result_time = time.time()
                  liveness_result_msg = 'Dang ky that bai!'

                register_mode = False
                register_step = 'idle'
                color = (
                    (0, 255, 0)
                    if liveness_status == 'approved'
                    else (0, 0, 255)
                )
                label = liveness_result_msg

              cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
              cv2.rectangle(frame, (x1, y1 - 25), (x2, y1), color, -1)
              cv2.putText(
                  frame,
                  label,
                  (x1 + 5, y1 - 7),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.5,
                  (255, 255, 255),
                  1,
                  cv2.LINE_AA,
              )

              cached_face_drawings.append(
                  ('box', x1, y1, x2, y2, color, label)
              )

            else:
              best_match_id = 'Unknown'
              best_match_name = 'Unknown'
              min_dist = float('inf')

              if db_matrix is not None and len(db_matrix) > 0:
                distances = batch_cosine_distance(
                    normalized_target_vec.flatten(), db_matrix
                )
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                if min_dist < THRESHOLD:
                  best_match_id, best_match_name = db_mapping[min_idx]

              if liveness_status in ['approved', 'rejected']:
                if current_time - liveness_result_time >= 2.0:
                  liveness_status = 'idle'

              if liveness_status == 'approved':
                color = (0, 255, 0)
                label = liveness_result_msg
              elif liveness_status == 'rejected':
                color = (0, 0, 255)
                label = liveness_result_msg
              else:
                if best_match_id != 'Unknown':
                  if best_match_id == last_detected_name:
                    consecutive_count += 1
                  else:
                    last_detected_name = best_match_id
                    consecutive_count = 1
                    motion_tracker.buffer.clear()

                  if consecutive_count >= 3:
                    if not depth_ok or not motion_ok:
                      liveness_status = 'rejected'
                      liveness_result_time = current_time
                      liveness_result_msg = '[X] - GIA MAO'
                      color = (0, 0, 255)
                      label = liveness_result_msg
                      consecutive_count = 0
                      motion_tracker.buffer.clear()
                    else:
                      in_cooldown = (best_match_id in last_logged_time) and (
                          current_time - last_logged_time[best_match_id]
                          <= LOG_COOLDOWN_SECONDS
                      )
                      if in_cooldown:
                        color = (0, 255, 0)
                        label = f'Chao {best_match_name}'
                      else:
                        ghi_nhan_cham_cong(best_match_id, active_mode)
                        last_logged_time[best_match_id] = current_time
                        liveness_status = 'approved'
                        liveness_result_time = current_time
                        liveness_result_msg = (
                            f'Chao {best_match_name}! ({active_mode} OK)'
                        )
                        consecutive_count = 0
                        motion_tracker.buffer.clear()
                        color = (0, 255, 0)
                        label = liveness_result_msg
                  else:
                    color = (0, 165, 255)
                    label = (
                        f'Nhan dang... {best_match_name}'
                        f' ({consecutive_count}/3)'
                    )
                else:
                  last_detected_name = None
                  consecutive_count = 0
                  motion_tracker.buffer.clear()
                  color = (0, 0, 255)
                  label = '[X] - Unknown'

              cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
              cv2.rectangle(frame, (x1, y1 - 25), (x2, y1), color, -1)
              cv2.putText(
                  frame,
                  label,
                  (x1 + 5, y1 - 7),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.5,
                  (255, 255, 255),
                  1,
                  cv2.LINE_AA,
              )

              cached_face_drawings.append(
                  ('box', x1, y1, x2, y2, color, label)
              )
        else:
          last_detected_name = None
          consecutive_count = 0
          motion_tracker.buffer.clear()

        last_process_time = current_time
      else:
        for drawing in cached_face_drawings:
          if drawing[0] == 'box':
            _, x1, y1, x2, y2, color, label = drawing
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x2, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

      mode_text = (
          f"CHE DO: {'VAO (Check-In)' if active_mode == 'VAO' else 'RA'}"
          ' (Check-Out)'
      )
      cv2.putText(
          frame,
          mode_text,
          (20, 30),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (0, 255, 255),
          2,
          cv2.LINE_AA,
      )

      fps_counter += 1
      if current_time - fps_start_time >= 1.0:
        current_fps = fps_counter / (current_time - fps_start_time)
        fps_counter = 0
        fps_start_time = current_time
      cv2.putText(
          frame,
          f'FPS: {current_fps:.1f}',
          (w_disp - 130, 30),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 255, 0),
          2,
          cv2.LINE_AA,
      )

      if os.path.exists(OFFLINE_QUEUE_FILE):
        try:
          with open(OFFLINE_QUEUE_FILE, 'r') as f:
            offline_count = len(json.load(f))
          if offline_count > 0:
            cv2.putText(
                frame,
                f'OFFLINE QUEUE: {offline_count} recs',
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        except Exception:
          pass

      key = 0xFF
      if not headless_mode:
        cv2.imshow('Face Recognition Demo', frame)
        key = cv2.waitKey(1) & 0xFF
      else:
        time.sleep(0.05)

      if key == ord('q'):
        break
      elif key == ord('r'):
        if register_mode or register_prompt_active:
          print(
              f'⚠️ {YELLOW}[CẢNH BÁO] Hệ thống đang trong tiến trình'
              f' đăng ký!{RESET}'
          )
        elif current_target_vec is not None:
          register_prompt_active = True
          threading.Thread(
              target=prompt_registration_input, daemon=True
          ).start()
        else:
          print(
              f'⚠️ {RED}[CẢNH BÁO] Không phát hiện khuôn mặt nào trước'
              f' camera để đăng ký!{RESET}'
          )
      elif key == ord('i'):
        active_mode = 'VAO'
        manual_override = True
        print(
            f'🔄 {CYAN}[CHẾ ĐỘ] Đã chuyển sang chấm công: VÀO (Check-In)'
            f' [Override]{RESET}'
        )
      elif key == ord('o'):
        active_mode = 'RA'
        manual_override = True
        print(
            f'🔄 {CYAN}[CHẾ ĐỘ] Đã chuyển sang chấm công: RA (Check-Out)'
            f' [Override]{RESET}'
        )
      elif key == ord('c'):
        if register_mode:
          register_mode = False
          register_step = 'idle'
          print(f'{RED}[HỦY BỎ] Đã hủy đăng ký học sinh.{RESET}\n')
  except KeyboardInterrupt:
    print(
        f'\n{YELLOW}[HỆ THỐNG] Nhận lệnh tắt từ người dùng (Ctrl+C). Đang đóng'
        f' ứng dụng...{RESET}'
    )

  cam_stream.stop()
  cv2.destroyAllWindows()
  print(f'\n{CYAN}=================================================={RESET}')
  print(f'{CYAN}       ĐÃ ĐÓNG CAMERA & KẾT THÚC DỰ ÁN DEMO       {RESET}')
  print(f'{CYAN}=================================================={RESET}\n')


if __name__ == '__main__':
  main()
