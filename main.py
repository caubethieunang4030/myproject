"""Webcam check-in that writes each arrival and departure to a database.

My father and I wanted an app so older people could scan a plant and have its disease recognized, because my grandmother has a garden. I was far away, and collecting leaf and disease samples was too difficult. Before joining Affina I switched to a face check-in for students in mountain areas. There was no real student data and no real test with them, so this file is the attendance prototype built at Affina. It records check-ins. It does not predict dropouts.
"""
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

# Force OpenCV/Qt to use the XCB backend and the system font
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


# Compatibility wrapper for MediaPipe 1.0+ (Tasks API)
class FaceMeshTaskWrapper:

  def __init__(self):
    model_path = 'face_landmarker.task'
    if not os.path.exists(model_path):
      print(
          f'{YELLOW}📥 Downloading the Face Landmarker model for MediaPipe'
          f' 1.0+...{RESET}'
      )
      url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
      urllib.request.urlretrieve(url, model_path)
      print(f'{GREEN}✅ Model download finished!{RESET}')

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

    # Normalize the wrapper around the returned result
    class WrapResult:

      def __init__(self, face_landmarks_list):
        self.multi_face_landmarks = face_landmarks_list

    return WrapResult(result.face_landmarks)


# --- 1. CREATE A MEDIAPIPE FACE MESH THAT WORKS ON EVERY VERSION ---
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
        f'{GREEN}🚀 MediaPipe 1.0+ is ready (Tasks API - Face'
        f' Landmarker)!{RESET}'
    )
  except Exception as e:
    print(f'⚠️ {RED}[ERROR] Could not create the MediaPipe Face Mesh: {e}{RESET}')

# --- 2. SQLITE DATABASE SETUP ---
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
      print(f'🔑 Generated a biometric encryption key and saved it to .env')
    except Exception as e:
      raise RuntimeError(
          f'Could not create ENCRYPTION_KEY or write it to .env: {e}'
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
        f'🚀 {GREEN}[OK] Checked and updated the SQLite schema'
        f' ({DB_PATH}).{RESET}'
    )
  except Exception as e:
    print(f'⚠️ {YELLOW}[WARNING] Could not create the SQLite schema: {e}.{RESET}')
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
            f'error while parsing the vector for student {ten_hs} ({ma_hs}): {ex}'
        )

  except Exception as e:
    print(f'error while connecting to SQLite to load vectors: {e}')
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
        print(f"error: student id '{ma_hs}' already exists in the database.")
        return False

      sql_query = """
                UPDATE vectormathocsinh 
                SET tenhocsinh = ?, facevector = ? 
                WHERE mahocsinh = ?
            """
      cursor.execute(sql_query, (ten_hs, chuoi_vector_encrypted, ma_hs))
      conn.commit()
      print(f'🚀 Updated the vector for student: {ten_hs}')
      return True
    else:
      sql_query = """
                INSERT INTO vectormathocsinh (mahocsinh, tenhocsinh, facevector) 
                VALUES (?, ?, ?)
            """
      cursor.execute(sql_query, (ma_hs, ten_hs, chuoi_vector_encrypted))
      conn.commit()
      print(f'🚀 Saved the vector for student: {ten_hs}')
      return True
  except Exception as e:
    print(f'error while saving the vector: {e}')
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
          f'⚠️ {YELLOW}[OFFLINE] Saved an offline attendance record for {ma_hs}'
          f' ({loai_chamcong}) to the local file.{RESET}'
      )
    except Exception as e:
      print(f'❌ Could not write the offline file: {e}')


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
        print(f'Error while renaming the temp sync file: {e}')
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
          f'🚀 {GREEN}[SYNC] Synced {success_count}/{len(records)}'
          f' records into SQLite.{RESET}'
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
          print(f'Error while rolling back the offline-record merge: {ex}')
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
        f'✅ {GREEN}Attendance ({loai_chamcong}) recorded for student:'
        f' {ma_hs}{RESET}'
    )
    return True
  except Exception as e:
    print(
        f'⚠️ {RED}Error while writing to SQLite: {e}. Saving an offline log'
        f'...{RESET}'
    )
    save_offline_log(ma_hs, loai_chamcong)
    return False
  finally:
    if conn:
      conn.close()


# Adapter that exposes landmarks to the external liveness function
class LandmarkAdapter:

  def __init__(self, raw_landmark_list):
    self.landmark = raw_landmark_list


def main():
  print(f'\n{CYAN}=================================================={RESET}')
  print(f'{CYAN}{BOLD}    WEBCAM ATTENDANCE SYSTEM DEMO     {RESET}')
  print(f'{CYAN}=================================================={RESET}')

  if face_mesh is None:
    print(
        f'{RED}[FATAL] Could not create the MediaPipe Face Mesh.{RESET}'
    )
    return

  upgrade_db_schema()

  sync_thread = threading.Thread(target=sync_offline_queue, daemon=True)
  sync_thread.start()

  cam_index = int(os.getenv('CAMERA_INDEX', '0'))
  print(
      f'\n[*] Starting ThreadedCamera (index={cam_index} - Camera)...'
  )
  cam_stream = ThreadedCamera(
      src=cam_index,
      width=config.DISPLAY_WIDTH,
      height=config.DISPLAY_HEIGHT,
      fps=config.DISPLAY_FPS,
  ).start()

  print(f'{GREEN}[OK] ThreadedCamera is ready!{RESET}')

  print(f'[*] Waiting for frames from the camera...')
  ret, frame_init = False, None
  for _ in range(30):
    ret, frame_init = cam_stream.read()
    if ret and frame_init is not None:
      break
    time.sleep(0.1)

  if not ret or frame_init is None:
    print(f'{RED}[ERROR] Could not read a frame from the camera.{RESET}')
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

  print(f'[*] Loading the face database...')
  database = load_database(db_w, db_h)
  db_matrix, db_mapping = build_db_matrix(database)
  print(
      f'{GREEN}[OK] Loaded {len(database)} students'
      f' ({len(db_mapping)} vectors).{RESET}'
  )

  print(f'{YELLOW}>>> CAMERA CONTROLS:{RESET}')
  if not headless_mode:
    print(
        f"  - Press {BOLD}'q'{RESET} on the camera window to"
        f' {BOLD}QUIT{RESET}.'
    )
    print(
        f"  - Press {BOLD}'r'{RESET} on the camera window to {BOLD}ENROLL"
        f' A NEW FACE{RESET} directly.'
    )
    print(
        f"  - Press {BOLD}'i'{RESET} to set the attendance mode to {BOLD}CHECK-IN"
        f'{RESET}.'
    )
    print(
        f"  - Press {BOLD}'o'{RESET} to set the attendance mode to {BOLD}CHECK-OUT"
        f'{RESET}.\n'
    )
  else:
    print(
        '  - Running with NO WINDOW (headless). Press'
        f' {BOLD}Ctrl+C{RESET} in the terminal to stop.\n'
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
      print(f'\n{YELLOW}[ENROLL] Enrollment requested.{RESET}')
      ma_hs = (
          input(f'{YELLOW} Enter the student id (example: hs001): {RESET}').strip()
      )
      name_input = input(f'{YELLOW} Enter the student name: {RESET}').strip()
      if ma_hs and name_input:
        overwrite = False
        if ma_hs in database:
          existing_name = database[ma_hs]['name']
          confirm = (
              input(
                  f"⚠️ Student id '{ma_hs}' already exists (name:"
                  f' {existing_name}). Overwrite it? (y/n):'
                  f' {RESET}'
              )
              .strip()
              .lower()
          )
          if confirm == 'y':
            overwrite = True
          else:
            print(
                f'{RED}[CANCELLED] Enrollment cancelled so existing data is not overwritten.{RESET}\n'
            )
            register_prompt_active = False
            return

        print(
            f'{YELLOW}[*] Starting pose capture. Look straight at the'
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
            f'{RED}[ENROLLMENT CANCELLED] Student id or name is missing.'
            f' Cancelled.{RESET}\n'
        )
    except Exception as e:
      print(f'{RED}[ERROR] Enrollment error: {e}{RESET}')
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
            f'\n{RED}[TIMED OUT] 20 seconds passed before all 3 angles were captured.'
            f' Returning to the prompt.{RESET}'
        )
        liveness_status = 'rejected'
        liveness_result_time = current_time
        liveness_result_msg = 'Timed out (20s)!'

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
            # Extract the landmark list regardless of the result type
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
                          f'{YELLOW}[*] Eyes closed. Open your eyes to finish'
                          f' the check...{RESET}'
                      )
                    label = '[1/3] ENROLL: BLINK TO VERIFY'
                  elif register_blink_state == 'closed':
                    if eye_l > 0.22 and eye_r > 0.22:
                      register_blink_state = 'verified'
                      register_stable_frames = 0
                      print(f'{GREEN}[OK] Liveness check passed!{RESET}')
                    label = '[1/3] ENROLL: OPEN EYES TO CONTINUE'
                  elif register_blink_state == 'verified':
                    register_stable_frames += 1
                    if register_stable_frames >= 5:
                      register_vectors['straight'] = (
                          target_vec.flatten().tolist()
                      )
                      register_step = 'left'
                      register_stable_frames = 0
                      print(
                          f'{GREEN}[OK] Captured and saved the STRAIGHT angle for student:'
                          f' {register_user_name}{RESET}'
                      )
                    label = (
                        '[1/3] ENROLL: LOOK STRAIGHT'
                        f' ({register_stable_frames}/5)'
                    )
                else:
                  if register_blink_state != 'verified':
                    register_blink_state = 'waiting'
                  register_stable_frames = 0
                  label = '[1/3] ENROLL: LOOK STRAIGHT'

              elif register_step == 'left':
                if yaw < -0.15:
                  register_stable_frames += 1
                  if register_stable_frames >= 5:
                    register_vectors['left'] = target_vec.flatten().tolist()
                    register_step = 'right'
                    register_stable_frames = 0
                    print(
                        f'{GREEN}[OK] Captured and saved the LEFT angle for student:'
                        f' {register_user_name}{RESET}'
                    )
                else:
                  register_stable_frames = 0

                label = f'[2/3] ENROLL: TURN LEFT ({register_stable_frames}/5)'

              elif register_step == 'right':
                if yaw > 0.15:
                  register_stable_frames += 1
                  if register_stable_frames >= 5:
                    register_vectors['right'] = target_vec.flatten().tolist()
                    register_step = 'complete'
                    register_stable_frames = 0
                    print(
                        f'{GREEN}[OK] Captured and saved the RIGHT angle for student:'
                        f' {register_user_name}{RESET}'
                    )
                else:
                  register_stable_frames = 0

                label = f'[3/3] ENROLL: TURN RIGHT ({register_stable_frames}/5)'

              if register_step == 'complete':
                print(
                    f'[*] Saving the 3-angle vector for {register_user_name} vao'
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
                      f'{GREEN}[SUCCESS] Enrolled student:'
                      f' {register_user_name}{RESET}\n'
                  )
                  liveness_status = 'approved'
                  liveness_result_time = time.time()
                  liveness_result_msg = (
                      f'Hello {register_user_name}! (enrollment succeeded)'
                  )
                else:
                  print(
                      f'{RED}[ERROR] Enrollment failed because it could not be saved to the'
                      f' database.{RESET}\n'
                  )
                  liveness_status = 'rejected'
                  liveness_result_time = time.time()
                  liveness_result_msg = 'Enrollment failed!'

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
                        label = f'Hello {best_match_name}'
                      else:
                        ghi_nhan_cham_cong(best_match_id, active_mode)
                        last_logged_time[best_match_id] = current_time
                        liveness_status = 'approved'
                        liveness_result_time = current_time
                        liveness_result_msg = (
                            f'Hello {best_match_name}! ({active_mode} OK)'
                        )
                        consecutive_count = 0
                        motion_tracker.buffer.clear()
                        color = (0, 255, 0)
                        label = liveness_result_msg
                  else:
                    color = (0, 165, 255)
                    label = (
                        f'Recognizing... {best_match_name}'
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
          f"MODE: {'VAO (Check-In)' if active_mode == 'VAO' else 'RA'}"
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
              f'⚠️ {YELLOW}[WARNING] Enrollment is already in'
              f' progress!{RESET}'
          )
        elif current_target_vec is not None:
          register_prompt_active = True
          threading.Thread(
              target=prompt_registration_input, daemon=True
          ).start()
        else:
          print(
              f'⚠️ {RED}[WARNING] No face is in front of the'
              f' camera to enroll!{RESET}'
          )
      elif key == ord('i'):
        active_mode = 'VAO'
        manual_override = True
        print(
            f'🔄 {CYAN}[MODE] Switched attendance to CHECK-IN'
            f' [Override]{RESET}'
        )
      elif key == ord('o'):
        active_mode = 'RA'
        manual_override = True
        print(
            f'🔄 {CYAN}[MODE] Switched attendance to CHECK-OUT'
            f' [Override]{RESET}'
        )
      elif key == ord('c'):
        if register_mode:
          register_mode = False
          register_step = 'idle'
          print(f'{RED}[CANCELLED] Student enrollment cancelled.{RESET}\n')
  except KeyboardInterrupt:
    print(
        f'\n{YELLOW}[SYSTEM] Shutdown requested (Ctrl+C). Closing'
        f' the app...{RESET}'
    )

  cam_stream.stop()
  cv2.destroyAllWindows()
  print(f'\n{CYAN}=================================================={RESET}')
  print(f'{CYAN}       CAMERA CLOSED AND DEMO FINISHED       {RESET}')
  print(f'{CYAN}=================================================={RESET}\n')


if __name__ == '__main__':
  main()
