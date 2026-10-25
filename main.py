"""Webcam check-in that writes each arrival and departure to a database.

My father and I wanted an app so older people could scan a plant and have its disease recognized, because my grandmother has a garden. I was far away, and collecting leaf and disease samples was too difficult. Before joining Affina I switched to a face check-in for students in mountain areas. There was no real student data and no real test with them, so this file is the attendance prototype built at Affina. It records check-ins. It does not predict dropouts.
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import csv
from datetime import datetime
import json
import pymssql
from dotenv import load_dotenv
import random
import threading
from cryptography.fernet import Fernet
from core.utils import normalize_vector, extract_liveness_features, check_depth_liveness, temporal_motion_tracker
load_dotenv()

GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Create a MediaPipe Face Mesh configured for a video stream (static_image_mode=False)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

db_password = os.getenv('DB_PASSWORD')
if not db_password:
    raise ValueError("ERROR: the DB_PASSWORD environment variable is not set in the .env file!")

db_config = {
    'server': os.getenv('DB_SERVER', '127.0.0.1'),
    'port': int(os.getenv('DB_PORT', 1433)),
    'user': os.getenv('DB_USER', 'sa'),
    'password': db_password,
    'database': os.getenv('DB_DATABASE', 'chamcongdatabase')
}

# Create and return the AES-256 (Fernet) cipher
def get_encryption_suite():
    key = os.getenv("ENCRYPTION_KEY")
    if not key:
        new_key = Fernet.generate_key().decode()
        env_path = '.env'
        try:
            with open(env_path, 'a') as f:
                f.write(f"\nENCRYPTION_KEY={new_key}\n")
            os.environ["ENCRYPTION_KEY"] = new_key
            key = new_key
            print(f"🔑 Generated a biometric encryption key and saved it to .env")
        except Exception as e:
            raise RuntimeError(f"Could not create ENCRYPTION_KEY or write it to .env: {e}")
    return Fernet(key.encode() if isinstance(key, str) else key)

cipher_suite = get_encryption_suite()

# Cosine-distance threshold (at most 2.5% difference, cosine similarity >= 97.5%)
THRESHOLD = 0.025  

# Offline queue configuration
OFFLINE_QUEUE_FILE = "offline_queue.json"
offline_lock = threading.Lock()

def upgrade_db_schema():
    """
    Add the loai_chamcong column to thongtinchamcong when it is missing.
    """
    conn = None
    try:
        conn = pymssql.connect(
            server=db_config['server'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        cursor = conn.cursor()
        cursor.execute("""
            IF NOT EXISTS (
                SELECT * FROM sys.columns 
                WHERE object_id = OBJECT_ID('thongtinchamcong') AND name = 'loai_chamcong'
            )
            BEGIN
                ALTER TABLE thongtinchamcong ADD loai_chamcong NVARCHAR(10) DEFAULT 'VAO';
            END
        """)
        conn.commit()
        print(f"🚀 {GREEN}[OK] Checked and updated the thongtinchamcong table (loai_chamcong column).{RESET}")
    except Exception as e:
        print(f"⚠️ {YELLOW}[WARNING] Could not upgrade the database schema automatically: {e}.{RESET}")
    finally:
        if conn:
            conn.close()

# Cooldown for attendance logs (15 minutes = 900 seconds)
LOG_COOLDOWN_SECONDS = 900
last_logged_time = {} # Stores {name: timestamp}

def load_database(w=640, h=480):
    """
    Load face templates from the vectormathocsinh table in SQL Server.
    Return shape: {mahocsinh: {"name": tenhocsinh, "vectors": [np_array, ...]}}
    """
    database = {}
    conn = None
    try:
        conn = pymssql.connect(
            server=db_config['server'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        cursor = conn.cursor()
        
        # SQL uses lowercase table and column names
        sql_query = "select mahocsinh, tenhocsinh, facevector from vectormathocsinh"
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        
        for row in rows:
            ma_hs, ten_hs, facevector_str = row
            try:
                # Decrypt the AES-256 biometric data
                try:
                    decrypted_str = cipher_suite.decrypt(facevector_str.encode()).decode()
                    data = json.loads(decrypted_str)
                except Exception:
                    # Backward compatible: if decryption fails, read the value as plain JSON
                    print(f"⚠️ {RED}[SECURITY] HS {ma_hs}: Read the vector as plaintext. It may be old unencrypted data, or it may have been modified!{RESET}")
                    data = json.loads(facevector_str)
                
                vectors = []
                if isinstance(data, dict):
                    # New format: holds several head-turn angles
                    for key in ["straight", "left", "right"]:
                        if key in data:
                            vec = np.array(data[key], dtype=np.float32)
                            if vec.size == 1434:
                                vectors.append(normalize_vector(vec.reshape(478, 3), w, h))
                else:
                    # Old format: holds a single straight-on vector
                    vec = np.array(data, dtype=np.float32)
                    if vec.size == 1434:
                        vectors.append(normalize_vector(vec.reshape(478, 3), w, h))
                
                if vectors:
                    database[ma_hs] = {
                        "name": ten_hs,
                        "vectors": vectors
                    }
            except Exception as ex:
                print(f"error while parsing the vector for student {ten_hs} ({ma_hs}): {ex}")
                
    except Exception as e:
        print(f"error while connecting to the database to load vectors: {e}")
    finally:
        if conn:
            conn.close()
    return database

def luu_vector_hoc_sinh(ma_hs, ten_hs, mang_vector, overwrite=False):
    """
    Save the student and the AES-256-encrypted template vectors to SQL Server.
    """
    conn = None
    try:
        conn = pymssql.connect(
            server=db_config['server'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        cursor = conn.cursor()
        
        # turn the float vector into clean JSON text and encrypt it
        chuoi_vector = json.dumps(mang_vector)
        chuoi_vector_encrypted = cipher_suite.encrypt(chuoi_vector.encode()).decode()
        
        # Check whether this student id already exists
        cursor.execute("select count(*) from vectormathocsinh where mahocsinh = %s", (ma_hs,))
        exists = cursor.fetchone()[0] > 0
        
        if exists:
            if not overwrite:
                print(f"error: student id '{ma_hs}' already exists in the database.")
                return False
            
            # UPDATE statement; table and column names are lowercase
            sql_query = """
                update vectormathocsinh 
                set tenhocsinh = %s, facevector = %s 
                where mahocsinh = %s
            """
            cursor.execute(sql_query, (ten_hs, chuoi_vector_encrypted, ma_hs))
            conn.commit()
            print(f"🚀 updated the vector for student: {ten_hs}")
            return True
        else:
            # INSERT statement; table and column names are lowercase
            sql_query = """
                insert into vectormathocsinh (mahocsinh, tenhocsinh, facevector) 
                values (%s, %s, %s)
            """
            cursor.execute(sql_query, (ma_hs, ten_hs, chuoi_vector_encrypted))
            conn.commit()
            print(f"🚀 saved the vector for student: {ten_hs}")
            return True
    except Exception as e:
        print(f"error while saving the vector: {e}")
        return False
    finally:
        if conn:
            conn.close()

def save_offline_log(ma_hs, loai_chamcong):
    """
    Write an offline attendance log to a local JSON file when the database is unreachable.
    """
    record = {
        "mahocsinh": ma_hs,
        "loai_chamcong": loai_chamcong,
        "thoigian": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
            print(f"⚠️ {YELLOW}[OFFLINE] Saved an offline attendance record for {ma_hs} ({loai_chamcong}) to the local file.{RESET}")
        except Exception as e:
            print(f"❌ Could not write the offline file: {e}")

def sync_offline_queue():
    """
    Background thread that syncs the offline queue to SQL Server when the network returns.
    Avoid a race by renaming the file to a temp file under the lock before syncing.
    """
    SYNC_TEMP_FILE = "offline_queue_sync.json"
    while True:
        time.sleep(10) # Check every 10 seconds
        
        if not os.path.exists(OFFLINE_QUEUE_FILE) and not os.path.exists(SYNC_TEMP_FILE):
            continue
            
        has_records = False
        with offline_lock:
            try:
                # If a temp file is left from a previous run (for example after a crash), merge it back into the main file
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
                print(f"Error while renaming the temp sync file: {e}")
                has_records = False
                
        if not has_records:
            continue
            
        # Read the temp file (no lock needed; the main thread does not write this file)
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
            
        # Try to connect to SQL Server
        conn = None
        try:
            conn = pymssql.connect(
                server=db_config['server'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database']
            )
            cursor = conn.cursor()
            print(f"📡 {GREEN}[SYNC] Database connection is back. Syncing {len(records)} offline logs...{RESET}")
            
            success_count = 0
            for record in records:
                sql_query = "insert into thongtinchamcong (mahocsinh, loai_chamcong, thoigianquet) values (%s, %s, %s)"
                cursor.execute(sql_query, (record["mahocsinh"], record["loai_chamcong"], record["thoigian"]))
                success_count += 1
                
            conn.commit()
            print(f"🚀 {GREEN}[SYNC] Synced {success_count}/{len(records)} records.{RESET}")
            
            # Delete the temp file after a successful sync
            try:
                os.remove(SYNC_TEMP_FILE)
            except Exception:
                pass
        except Exception as e:
            # On a database error, merge the temp file back into the main file under the lock so no records are lost
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
                    print(f"Error while rolling back the offline-record merge: {ex}")
        finally:
            if conn:
                conn.close()

def ghi_nhan_cham_cong(ma_hs, loai_chamcong="VAO"):
    """
    Record a check-in/out in the thongtinchamcong table in SQL Server.
    If the connection drops, save the record offline.
    """
    conn = None
    try:
        conn = pymssql.connect(
            server=db_config['server'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        cursor = conn.cursor()
        
        # SQL uses lowercase table and column names and also writes loai_chamcong
        sql_query = "insert into thongtinchamcong (mahocsinh, loai_chamcong) values (%s, %s)"
        cursor.execute(sql_query, (ma_hs, loai_chamcong))
        conn.commit()
        print(f"✅ {GREEN}Attendance ({loai_chamcong}) recorded for student: {ma_hs}{RESET}")
        return True
    except Exception as e:
        print(f"⚠️ {RED}Error while connecting to SQL Server: {e}. Saving an offline log...{RESET}")
        save_offline_log(ma_hs, loai_chamcong)
        return False
    finally:
        if conn:
            conn.close()

def main():
    print(f"\n{CYAN}=================================================={RESET}")
    print(f"{CYAN}{BOLD}    WEBCAM ATTENDANCE SYSTEM DEMO     {RESET}")
    print(f"{CYAN}=================================================={RESET}")
    
    # Upgrade the database schema if needed
    upgrade_db_schema()
    
    # Start the background thread that syncs offline attendance
    sync_thread = threading.Thread(target=sync_offline_queue, daemon=True)
    sync_thread.start()
    
    # Open the camera (index from .env, default 0, or an IP-camera URL)
    camera_index_raw = os.getenv('CAMERA_INDEX', '0')
    try:
        camera_index = int(camera_index_raw)
    except ValueError:
        camera_index = camera_index_raw
        
    print(f"\n[*] Starting camera: {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"{RED}[ERROR] Could not open the camera (index: {camera_index}). Check the connection.{RESET}")
        return
        
    print(f"{GREEN}[OK] Camera is ready!{RESET}")
    
    # Read one frame so we know the real size
    ret, frame_init = cap.read()
    if not ret:
        print(f"{RED}[ERROR] Could not read the initial frame from the camera.{RESET}")
        cap.release()
        return
    h, w, _ = frame_init.shape
    
    # Load the mobile optimization settings from .env
    mobile_opt = os.getenv('MOBILE_OPTIMIZATION', 'True').lower() in ('true', '1', 'yes')
    process_w = int(os.getenv('PROCESS_WIDTH', 640))
    process_h = int(os.getenv('PROCESS_HEIGHT', 480))
    target_fps = int(os.getenv('TARGET_FPS', 12))
    headless_mode = os.getenv('HEADLESS_MODE', 'False').lower() in ('true', '1', 'yes')
    
    # When mobile optimization is on, use the processing size so vector normalization stays consistent
    db_w = process_w if mobile_opt else w
    db_h = process_h if mobile_opt else h
    
    # Load face templates from SQL Server using the db_w and db_h ratio
    print(f"[*] Loading the face database...")
    database = load_database(db_w, db_h)
    print(f"{GREEN}[OK] Loaded {len(database)} faces.{RESET}")
    
    print(f"{YELLOW}>>> CAMERA CONTROLS:{RESET}")
    if not headless_mode:
        print(f"  - Press {BOLD}'q'{RESET} on the camera window to {BOLD}QUIT{RESET}.")
        print(f"  - Press {BOLD}'r'{RESET} on the camera window to {BOLD}ENROLL A NEW FACE{RESET} directly.")
        print(f"  - Press {BOLD}'i'{RESET} to set the attendance mode to {BOLD}CHECK-IN{RESET}.")
        print(f"  - Press {BOLD}'o'{RESET} to set the attendance mode to {BOLD}CHECK-OUT{RESET}.\n")
    else:
        print(f"  - Running with NO WINDOW (headless). Press {BOLD}Ctrl+C{RESET} in the terminal to stop.\n")
    
    # Holds the face vector currently in frame so 'r' can start enrollment
    current_target_vec = None
    
    # Counts consecutive matches so a noisy frame does not count
    last_detected_name = None
    consecutive_count = 0
    
    # Controls how long the match or enrollment result stays on screen
    liveness_status = "idle"     
    liveness_result_time = 0.0 
    liveness_result_msg = ""    
    
    # State for enrolling a new student by turning their head
    register_mode = False
    register_step = "idle"  
    register_user_id = ""
    register_user_name = ""
    register_overwrite = False
    register_vectors = {}
    register_stable_frames = 0
    register_start_time = 0.0
    register_blink_state = "waiting" # "waiting", "closed", "verified"
    
    # Current attendance mode (defaults from the system clock)
    current_hour = datetime.now().hour
    active_mode = "VAO" if current_hour < 12 else "RA"
    manual_override = False
    
    register_prompt_active = False
    
    # Temporal motion tracker used to reject spoofs
    motion_tracker = temporal_motion_tracker()
    
    # Mobile settings for how often frames are processed and how drawings are cached
    last_process_time = 0.0
    cached_face_drawings = []

    def prompt_registration_input():
        nonlocal register_mode, register_step, register_user_id, register_user_name
        nonlocal register_overwrite, register_vectors, register_stable_frames
        nonlocal register_start_time, register_blink_state, register_prompt_active
        nonlocal database
        
        try:
            print(f"\n{YELLOW}[ENROLL] Enrollment requested.{RESET}")
            ma_hs = input(f"{YELLOW} Enter the student id (example: hs001): {RESET}").strip()
            name_input = input(f"{YELLOW} Enter the student name: {RESET}").strip()
            if ma_hs and name_input:
                overwrite = False
                if ma_hs in database:
                    existing_name = database[ma_hs]["name"]
                    confirm = input(f"{YELLOW}⚠️ Student id '{ma_hs}' already exists (name: {existing_name}). Overwrite it? (y/n): {RESET}").strip().lower()
                    if confirm == 'y':
                        overwrite = True
                    else:
                        print(f"{RED}[CANCELLED] Enrollment cancelled so existing data is not overwritten.{RESET}\n")
                        register_prompt_active = False
                        return
                
                print(f"{YELLOW}[*] Starting pose capture. Look straight at the camera...{RESET}")
                register_user_id = ma_hs
                register_user_name = name_input
                register_overwrite = overwrite
                register_vectors = {}
                register_stable_frames = 0
                register_start_time = time.time()
                register_blink_state = "waiting"
                register_step = "straight"
                register_mode = True
            else:
                print(f"{RED}[ENROLLMENT CANCELLED] Student id or name is missing. Cancelled.{RESET}\n")
        except Exception as e:
            print(f"{RED}[ERROR] Enrollment error: {e}{RESET}")
        finally:
            register_prompt_active = False
            if not register_mode:
                register_step = "idle"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"{RED}[ERROR] No frame received from the camera.{RESET}")
                break
                
            current_time = time.time()
            
            # Update the attendance mode from the clock unless the user overrode it
            if not manual_override:
                active_mode = "VAO" if datetime.now().hour < 12 else "RA"
            
            # Abort enrollment if the three poses are not finished within 20 seconds
            if register_mode and (current_time - register_start_time > 20.0):
                print(f"\n{RED}[TIMED OUT] 20 seconds passed before all 3 angles were captured. Returning to the prompt.{RESET}")
                liveness_status = "rejected"
                liveness_result_time = current_time
                liveness_result_msg = "Timed out (20s)!"
                
                register_mode = False
                register_step = "idle"
                
                register_prompt_active = True
                threading.Thread(target=prompt_registration_input, daemon=True).start()
                continue
                
            # Flip the frame horizontally so it feels like a mirror
            frame = cv2.flip(frame, 1)
            
            # --- LOW-LIGHT HANDLING ---
            # Raise contrast (alpha) and brightness (beta)
            alpha = 1.3  # Contrast factor (1.0-3.0); makes edges clearer
            beta = 40    # Added brightness (0-100); lifts a dark room
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            
            h_disp, w_disp, _ = frame.shape
            
            # Limit how often AI runs when mobile optimization is on and enrollment is not active
            should_process_face = True
            elapsed = current_time - last_process_time
            if mobile_opt and not register_mode:
                min_interval = 1.0 / target_fps
                if elapsed < min_interval:
                    should_process_face = False
                    
            if should_process_face:
                # Pick the processing size from the config
                if mobile_opt:
                    h, w = process_h, process_w
                    frame_proc = cv2.resize(frame, (w, h))
                else:
                    h, w = h_disp, w_disp
                    frame_proc = frame
                    
                # Performance: convert BGR to RGB and set writable = False
                rgb_frame = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = face_mesh.process(rgb_frame)
                rgb_frame.flags.writeable = True
                
                current_target_vec = None
                cached_face_drawings = []  # Clear the previous drawing cache
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # 1. Extract a 1434-D vector from 478 landmarks
                        target_vec = np.array([[v.x, v.y, v.z] for v in face_landmarks.landmark])
                        current_target_vec = target_vec  # Keep the current vector for enrollment
                        
                        # Compute a bounding box from the landmarks so a box can be drawn around the face (in display size)
                        x_coords = [lm.x for lm in face_landmarks.landmark]
                        y_coords = [lm.y for lm in face_landmarks.landmark]
                        
                        x_min, x_max = int(min(x_coords) * w_disp), int(max(x_coords) * w_disp)
                        y_min, y_max = int(min(y_coords) * h_disp), int(max(y_coords) * h_disp)
                        
                        padding_x = int((x_max - x_min) * 0.1)
                        padding_y = int((y_max - y_min) * 0.1)
                        x1 = max(0, x_min - padding_x)
                        y1 = max(0, y_min - padding_y)
                        x2 = min(w_disp, x_max + padding_x)
                        y2 = min(h_disp, y_max + padding_y)
                        
                        # 2. Compare and find the closest face in the database (using the processing width and height)
                        best_match_id = "Unknown"
                        best_match_name = "Unknown"
                        min_dist = float('inf')
                        
                        # Normalize target_vec before comparing so camera scale does not skew the match
                        normalized_target_vec = normalize_vector(target_vec, w, h)
                        
                        # Run the liveness check (3D depth and biological motion) during normal attendance
                        if not register_mode:
                            depth_ok, z_std = check_depth_liveness(face_landmarks)
                            motion_ok, avg_motion = motion_tracker.update_and_check(face_landmarks)
                        else:
                            depth_ok, z_std = True, 0.0
                            motion_ok, avg_motion = True, 0.0
                        
                        if register_mode:
                            # --- ENROLLMENT MODE (CAPTURE SEVERAL FACE ANGLES) ---
                            features = extract_liveness_features(face_landmarks)
                            eye_l, eye_r, mouth_openness, yaw, pitch = features
                            
                            color = (255, 191, 0)  # Bright cyan
                            
                            if register_step == "straight":
                                if abs(yaw) < 0.08:
                                    # Blink state machine that rejects a still photo
                                    if register_blink_state == "waiting":
                                        if eye_l < 0.15 and eye_r < 0.15:
                                            register_blink_state = "closed"
                                            print(f"{YELLOW}[*] Eyes closed. Open your eyes to finish the check...{RESET}")
                                        label = "[1/3] ENROLL: BLINK TO VERIFY"
                                    elif register_blink_state == "closed":
                                        if eye_l > 0.22 and eye_r > 0.22:
                                            register_blink_state = "verified"
                                            register_stable_frames = 0
                                            print(f"{GREEN}[OK] Liveness check passed!{RESET}")
                                        label = "[1/3] ENROLL: OPEN EYES TO CONTINUE"
                                    elif register_blink_state == "verified":
                                        register_stable_frames += 1
                                        if register_stable_frames >= 5:
                                            register_vectors["straight"] = target_vec.flatten().tolist()
                                            register_step = "left"
                                            register_stable_frames = 0
                                            print(f"{GREEN}[OK] Captured and saved the STRAIGHT angle for student: {register_user_name}{RESET}")
                                        label = f"[1/3] ENROLL: LOOK STRAIGHT ({register_stable_frames}/5)"
                                else:
                                    if register_blink_state != "verified":
                                        register_blink_state = "waiting"
                                    register_stable_frames = 0
                                    label = "[1/3] ENROLL: LOOK STRAIGHT"
                                
                            elif register_step == "left":
                                if yaw < -0.15:
                                    register_stable_frames += 1
                                    if register_stable_frames >= 5:
                                        register_vectors["left"] = target_vec.flatten().tolist()
                                        register_step = "right"
                                        register_stable_frames = 0
                                        print(f"{GREEN}[OK] Captured and saved the LEFT angle for student: {register_user_name}{RESET}")
                                else:
                                    register_stable_frames = 0
                                
                                label = f"[2/3] ENROLL: TURN LEFT ({register_stable_frames}/5)"
                                
                            elif register_step == "right":
                                if yaw > 0.15:
                                    register_stable_frames += 1
                                    if register_stable_frames >= 5:
                                        register_vectors["right"] = target_vec.flatten().tolist()
                                        register_step = "complete"
                                        register_stable_frames = 0
                                        print(f"{GREEN}[OK] Captured and saved the RIGHT angle for student: {register_user_name}{RESET}")
                                else:
                                    register_stable_frames = 0
                                
                                label = f"[3/3] ENROLL: TURN RIGHT ({register_stable_frames}/5)"
                            
                            if register_step == "complete":
                                print(f"[*] Saving the 3-angle vector for {register_user_name} to SQL Server...")
                                if luu_vector_hoc_sinh(register_user_id, register_user_name, register_vectors, overwrite=register_overwrite):
                                    database = load_database(db_w, db_h)
                                    print(f"{GREEN}[SUCCESS] Enrolled student: {register_user_name}{RESET}\n")
                                    liveness_status = "approved"
                                    liveness_result_time = time.time()
                                    liveness_result_msg = f"Hello {register_user_name}! (enrollment succeeded)"
                                else:
                                    print(f"{RED}[ERROR] Enrollment failed because it could not be saved to the database.{RESET}\n")
                                    liveness_status = "rejected"
                                    liveness_result_time = time.time()
                                    liveness_result_msg = "Enrollment failed!"
                                    
                                register_mode = False
                                register_step = "idle"
                                color = (0, 255, 0) if liveness_status == "approved" else (0, 0, 255)
                                label = liveness_result_msg
                            
                            # Draw onto the frame immediately
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.rectangle(frame, (x1, y1 - 25), (x2, y1), color, -1)
                            cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                            
                            # Store in the cache
                            cached_face_drawings.append(("box", x1, y1, x2, y2, color, label))
                            
                            # Extra on-screen hints for enrollment
                            if register_step == "straight" and register_blink_state == "waiting":
                                cv2.putText(frame, "BLINK TO VERIFY LIVENESS", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                            elif register_step == "straight" and register_blink_state == "closed":
                                cv2.putText(frame, "OPEN YOUR EYES TO FINISH", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                            elif register_step == "straight" and register_blink_state == "verified":
                                cv2.putText(frame, "KEEP LOOKING STRAIGHT", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                            elif register_step == "straight":
                                cv2.putText(frame, "LOOK STRAIGHT AT THE CAMERA", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                            elif register_step == "left":
                                cv2.putText(frame, "TURN YOUR HEAD LEFT (<-)", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                            elif register_step == "right":
                                cv2.putText(frame, "TURN YOUR HEAD RIGHT (->)", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                                
                        else:
                            # --- NORMAL ATTENDANCE (MATCH ANY SAVED ANGLE AND RECORD IMMEDIATELY) ---
                            best_match_id = "Unknown"
                            best_match_name = "Unknown"
                            min_dist = float('inf')
                            
                            for ma_hs, info in database.items():
                                for saved_vec in info["vectors"]:
                                    dist = 1.0 - np.dot(normalized_target_vec.flatten(), saved_vec.flatten())
                                    if dist < min_dist:
                                        min_dist = dist
                                        if dist < THRESHOLD:
                                            best_match_id = ma_hs
                                            best_match_name = info["name"]
                                            
                            print(f"[DEBUG] Khop: {best_match_name} (ID: {best_match_id}) | Cos Dist: {min_dist:.4f} | Nguong THRESHOLD: {THRESHOLD}")
                            
                            if liveness_status in ["approved", "rejected"]:
                                if current_time - liveness_result_time >= 2.0:
                                    liveness_status = "idle"
                            
                            if liveness_status == "approved":
                                color = (0, 255, 0)
                                label = liveness_result_msg
                            elif liveness_status == "rejected":
                                color = (0, 0, 255)
                                label = liveness_result_msg
                            else:
                                if best_match_id != "Unknown":
                                    if best_match_id == last_detected_name:
                                        consecutive_count += 1
                                    else:
                                        last_detected_name = best_match_id
                                        consecutive_count = 1
                                        motion_tracker.buffer.clear()
                                    
                                    if consecutive_count >= 3:
                                        if not depth_ok or not motion_ok:
                                            liveness_status = "rejected"
                                            liveness_result_time = current_time
                                            liveness_result_msg = "[X] - GIA MAO"
                                            color = (0, 0, 255)
                                            label = liveness_result_msg
                                            print(f"{RED}[WARNING] Spoof detected for {best_match_name}! Depth OK: {depth_ok} (z_std: {z_std:.4f}), Motion OK: {motion_ok} (avg_motion: {avg_motion:.4f}){RESET}")
                                            consecutive_count = 0
                                            motion_tracker.buffer.clear()
                                        else:
                                            in_cooldown = (best_match_id in last_logged_time) and (current_time - last_logged_time[best_match_id] <= LOG_COOLDOWN_SECONDS)
                                            if in_cooldown:
                                                color = (0, 255, 0)
                                                label = f"Hello {best_match_name}"
                                            else:
                                                ghi_nhan_cham_cong(best_match_id, active_mode)
                                                last_logged_time[best_match_id] = current_time
                                                liveness_status = "approved"
                                                liveness_result_time = current_time
                                                liveness_result_msg = f"Hello {best_match_name}! ({active_mode} OK)"
                                                print(f"{GREEN}[OK] Attendance ({active_mode}) succeeded for {best_match_name}!{RESET}")
                                                consecutive_count = 0
                                                motion_tracker.buffer.clear()
                                                color = (0, 255, 0)
                                                label = liveness_result_msg
                                    else:
                                        color = (0, 165, 255)
                                        label = f"Recognizing... {best_match_name} ({consecutive_count}/3)"
                                else:
                                    last_detected_name = None
                                    consecutive_count = 0
                                    motion_tracker.buffer.clear()
                                    color = (0, 0, 255)
                                    label = "[X] - Unknown"
                            
                            # Draw onto the frame
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.rectangle(frame, (x1, y1 - 25), (x2, y1), color, -1)
                            cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                            
                            # Store in the cache
                            cached_face_drawings.append(("box", x1, y1, x2, y2, color, label))
                else:
                    # No face detected
                    last_detected_name = None
                    consecutive_count = 0
                    motion_tracker.buffer.clear()
                    if register_mode:
                        register_stable_frames = 0
                        cv2.putText(frame, "NO FACE FOUND", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, "Press 'c' to CANCEL ENROLLMENT", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cached_face_drawings.append(("text", "NO FACE FOUND", (30, 50), 0.6, (0, 0, 255)))
                        cached_face_drawings.append(("text", "Press 'c' to CANCEL ENROLLMENT", (30, 80), 0.5, (0, 0, 255)))
                
                last_process_time = current_time
            else:
                # Redraw cached boxes on the new frame when AI processing is skipped
                for drawing in cached_face_drawings:
                    if drawing[0] == "box":
                        _, x1, y1, x2, y2, color, label = drawing
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.rectangle(frame, (x1, y1 - 25), (x2, y1), color, -1)
                        cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    elif drawing[0] == "text":
                        _, text, pos, scale, color = drawing
                        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            
            # Draw the current attendance mode
            mode_text = f"MODE: {'VAO (Check-In)' if active_mode == 'VAO' else 'CHECK-OUT'}"
            cv2.putText(frame, mode_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Draw the offline attendance count when there is one
            if os.path.exists(OFFLINE_QUEUE_FILE):
                try:
                    with open(OFFLINE_QUEUE_FILE, 'r') as f:
                        offline_count = len(json.load(f))
                    if offline_count > 0:
                        cv2.putText(frame, f"OFFLINE QUEUE: {offline_count} recs", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                except Exception:
                    pass
                    
            # Draw an overlay while the terminal is waiting for input
            if register_prompt_active:
                overlay = frame.copy()
                cv2.rectangle(overlay, (30, h_disp // 2 - 40), (w_disp - 30, h_disp // 2 + 40), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                cv2.putText(frame, "ENTER STUDENT INFO IN THE TERMINAL...", (45, h_disp // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            
            # 4. Show the camera window (unless headless)
            key = 0xFF
            if not headless_mode:
                cv2.imshow('Face Recognition Demo', frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                # Avoid using 100% CPU when there is no window
                time.sleep(0.05)
            
            # 5. Handle key presses
            # QUIT on 'q', or when the loop ends
            if key == ord('q'):
                break
                
            # ENROLL a new face directly when 'r' is pressed
            elif key == ord('r'):
                if register_mode or register_prompt_active:
                    print(f"⚠️ {YELLOW}[WARNING] Enrollment is already in progress!{RESET}")
                elif current_target_vec is not None:
                    register_prompt_active = True
                    threading.Thread(target=prompt_registration_input, daemon=True).start()
                else:
                    print(f"⚠️ {RED}[WARNING] No face is in front of the camera to enroll!{RESET}")
                    
            # Hotkey: switch to check-in
            elif key == ord('i'):
                active_mode = "VAO"
                manual_override = True
                print(f"🔄 {CYAN}[MODE] Switched attendance to CHECK-IN [override]{RESET}")
                
            # Hotkey: switch to check-out
            elif key == ord('o'):
                active_mode = "RA"
                manual_override = True
                print(f"🔄 {CYAN}[MODE] Switched attendance to CHECK-OUT [override]{RESET}")
                
            # Press 'c' to cancel while enrollment is active
            elif key == ord('c'):
                if register_mode:
                    register_mode = False
                    register_step = "idle"
                    print(f"{RED}[CANCELLED] Student enrollment cancelled.{RESET}\n")
    except KeyboardInterrupt:
        print(f"\n{YELLOW}[SYSTEM] Shutdown requested (Ctrl+C). Closing the app...{RESET}")

            
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{CYAN}=================================================={RESET}")
    print(f"{CYAN}       CAMERA CLOSED AND DEMO FINISHED       {RESET}")
    print(f"{CYAN}=================================================={RESET}\n")

if __name__ == "__main__":
    main()

