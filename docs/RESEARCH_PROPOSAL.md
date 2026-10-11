# RESEARCH PROPOSAL

**Autonomous Biometric Edge AI Appliance**  
**Empirical Optimization of Computer Vision, Liveness Verification, and Resilient Storage on ARM64 Embedded Hardware**

| Field | Detail |
|---|---|
| **Student Researcher** | Duy Anh Huynh (Vincent) |
| **Affiliation** | Grade 12 \| Rabun Gap-Nacoochee School |
| **Research Area** | Computer Science — Embedded Systems, Edge AI & Computer Vision |
| **Technical Advisor** | Huong Nguyen \| Head of AI & Automation, Affina (InsurTech) \| Master in Information Technology \| PhD in Data Analytics (In Progress) \| huong.nguyen3@affina.com.vn |
| **Project Origin** | Independent Exploration & Edge AI Internship \| Affina (InsurTech) \| Ho Chi Minh City, Vietnam |
| **Project Development** | Summer 2026 \| 12-Week Research & Exploration at Affina, including a 4-Week Solution Implementation Phase |

**Online Profile & Code**

- GitHub Repository: https://github.com/caubethieunang4030/myproject.git
- LinkedIn: https://linkedin.com/in/duy-anh-huynh-34b3ab380

**Media Coverage**

- VnExpress International — Education: https://e.vnexpress.net/news/news/education/grade-11-student-develops-edge-ai-attendance-system-from-real-world-problem-5108390.html — “Grade 11 student develops Edge AI attendance system from a real-world problem” \| August 13, 2026

> **Revision marking legend:** Text marked with <span style="background-color:#FFFF00;color:#FF0000;">yellow highlight and red text</span> must be reviewed or corrected with final measured numbers / official links before submission. Original intent is preserved.

---

## PROJECT OVERVIEW

### PROJECT GOAL
Develop and evaluate an autonomous, on-premise Edge AI biometric attendance appliance running on Raspberry Pi 5 — so schools and SMEs can check people in without sending face photos to the cloud, and without stopping when the internet drops.

### CORE SOLUTION (Four Techniques)
1. **Threaded V4L2/MJPG Camera Ingestion** — keeps the live video smooth on a small computer.
2. **1434-D MediaPipe Facial Mesh Matching** — turns a face into a compact math “fingerprint,” not a photo.
3. **Heuristic Liveness Verification** — checks that a real 3D face (and blink/motion) is present, not a paper photo or phone screen.
4. **SQLite WAL Offline-First Storage** — saves attendance locally even during outages, then syncs later.

### RESEARCH FOCUS
- Recognition Accuracy
- Throughput & Latency
- 2D Presentation-Attack Resistance
- Storage Reliability

### KEY TECHNOLOGIES
- Raspberry Pi 5
- Python
- OpenCV
- MediaPipe
- V4L2/MJPG
- SQLite WAL
- Fernet Encryption (AES-128-CBC + HMAC)

### EVALUATION TARGETS
<span style="background-color:#FFFF00;color:#FF0000;">• TAR ≥98.5%</span>  
<span style="background-color:#FFFF00;color:#FF0000;">• 30 FPS</span>  
<span style="background-color:#FFFF00;color:#FF0000;">• &lt;150 ms Latency</span>  
<span style="background-color:#FFFF00;color:#FF0000;">• APCER ≤3% Paper / ≤5% Screen</span>  
<span style="background-color:#FFFF00;color:#FF0000;">• Zero Record Loss during Simulated Outages of up to 48 Hours</span>

\*Proposed evaluation targets; final performance will be determined through empirical testing (see Section 5).

---

## 1. PROJECT SUMMARY & ABSTRACT

Biometric attendance systems in educational institutions and small-to-medium enterprises (SMEs) face major operational challenges: cloud-based solutions expose student biometric data to external networks and fail during broadband outages, while naive desktop vision code deployed to single-board computers (SBCs) suffers from camera driver latency, high CPU load, and vulnerability to 2D photo presentation attacks.

This research proposes the design, empirical optimization, and validation of an autonomous, on-premise **Edge AI Biometric Appliance** running on the ARM64-based Raspberry Pi 5. Based on the actual codebase implementation, the project addresses hardware and software constraints through a four-part architecture (explained for a non-specialist reader in Section 1.1 below):

1. **Decoupled Threaded V4L2 MJPG Frame Ingestion** — A background `ThreadedCamera` worker uses native Linux V4L2 MJPG hardware encoding with `CAP_PROP_BUFFERSIZE = 1` at 1280×720@30 FPS (downscaled to 640×480 for inference), eliminating image-buffer lag.
2. **Normalized 1434-D Facial Mesh Vector Matching** — MediaPipe Face Mesh extracts 478 3D landmarks (478×3 = 1434 values). Vectors are centered to the origin and L2-normalized. Identity matching uses batch Cosine Distance with `COSINE_THRESHOLD = 0.025` (≥97.5% similarity).
3. **Two-Tier Heuristic Liveness Engine & Enrollment Verification** — Real-time checks combine 3D cranial Z-depth dispersion and ICD-normalized temporal motion; enrollment also requires blink transitions measured by Eye Aspect Ratio (EAR).
4. **SQLite WAL Persistence & Resilient JSON Sync Daemon** — Check-ins and encrypted vectors persist in SQLite WAL mode; if the database is locked or offline, records fall back to `offline_queue.json` and a background daemon syncs every 10 seconds.

The appliance will be benchmarked on public academic datasets (LFW) and deployed in an institutional pilot under formal consent.

### 1.1 The Four Techniques — What They Are, How They Work, and Why They Matter

Think of the appliance as a small “attendance guard” that must (a) see clearly, (b) recognize the right person, (c) refuse fake photos, and (d) never lose a check-in. Each technique owns one of those jobs.

#### Technique A — Threaded V4L2/MJPG Camera Ingestion
| Question | Answer in plain language |
|---|---|
| **What is it?** | A dedicated background thread that continuously reads the newest camera frame using Linux’s V4L2 driver and the webcam’s MJPG hardware compression. |
| **How does it work?** | Instead of letting OpenCV pile up old frames in a queue (which caused ~2 seconds of lag on Pi 5), the code keeps only one frame (`BUFFERSIZE = 1`) and always hands the *latest* image to the AI. Capture stays at 1280×720 for a clear display; inference uses 640×480 to save CPU. |
| **Contribution to the whole project** | Without this layer, recognition and liveness would run on *stale* video — students would already have walked past before the system “saw” them. Smooth, low-latency capture is the foundation that makes the other three techniques usable in a real hallway. |

#### Technique B — 1434-D MediaPipe Facial Mesh Matching
| Question | Answer in plain language |
|---|---|
| **What is it?** | A privacy-preserving way to describe a face as a list of 1434 numbers (a vector), not as a stored photograph. |
| **How does it work?** | MediaPipe draws a 3D “mesh” of 478 points on the face. Those points are moved so the face center is at (0,0,0), stretched into one long vector, and scaled to unit length (L2-normalized). Two faces are compared with Cosine Distance: if distance ≤ 0.025, they are treated as the same person. |
| **Contribution to the whole project** | This is the *identity* engine. It answers “who is this?” quickly on a Pi 5 without a GPU, and it supports the privacy claim that **no raw face images are retained** — only encrypted math vectors. |

#### Technique C — Heuristic Liveness Verification
| Question | Answer in plain language |
|---|---|
| **What is it?** | A two-layer “is this a live person?” filter plus a blink check during enrollment. |
| **How does it work?** | Layer 1 measures 3D depth variation across five rigid skull landmarks (forehead, chin, nose, two inner eye corners). A flat paper/screen has almost no depth variation. Layer 2 watches tiny natural motion across three frames, scaled by the distance between the eyes so standing closer/farther does not break the test. During registration, the student must blink (EAR closed → open). |
| **Contribution to the whole project** | Recognition alone can be fooled by holding up a photo. Liveness is the *security* layer that protects attendance integrity and prevents fake enrollments — critical for any school deployment. |

#### Technique D — SQLite WAL Offline-First Storage (+ Fernet encryption)
| Question | Answer in plain language |
|---|---|
| **What is it?** | A local file database on the Pi that can accept writes even when the network or a remote server is down, with encrypted biometric vectors at rest. |
| **How does it work?** | SQLite runs in Write-Ahead Logging (WAL) mode so one thread can write attendance while another reads vectors. If a write fails, the check-in is appended to `offline_queue.json`; a daemon renames/merges that file every 10 seconds and batch-inserts safely. Vectors are encrypted with Fernet (AES-128-CBC + HMAC) before storage. |
| **Contribution to the whole project** | This is the *reliability & privacy* layer. Morning attendance must not stop because Wi-Fi fails, and stolen storage media should not expose readable face templates. Together with Techniques A–C, it completes an on-premise appliance rather than a fragile demo script. |

---

## 2. PROBLEM STATEMENT & RESEARCH OBJECTIVES

### 2.1 Background & Context
Centralized cloud attendance systems continuously transmit facial telemetry over wide-area networks (WAN). This creates three primary issues:

1. **Privacy Risks:** Transmitting unencrypted facial data conflicts with FERPA (34 CFR Part 99) and BIPA (740 ILCS 14/) expectations for biometric handling.
2. **Network Dependence:** Network dropouts halt check-in operations during peak morning arrival hours.
3. **Continuous Cost:** SaaS subscriptions generate ongoing operational expense for schools.

Desktop computer-vision scripts using `dlib` or uncompressed camera feeds fail on embedded Linux: YUYV video streams saturate USB bandwidth (≈2-second visual delay), and simple image blur filters (Laplacian variance) are easily bypassed by crisp smartphone photographs.

### 2.2 Research Questions
1. How can an ARM64 single-board computer achieve sub-150 ms biometric recognition at 30 FPS using normalized 3D facial landmark vectors without a discrete GPU?
2. How effectively can a two-tier heuristic combining Z-depth dispersion (σ<sub>z</sub>) and ICD-normalized temporal motion detect 2D paper-print and mobile-screen presentation attacks without active depth hardware?
3. How can an offline-first hybrid storage architecture combining SQLite WAL mode with a thread-locked atomic JSON queue ensure 100% record recovery during network disruptions exceeding 48 hours?

### 2.3 Specific Measurable Objectives
1. **Recognition Accuracy:** True Accept Rate (TAR) ≥98.5% at Cosine Distance ≤0.025 (≥97.5% similarity).
2. **Throughput & Latency:** Sustained frame capture at 30 FPS with total single-subject identification latency under 150 ms.
3. **Spoofing Resistance:** Attack Presentation Classification Error Rate (APCER) ≤3.0% for paper prints and ≤5.0% for mobile screen replays.
4. **Storage Reliability:** 100% log recovery with zero dropped check-in records across simulated 48-hour network outages.

---

## 3. PRELIMINARY WORK & LESSONS LEARNED (CODEBASE EVOLUTION)

Deployment testing on a Raspberry Pi 5 running Debian Bookworm exposed concrete technical challenges that shaped the codebase. Each item below is tied to real iteration in the Git history (see also Section 5.4).

| Observed Roadblock | Implemented Code Solution | Why it mattered |
|---|---|---|
| **Dependency & ABI Failures** — `dlib` failed (missing AVX); NumPy 2.x crashed MediaPipe C-extensions | Pinned deps in `requirements-rpi5.txt` (`numpy>=1.24,<2.0`, `mediapipe>=0.10.9,<0.10.20`); automated via `setup_rpi5.sh`; MediaPipe Tasks API fallback (`face_landmarker.task`) in `main.py` | Made the appliance *installable* on ARM64 instead of only on a laptop |
| **Camera Driver Latency (~2 s lag)** — Default V4L2 queue accumulated uncompressed YUYV frames | `ThreadedCamera` (`core/camera_stream.py`) with `cv2.CAP_V4L2`, MJPG FourCC, `CAP_PROP_BUFFERSIZE = 1` | Restored real-time usability in a hallway setting |
| **High CPU Consumption** — Full 1280×720 through MediaPipe overloaded Pi 5 | Separated display vs process resolution in `config.py` (`PROCESS_WIDTH/HEIGHT = 640×480`) | Freed CPU budget for liveness + matching in the same loop |
| **Passive Blur Filter Limitations** — Phone screens passed Laplacian sharpness checks | Two-tier biological liveness in `core/utils.py`: Z-depth σ<sub>z</sub> > 0.015 + ICD temporal motion | Moved security from “image looks sharp” to “geometry behaves like a real head” |
| **SQLite Concurrency & Lockouts** — Concurrent threads locked the DB at peak check-in | `PRAGMA journal_mode=WAL;` + `save_offline_log()` → `offline_queue.json` + sync daemon | Preserved the “never lose a check-in” requirement |

---

## 4. SYSTEM / SOFTWARE ARCHITECTURE AND METHODOLOGY

### 4.0 End-to-End Software Pipeline (How the Pieces Fit)

```text
USB Webcam (Brio 100)
        │  MJPG @ 1280×720 / 30 FPS
        ▼
 ThreadedCamera (V4L2, buffer=1)     ← Technique A
        │
        ▼
 Resize to 640×480 → MediaPipe Face Mesh (478 landmarks)
        │
        ├─→ Liveness (depth σz + temporal motion + EAR)  ← Technique C
        │         │ fail → reject frame (possible spoof)
        │         ▼ pass
        └─→ normalize_vector (1434-D) → Cosine match     ← Technique B
                  │
                  ▼ match + stable N frames
         Encrypt vector / Write attendance
                  │
         SQLite WAL  ←or→  offline_queue.json daemon     ← Technique D
```

### 4.0.1 Source-Code & Repository Architecture (Why a Monorepo)

This project uses a **monorepo**: one Git repository contains the appliance runtime, helper scripts, docs, images, and Pi setup tooling together.

```text
myproject/                          # single repository (monorepo)
├── main.py                         # runtime entry: camera → mesh → liveness → match → storage
├── delete_student.py               # privacy / opt-out tooling (purge student rows)
├── sync_to_web.py                  # optional SQLite → web API sync (Pi deployment branch)
├── config.py                       # hardware & threshold knobs (resolution, COSINE_THRESHOLD, …)
├── setup_rpi5.sh                   # one-shot Pi environment bootstrap
├── requirements.txt                # laptop / general dependencies
├── requirements-rpi5.txt           # ARM64-pinned dependencies
├── .env.example                    # safe template (no secrets committed)
├── core/                           # reusable vision & liveness library
│   ├── camera_stream.py            # ThreadedCamera (V4L2/MJPG)
│   ├── utils.py                    # normalize, depth liveness, temporal motion, EAR helpers
│   ├── face_detection.py           # early experiments / helpers
│   ├── facial_landmark.py
│   ├── landmark_utils.py
│   └── compare_face.py             # static image comparison utility
├── docs/                           # research proposal & figures
│   ├── RESEARCH_PROPOSAL.md
│   └── images/
├── images/                         # sample stills used during early matching experiments
├── database.db                     # local SQLite file (Pi / offline-first path; not for secrets)
└── README.md                       # operator-facing English documentation
```

**Why monorepo (and not polyrepo / separate packages)?**

| Option | Why it was *not* chosen for this project | Why monorepo *was* chosen |
|---|---|---|
| **Polyrepo** (e.g., `vision-core` repo + `pi-firmware` repo + `docs` repo) | Extra coordination cost for a solo student researcher; version drift between camera code and docs; harder for advisors to review one PR | One clone = whole appliance. Capstone reviewers and Affina mentors see code + methodology + figures together |
| **Publishable multi-package layout** (`pip install my-face-lib`) | Premature abstraction; the value is the *integrated appliance*, not a reusable SDK yet | Keeps `core/` as an internal library without packaging overhead on ARM64 |
| **Monorepo (chosen)** | — | Matches Edge AI reality: camera thread, thresholds, DB schema, and setup scripts must evolve **in lockstep**. A single commit can fix lag *and* retune liveness *and* update the proposal evidence trail |

**How monorepo contributes to project management**

- **Atomic changes:** threshold tuning (e.g., Cosine 0.25 → 0.18 → 0.13 → 0.15 → 0.025) stays next to the code that uses it.
- **Reproducible Pi bring-up:** `setup_rpi5.sh` + `requirements-rpi5.txt` live beside `main.py`.
- **Auditability for ethics/privacy:** `delete_student.py` and encryption path are in the same tree reviewers inspect.
- **Simpler internship handoff:** Affina advisors review one repository URL rather than stitching multiple services.

### 4.1 Hardware Architecture

**Figure 4.1 — Physical appliance and labeled parts**

![Hardware Architecture — Edge AI Biometric Appliance](images/hardware_architecture_rpi5_appliance.jpg)

| Part | What it is | Role in the appliance |
|---|---|---|
| **Raspberry Pi 5 (8GB LPDDR4X, BCM2712 Cortex-A76 @ 2.4 GHz)** | The small computer (green board) | Runs Linux + Python vision pipeline entirely on-device |
| **Active PWM cooler fan** | Fan on the CPU | Keeps thermal headroom so MediaPipe does not throttle mid-check-in |
| **Official 27W USB-C power supply** | Stable power brick | Prevents brownouts under camera + CPU load |
| **Logitech Brio 100 USB webcam** | Optical sensor @ 1280×720 / 30 FPS MJPG | Provides the video stream for recognition & liveness |
| **64GB microSD (A2/U3)** | Boot + data storage | Holds OS, code, SQLite DB, offline queue |
| **CP2102 USB-to-TTL (GPIO 14 TXD / 15 RXD @ 115200 baud)** | Serial debug adapter | Headless recovery console when HDMI/GUI is unavailable |
| **RP1 I/O controller (on Pi 5)** | USB packet routing silicon | Sustains USB3/2 webcam traffic without starving the CPU path |

### 4.2 Vision Processing & Facial Vector Normalization

**Threaded Capture (`core/camera_stream.py`):** `ThreadedCamera` runs a background thread polling `cv2.VideoCapture`. `CAP_PROP_BUFFERSIZE = 1` retains only the latest frame, eliminating ring-buffer delay.

**Dual-Resolution Pipeline:** Capture uses `DISPLAY_WIDTH/HEIGHT = 1280×720` for operator display; MediaPipe inference uses `PROCESS_WIDTH/HEIGHT = 640×480` to lower CPU load.

**Facial Landmark Extraction:** MediaPipe Face Mesh detects 478 3D facial landmarks (478×3 = 1434 values).

**Vector Normalization (`normalize_vector` in `core/utils.py`):**

1. Landmark coordinates are scaled to pixel space \((x\cdot w,\ y\cdot h,\ z\cdot w)\).
2. The centroid is shifted to the origin: \(v_{\text{centered}} = v - \bar{v}\).
3. The vector is flattened to 1434 dimensions and L2-normalized: \(v_{\text{norm}} = v_{\text{flat}} / \|v_{\text{flat}}\|_2\).

**Batch Cosine Distance Matching:** Candidate matrix \(M\) (\(N\times1434\)) is compared to \(v_{\text{norm}}\) by \( \text{Distance} = 1.0 - (M \cdot v_{\text{norm}}) \). A match is declared when Distance ≤ `COSINE_THRESHOLD = 0.025` (≥97.5% similarity).

### 4.3 Two-Tier Liveness Verification & Enrollment Check

**Goal of this subsection (for a non-engineer reader):**  
The system never stores a photograph. Instead it tracks a **digital skeleton** — numbered points on a face mesh. Some points act like **rigid bones** (skull structure that should show real 3D depth). Other points track **eye motion** (blinks). Together they answer: *“Is a living person in front of the camera, or is someone holding a flat picture / phone?”*

**Figure 4.3 — Facial landmark skeleton used for liveness**

![Facial Landmark Skeleton Used for Liveness](images/facial_landmark_skeleton_liveness.jpg)

#### Why these “bones” (landmarks) exist
MediaPipe returns hundreds of points. Using all of them for anti-spoofing would be noisy and slow. The project therefore selects a **small rigid set** that should *not* move relative to each other like soft tissue, but *should* show depth differences on a real head:

| Landmark ID | Everyday meaning | Why it was chosen |
|---|---|---|
| **10** | Forehead | Top of the cranial structure |
| **152** | Chin | Bottom of the cranial structure |
| **4** | Nose tip | Forward-most point — strong depth signal on a real face |
| **133 / 362** | Left / right inner eye corners (inner canthi) | Stable eye-socket anchors; also define ICD (scale reference) |

On a **live face**, nose tip Z differs from forehead/chin Z → σ<sub>z</sub> is larger.  
On a **paper print or phone screen**, almost everything sits on one flat plane → σ<sub>z</sub> collapses → reject.

#### Tier 1 — Z-Depth Dispersion (`check_depth_liveness`)
- Computes \(\sigma_z = \mathrm{std}(z_{\text{rigid}})\) over the five landmarks above.
- Frames with \(\sigma_z \le 0.015\) are treated as flat 2D attacks and rejected.
- **Project-level contribution:** Blocks the most common hallway spoof (holding a classmate’s photo) *before* identity matching runs — saving CPU and protecting attendance integrity.

#### Tier 2 — ICD-Normalized Temporal Motion (`temporal_motion_tracker`)
- Rolling buffer of **3 consecutive frames** on Nose (4), Eyes (133/362), Lips (13/14).
- Motion is divided by Inner Canthus Distance \(\mathrm{ICD}=\|p_{133}-p_{362}\|_2\) so standing nearer/farther does not fake the score.
- Ignores sensor noise (Motion &lt; 0.002); requires average Motion &gt; 0.015 for biological liveness.
- **Project-level contribution:** Catches *static* spoofs that might still show tiny depth noise, and confirms micro-motion consistent with a living person — without buying a depth camera.

#### Interactive Enrollment Verification (`extract_liveness_features` + EAR blink)
Enrollment captures **3 poses** (straight, left, right) so later check-ins tolerate natural head angle.

During the straight pose, blink is required via Eye Aspect Ratio:

\[
\mathrm{EAR} = \frac{\|p_{159}-p_{145}\|_2}{\mathrm{ICD}}
\]

- Closed eyes: EAR &lt; 0.15 → Open eyes: EAR &gt; 0.22 must occur as a **state transition**.
- **Why eye motion matters:** A printed photo cannot execute a closed→open blink on command. This stops *fake student registration*, which is as dangerous as fake check-in.
- **Commit-learned fix:** Glasses sometimes broke blink detection (`00b46f6`); thresholds and feature extraction were adjusted so real students with glasses could still enroll without weakening photo rejection.

### 4.4 SQLite WAL Persistence & Atomic Offline Queue Daemon

#### Why this storage design (design rationale)
The appliance must survive three hard constraints at once:

1. **No guaranteed always-on network** at school doors.
2. **Concurrent threads** (camera loop + sync daemon + UI) touching the same records.
3. **Biometric privacy** if the microSD card is removed or copied.

That combination ruled out “only talk to a remote SQL Server” as the *sole* store on the Pi path, and favored an **embedded offline-first database** with an encrypted vector column and a crash-safe JSON fallback.

#### Why SQLite (and why WAL mode)
| Design choice | Why it fits *this* project |
|---|---|
| **SQLite file on device (`database.db`)** | Zero server install on Pi; single file is easy to back up/purge after the capstone; matches “appliance” mental model |
| **WAL (`PRAGMA journal_mode=WAL;`)** | Readers (load face vectors) do not block writers (insert attendance) during morning rush — the exact failure mode observed with default locking |
| **Not a heavy remote DB as the only store** | Remote SQL Server is useful in lab/internship setups, but a door appliance cannot freeze when WAN/VPN dies |
| **Not a pure JSON/CSV log** | Need indexed student IDs, transactional inserts, and concurrent access — SQLite provides that with minimal RAM |

#### Schema (what is stored)
- **`vectormathocsinh`:** `mahocsinh`, `tenhocsinh`, Fernet-encrypted JSON face vectors (`facevector`).
- **`thongtinchamcong`:** attendance rows (`id`, `mahocsinh`, `loai_chamcong`, `thoigianquet`).

#### Why Fernet / AES-128 (and why this processing path)
Fernet (PyCA `cryptography`) uses **AES-128-CBC** for confidentiality plus **HMAC** for integrity (detects tampering).

| Question | Project-specific answer |
|---|---|
| **Why encrypt at all?** | Face vectors are still biometric identifiers. Encryption-at-rest protects a stolen microSD / copied `database.db`. |
| **Why AES-128 (Fernet) rather than hand-rolled AES-256?** | Fernet is a *high-level, misuse-resistant* construction: correct IV handling, authentication, and key format. On a Pi 5, AES-128 is computationally light for small JSON vectors and widely audited. Rolling custom AES-256 without HMAC would be *less* safe in practice. |
| **Why a key in `.env`?** | Keeps secrets out of Git; key is auto-generated on first run; rotation/purge is operationally simple for a school pilot. |
| **Special processing path** | Encrypt only the vector payload (not the whole DB) → fast inserts; decrypt on load into RAM for matching; warn if legacy plaintext JSON is detected; never commit `.env`. |

#### Atomic Offline Fallback Queue (`save_offline_log` & `sync_offline_queue`)
Special failure handling (why each step exists):

1. If SQLite fails / offline mode triggers → append to `offline_queue.json` under `offline_lock` (prevents two threads from corrupting the file).
2. Background thread polls every **10 seconds**.
3. **Rename** `offline_queue.json` → `offline_queue_sync.json` (atomic hand-off so new check-ins can keep writing).
4. **Batch-insert** into `thongtinchamcong` inside one transaction (`commit`).
5. **Delete** sync file on success; on failure, **merge back** unsynced rows (fix `8693d69` race-condition repair).

**Project-level contribution:** This is how the research claim “zero lost check-ins during outages” is engineered — not hoped for.

---

## 5. EXPERIMENTAL DESIGN, TESTING, ETHICS & PRIVACY

### 5.0 Experimental Goal (what Section 5 is trying to prove)

The primary experimental goal is **anti-spoofing (presentation-attack) resistance under realistic school conditions**, while confirming that the appliance still **functions end-to-end** (enroll → recognize → log → recover offline) at acceptable speed.

In one sentence:  
**Sit down, try to fool the system with paper/phone attacks, tune thresholds until real students pass and spoofs fail (the “sweet spot”), then measure functional pass rate and document every bug fix that came from that process.**

### 5.1 Presentation Attack Testing Protocol

#### 5.1.1 2D Paper Print Attack
- **Setup:** Color portrait photos printed on matte/glossy paper (≈1:1 scale), held at natural check-in distance.
- **What is measured:** Does Tier-1 depth (σ<sub>z</sub> ≤ 0.015) and/or Tier-2 motion reject the print?
- **Target metric:** <span style="background-color:#FFFF00;color:#FF0000;">APCER ≤ 3.0%</span>

#### 5.1.2 2D Mobile Screen Replay Attack
- **Setup:** High-resolution still/video playback on a smartphone screen.
- **What is measured:** ICD-normalized motion + enrollment EAR blink curve (for register-time attacks).
- **Target metric:** <span style="background-color:#FFFF00;color:#FF0000;">APCER ≤ 5.0%</span>

### 5.2 Coefficient / Threshold Tuning Process (how the “sweet spot” was approached)

This project did **not** pick final numbers on day one. Thresholds were moved in small steps, guided by false accepts (wrong person / photo passes) vs false rejects (real student fails). The Git history is the lab notebook:

| Stage (commit trail) | Change | Why it was tried | Outcome / next move |
|---|---|---|---|
| Early matching | Cosine-related threshold around **0.25** | Need anyone recognizable at all | Too loose → cross-recognition risk |
| `65b739e` | Lower toward **0.18** + distance debug logs | Stop matching the wrong student | Better separation; still not strict enough for spoof resistance |
| `b295e50` | Pixel-space normalization + blink-at-enroll; threshold **0.13** | Photos were still too close to live faces in vector space | Stronger anti-photo; some real faces rejected |
| `5e65629` | Raise recognition threshold to **0.15** | Recover True Accept Rate for real students | Temporary operating point during internship tests |
| Later Edge calibration | Move toward **`COSINE_THRESHOLD = 0.025`** (≥97.5% similarity) after L2 unit-vector normalization matured | With proper normalization, a tighter angular threshold becomes meaningful | <span style="background-color:#FFFF00;color:#FF0000;">Confirm final TAR/FAR on your recorded trial sheet before submission</span> |
| Liveness depth | σ<sub>z</sub> gate **0.015** | Separate flat media from cranial depth | Tuned against print/screen trials |
| Temporal motion | noise **0.002** / human **0.015**, buffer **3 frames** | Ignore sensor jitter; require biological micro-motion | Prevents “frozen photo” passes |
| EAR blink | closed **&lt;0.15** → open **&gt;0.22** | Force a dynamic blink at enrollment | Glasses edge case fixed in `00b46f6` |

**How to read this as a non-technical reviewer:**  
Each number is a *dial*. Turning the dial tighter blocks fakes but may reject real people; turning it looser does the opposite. The research work is finding the dial position where both goals are “good enough” for a school door — then locking that position in `config.py` / constants and re-testing.

### 5.3 Functional Testing & Test-Case Pass Rate

Functional tests check that the *product* works, not only that a math formula looks right.

| ID | Test case | Expected result | <span style="background-color:#FFFF00;color:#FF0000;">Status / notes (fill measured %)</span> |
|---|---|---|---|
| F01 | Cold start Pi + camera open | Preview within warm-up retry window | <span style="background-color:#FFFF00;color:#FF0000;">Pass/Fail — __%</span> |
| F02 | Enroll with 3 poses + blink | Encrypted vector row created | <span style="background-color:#FFFF00;color:#FF0000;">Pass/Fail — __%</span> |
| F03 | Genuine check-in under threshold | Attendance row written (`VAO`/`RA`) | <span style="background-color:#FFFF00;color:#FF0000;">Pass/Fail — __%</span> |
| F04 | Paper-print attack | Rejected by liveness | <span style="background-color:#FFFF00;color:#FF0000;">Pass/Fail — __%</span> |
| F05 | Phone-screen attack | Rejected by liveness / blink rules | <span style="background-color:#FFFF00;color:#FF0000;">Pass/Fail — __%</span> |
| F06 | Wrong-person attempt | No attendance write | <span style="background-color:#FFFF00;color:#FF0000;">Pass/Fail — __%</span> |
| F07 | Simulated DB/network failure | Record appears in `offline_queue.json` | <span style="background-color:#FFFF00;color:#FF0000;">Pass/Fail — __%</span> |
| F08 | Restore DB / online path | Queue drains; row lands in `thongtinchamcong` | <span style="background-color:#FFFF00;color:#FF0000;">Pass/Fail — __%</span> |
| F09 | Opt-out purge (`delete_student.py`) | Student vectors + related logs removed | <span style="background-color:#FFFF00;color:#FF0000;">Pass/Fail — __%</span> |
| F10 | Cooldown (15 min) | Duplicate spam check-ins suppressed | <span style="background-color:#FFFF00;color:#FF0000;">Pass/Fail — __%</span> |

<span style="background-color:#FFFF00;color:#FF0000;">**Overall functional pass rate (recorded trials): __ / __ cases = __% — replace before submission.**</span>

### 5.4 Defects Encountered and How They Were Fixed (commit-backed)

| Defect (what a user would see) | Root cause | Fix (commit / code) |
|---|---|---|
| ~2 s video lag on Pi | YUYV USB backlog | MJPG + threaded buffer=1 (`camera_stream.py`, RPi5 optimization commits) |
| App crash after dependency install | NumPy 2.x vs MediaPipe ABI; `dlib`/AVX | Pin wheels in `requirements-rpi5.txt`; prefer MediaPipe path |
| Real student with glasses fails blink enroll | EAR landmarks unstable with lenses/reflections | Liveness tolerance / feature path fix (`00b46f6`) |
| Wrong student matched | Threshold too loose / unnormalized vectors | Progressive threshold + L2 normalization commits (`65b739e`, `b295e50`, …) |
| Camera UI freezes while typing student ID | Blocking terminal input on main thread | Async terminal input thread (`6f50536`) |
| Offline records lost or duplicated | Race on queue file while syncing | Atomic rename + merge-back (`8693d69`) |
| Encryption fallback weakened security | Static Fernet key fallback | Removed static fallback; enforce `.env` key (`8693d69`) |
| Cosine distance shape bugs | Missing `.flatten()` | Explicit flatten (`6f50536`) |

### 5.5 Two-Stage Validation Strategy
1. **Stage 1 — Public Dataset Benchmark (Weeks 3–4):** Evaluate vector matching / Cosine thresholds on LFW.  
2. **Stage 2 — Institutional Pilot (Weeks 9–10):** Deploy the physical appliance for a pilot with voluntary participants under consent.  
<span style="background-color:#FFFF00;color:#FF0000;">Record final N, dates, and measured TAR/APCER from the pilot logbook here before submission.</span>

### 5.6 Ethics, Consent & Legal Compliance
- **Institutional Approval:** Protocols submitted to school administration for capstone ethics review before enrollment.
- **Regulatory Alignment:** FERPA (34 CFR Part 99) and BIPA (740 ILCS 14/) biometric handling expectations.
- **Informed Written Consent** must state:
  - Only 1434-D landmark vectors are stored; no raw photos/videos retained.
  - Vectors encrypted with Fernet (AES-128) keys in `.env`.
  - Participation is voluntary; opt-out runs `delete_student.py`.
  - Records purged after capstone defense.

---

## 6. PROJECT TIMELINE, BUDGET & FUTURE SCOPE

### 6.1 12-Week Project Development Timeline (rewritten with data testing)

The project developed during Summer 2026 through a 12-week research and exploration period at Affina (InsurTech), beginning with observation of real-world operational workflows and identification of practical attendance-management challenges. Those observations led to an Edge AI biometric attendance concept, then technical research, architecture design, and a four-week solution implementation **and structured testing** phase.

| Phase | Timeline | Development & Testing Activities |
|---|---|---|
| **1. Industry Exploration & Problem Discovery** | Week 1 | Explored attendance / workforce workflows at Affina; identified privacy, connectivity, and on-device automation gaps. |
| **2. Exploration, Setup, Benchmark Prep & Ethics** | Weeks 2–4 | Edge AI / embedded CV research; OS + `setup_rpi5.sh`; consent drafting; V4L2 MJPG `ThreadedCamera` tuning; MediaPipe mesh ingestion; **LFW-oriented matching data tests** (threshold sweeps, confusion checks). |
| **3. Liveness Heuristic Tuning & Anti-Spoof Data Tests** | Weeks 5–6 | Z-depth / ICD motion / EAR calibration; **paper-print & phone-screen attack trials**; glasses edge-case fixes; log false accept/reject while moving coefficients toward the sweet spot. |
| **4. Storage Resilience, Encryption & Offline Queue Tests** | Weeks 7–8 | SQLite WAL schema; Fernet integration; **48-hour offline queue simulation**; race-condition fixes; verify zero lost records after sync. |
| **5. Functional Regression + On-Premise Pilot Implementation** | Weeks 9–10 | Run full functional suite (F01–F10); hardware deployment; live attendance logging; thermal/performance profiling; collect pilot metrics. |
| **6. Capstone Defense, Metric Lock-in & Data Purge** | Weeks 11–12 | Compile TAR / latency / APCER / offline-recovery metrics; finalize documentation & README; oral defense; cryptographic DB purge. |

### 6.2 Hardware Bill of Materials (BOM) & Direct Project Cost

| Component | Technical Specification | Procurement Source | Direct Cost |
|---|---|---|---|
| Compute Core | Raspberry Pi 5 (8GB LPDDR4X), Broadcom BCM2712 @ 2.4 GHz | Academic Lab Loan (School Inventory) | $0.00 |
| Power & Thermal | Official Raspberry Pi 27W USB-C PSU + Active Cooler Fan | Academic Lab Loan (School Inventory) | $0.00 |
| Optical Camera | Logitech Brio 100 HD USB Webcam (1280×720 @ 30 FPS MJPG) | Direct Out-of-Pocket Purchase | $35.00 |
| Storage Medium | 64GB MicroSD Class 10 (A2/U3) | Personal Inventory | $0.00 |
| Serial Debugging | CP2102 USB-to-TTL Adapter (UART) | Personal/Lab Inventory | $0.00 |
| **TOTAL DIRECT PROJECT EXPENDITURE** | | | **$35.00** |

---

## 7. REFERENCES & TECHNICAL RESOURCES

### Company & Project Context
- Affina Vietnam (InsurTech) — Official website: https://www.affina.com.vn

### Project Media Coverage
- VnExpress International — Education: https://e.vnexpress.net/news/news/education/grade-11-student-develops-edge-ai-attendance-system-from-real-world-problem-5108390.html — “Grade 11 student develops Edge AI attendance system from a real-world problem.” August 13, 2026.

### Academic & Technical References
1. Lugaresi, C., et al. (2019). MediaPipe: A Framework for Building Perception Pipelines. arXiv:1906.08172.
2. Soukupová, T., & Čech, J. (2016). Real-Time Eye Blink Detection using Facial Landmarks. Computer Vision Winter Workshop.
3. Rosebrock, A. (2015). Increasing webcam FPS with Python and OpenCV. PyImageSearch.
4. SQLite Consortium. (2024). Write-Ahead Logging (WAL) Mode Specification. SQLite Official Documentation.
5. Python Cryptography Authority. (2024). Fernet (Symmetric Encryption) Specification. PyCA Documentation.
6. Linux TV Developers. (2024). Video for Linux Two (V4L2) API Driver Specification. Linux Kernel Documentation.
7. Huang, G. B., et al. (2007). Labeled Faces in the Wild. University of Massachusetts Technical Report.
8. U.S. Department of Education. (2021). FERPA Regulations: 34 CFR Part 99.
9. General Assembly of the State of Illinois. (2008). BIPA: 740 ILCS 14/.

### Video Technical Resources & Reference Links
- Database Foundations & Systems: https://www.youtube.com/watch?v=V7ZU0RKtSoQ
- Architecture & Programming Techniques: https://www.youtube.com/watch?v=uHpzKcm8qh0
- Data Processing & Backend Systems: https://www.youtube.com/watch?v=cv7EPNXXTxA
- Video Streaming & Camera Protocols: https://www.youtube.com/watch?v=_FErnBwMpj8
- System Deployment & Optimization: https://www.youtube.com/watch?v=4FWrmuKQq1s
- Linux Kernel & Hardware Interfaces: https://www.youtube.com/watch?v=sRNN_wvw1Zo
- Object detection for Web — ML on Web with MediaPipe (Episode 1 → 12): https://www.youtube.com/watch?v=LF7Lgz4_lus
