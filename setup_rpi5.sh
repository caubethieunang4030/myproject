#!/bin/bash

echo "=========================================================="
echo "🚀 Set up the attendance system environment on a Raspberry Pi 5"
echo "   Camera Target: Logitech Brio 100"
echo "=========================================================="

# 1. Install the required system packages with apt
echo "[1/4] Installing system libraries (OpenCV and V4L2 dependencies)..."
sudo apt update
sudo apt install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0 v4l-utils ffmpeg

# 2. Create a virtualenv if it does not exist yet
if [ ! -d "venv" ]; then
    echo "[2/4] Creating the Python virtual environment (venv)..."
    python3 -m venv venv
else
    echo "[2/4] The virtual environment (venv) already exists."
fi

# 3. Activate the venv and install packages
echo "[3/4] Installing Python packages from requirements-rpi5.txt..."
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements-rpi5.txt

# 4. Check the Logitech Brio 100 through v4l2-ctl
echo "[4/4] Listing video devices through v4l2-ctl..."
v4l2-ctl --list-devices || true

echo "=========================================================="
echo "✅ Setup complete!"
echo "Run the app with: source venv/bin/activate && python main.py"
echo "=========================================================="
