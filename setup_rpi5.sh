#!/bin/bash

echo "=========================================================="
echo "🚀 Khởi tạo Môi trường Hệ thống Chấm công trên Raspberry Pi 5"
echo "   Camera Target: Logitech Brio 100"
echo "=========================================================="

# 1. Cài đặt các gói hệ thống bắt buộc qua apt
echo "[1/4] Đang cài đặt thư viện hệ thống (OpenCV & V4L2 dependencies)..."
sudo apt update
sudo apt install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0 v4l-utils ffmpeg

# 2. Tạo virtualenv nếu chưa có
if [ ! -d "venv" ]; then
    echo "[2/4] Đang tạo môi trường ảo Python (venv)..."
    python3 -m venv venv
else
    echo "[2/4] Môi trường ảo (venv) đã tồn tại."
fi

# 3. Kích hoạt venv và cài đặt packages
echo "[3/4] Đang cài đặt Python packages từ requirements-rpi5.txt..."
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements-rpi5.txt

# 4. Kiểm tra camera Logitech Brio 100 qua v4l2-ctl
echo "[4/4] Kiểm tra danh sách thiết bị Video qua v4l2-ctl..."
v4l2-ctl --list-devices || true

echo "=========================================================="
echo "✅ Hoàn tất cài đặt!"
echo "Chạy ứng dụng bằng lệnh: source venv/bin/activate && python main.py"
echo "=========================================================="
