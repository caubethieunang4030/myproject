import pymssql
import os
import sys
from dotenv import load_dotenv

# Tải cấu hình từ file .env
load_dotenv()

db_password = os.getenv('DB_PASSWORD')
if not db_password:
    print("\033[91m[ERROR] Biến môi trường DB_PASSWORD chưa được cấu hình trong file .env!\033[0m")
    sys.exit(1)

db_config = {
    'server': os.getenv('DB_SERVER', '127.0.0.1'),
    'port': int(os.getenv('DB_PORT', 1433)),
    'user': os.getenv('DB_USER', 'sa'),
    'password': db_password,
    'database': os.getenv('DB_DATABASE', 'chamcongdatabase')
}

def delete_student(ma_hs):
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
        
        # 1. Kiểm tra sự tồn tại của học sinh
        cursor.execute("select tenhocsinh from vectormathocsinh where mahocsinh = %s", (ma_hs,))
        row = cursor.fetchone()
        if not row:
            print(f"\033[91m[!] Không tìm thấy học sinh có mã '{ma_hs}' trong database.\033[0m")
            return
            
        ten_hs = row[0]
        print(f"\033[93m[*] Tìm thấy học sinh: {ten_hs} ({ma_hs})\033[0m")
        
        # 2. Xóa lịch sử chấm công trước (để tránh lỗi khóa ngoại nếu có)
        cursor.execute("delete from thongtinchamcong where mahocsinh = %s", (ma_hs,))
        print(f"[+] Đã xóa lịch sử chấm công của học sinh '{ten_hs}'")
        
        # 3. Xóa thông tin khuôn mặt
        cursor.execute("delete from vectormathocsinh where mahocsinh = %s", (ma_hs,))
        print(f"[+] Đã xóa dữ liệu khuôn mặt học sinh '{ten_hs}'")
        
        conn.commit()
        print(f"\033[92m[SUCCESS] Đã xóa hoàn toàn học sinh '{ten_hs}' ({ma_hs}) khỏi database!\033[0m")
        
    except Exception as e:
        print(f"\033[91m[ERROR] Lỗi thực thi: {e}\033[0m")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Sử dụng: python delete_student.py <ma_hoc_sinh>")
        print("Ví dụ:  python delete_student.py hs001")
        sys.exit(1)
        
    ma_hs_can_xoa = sys.argv[1].strip()
    delete_student(ma_hs_can_xoa)
