import pymssql
import os
import sys
from dotenv import load_dotenv

# Load configuration from the .env file
load_dotenv()

db_password = os.getenv('DB_PASSWORD')
if not db_password:
    print("\033[91m[ERROR] The DB_PASSWORD environment variable is not set in the .env file!\033[0m")
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
        
        # 1. Check that the student exists
        cursor.execute("select tenhocsinh from vectormathocsinh where mahocsinh = %s", (ma_hs,))
        row = cursor.fetchone()
        if not row:
            print(f"\033[91m[!] No student found with id '{ma_hs}' in the database.\033[0m")
            return
            
        ten_hs = row[0]
        print(f"\033[93m[*] Found student: {ten_hs} ({ma_hs})\033[0m")
        
        # 2. Delete attendance history first (avoids a foreign-key error if one exists)
        cursor.execute("delete from thongtinchamcong where mahocsinh = %s", (ma_hs,))
        print(f"[+] Deleted attendance history for student '{ten_hs}'")
        
        # 3. Delete the face record
        cursor.execute("delete from vectormathocsinh where mahocsinh = %s", (ma_hs,))
        print(f"[+] Deleted face data for student '{ten_hs}'")
        
        conn.commit()
        print(f"\033[92m[SUCCESS] Fully deleted student '{ten_hs}' ({ma_hs}) khỏi database!\033[0m")
        
    except Exception as e:
        print(f"\033[91m[ERROR] Execution error: {e}\033[0m")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python delete_student.py <student_id>")
        print("Example: python delete_student.py hs001")
        sys.exit(1)
        
    ma_hs_can_xoa = sys.argv[1].strip()
    delete_student(ma_hs_can_xoa)
