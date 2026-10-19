#!/usr/bin/env python3
import os
import sys
import sqlite3
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_PATH = os.getenv("DB_PATH", "database.db")
WEB_API_URL = os.getenv("WEB_API_URL", "http://localhost:8080/api/attendance/sync-pi5")

def sync_attendance():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database file not found at: {DB_PATH}")
        return

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, mahocsinh, loai_chamcong, thoigianquet FROM thongtinchamcong ORDER BY id ASC")
        rows = cursor.fetchall()
        
        if not rows:
            print("ℹ️ There are no attendance records in the database.")
            return

        payload_records = []
        for row in rows:
            rec_id, mahocsinh, loai_chamcong, thoigianquet = row
            payload_records.append({
                "sqlite_id": rec_id,
                "mahocsinh": mahocsinh,
                "loai_chamcong": loai_chamcong or "VAO",
                "thoigianquet": thoigianquet,
                "deviceId": "pi5_brio100"
            })

        print(f"🔄 Syncing {len(payload_records)} records to the web app: {WEB_API_URL} ...")
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(WEB_API_URL, json={"records": payload_records}, headers=headers, timeout=10)
        
        if response.status_code == 200:
            res_data = response.json()
            print(f"✅ Synced {res_data.get('synced', 0)} attendance records to the web server!")
        else:
            print(f"❌ Sync failed (HTTP {response.status_code}): {response.text}")
            
    except Exception as e:
        print(f"❌ Error while connecting or sending data: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    sync_attendance()
