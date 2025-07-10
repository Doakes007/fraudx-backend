from db.db_config import get_connection
from datetime import datetime

def is_known_device(user_id, device_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM user_devices WHERE user_id = %s AND device_id = %s", (user_id, device_id))
    exists = cur.fetchone()
    cur.close()
    conn.close()
    return bool(exists)

def register_device_if_new(user_id, device_id, location=None):
    if not is_known_device(user_id, device_id):
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO user_devices (user_id, device_id, location, last_login) VALUES (%s, %s, %s, %s)",
            (user_id, device_id, location, datetime.utcnow())
        )
        conn.commit()
        cur.close()
        conn.close()
    else:
        # Just update login timestamp
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE user_devices SET last_login = %s WHERE user_id = %s AND device_id = %s",
            (datetime.utcnow(), user_id, device_id)
        )
        conn.commit()
        cur.close()
        conn.close()

