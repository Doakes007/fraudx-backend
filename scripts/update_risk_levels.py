import sys
import os

# 🔧 Add backend path so db_config and other modules can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db.db_config import get_connection

def update_user_risks(last_n=10):
    conn = get_connection()
    cur = conn.cursor()

    # Get all user IDs
    cur.execute("SELECT id FROM users")
    users = cur.fetchall()

    for user in users:
        user_id = user[0]

        # Get last N transactions
        cur.execute("""
            SELECT is_fraud
            FROM transactions
            WHERE from_id = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """, (user_id, last_n))

        txns = cur.fetchall()
        if not txns:
            continue

        frauds = sum(1 for row in txns if row[0])
        total = len(txns)
        rate = frauds / total

        if rate >= 0.5:
            risk = "high"
        elif rate >= 0.2:
            risk = "medium"
        else:
            risk = "low"

        cur.execute("UPDATE users SET risk_level = %s WHERE id = %s", (risk, user_id))
        print(f"User {user_id}: {rate*100:.1f}% frauds → {risk}")

    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    update_user_risks()

