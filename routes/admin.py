from flask import Blueprint, jsonify
from db.db_config import get_connection

admin_bp = Blueprint("admin", __name__)

@admin_bp.route("/stats", methods=["GET"])
def get_admin_stats():
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Get total users
        cur.execute("SELECT COUNT(*) FROM users")
        total_users = cur.fetchone()[0]

        # Get total transactions
        cur.execute("SELECT COUNT(*) FROM transactions")
        total_txns = cur.fetchone()[0]

        # Get total fraud transactions
        cur.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = TRUE")
        total_frauds = cur.fetchone()[0]

        # Get risk level breakdown
        cur.execute("""
            SELECT risk_level, COUNT(*) FROM users GROUP BY risk_level
        """)
        risk_data = cur.fetchall()
        risk_breakdown = {level: count for level, count in risk_data}

        cur.close()
        conn.close()

        return jsonify({
            "total_users": total_users,
            "total_transactions": total_txns,
            "total_frauds": total_frauds,
            "risk_breakdown": risk_breakdown
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

