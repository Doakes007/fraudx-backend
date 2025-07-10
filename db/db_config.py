import psycopg2

def get_connection():
    return psycopg2.connect(
        host="localhost",
        dbname="fraudx_db",
        user="postgres",
        password="fraudx123"  # <-- Must match what you just set
    )

