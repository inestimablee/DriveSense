import sqlite3
import datetime

DB_FILE = "drivesense.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            ear         REAL,
            mar         REAL,
            pitch       REAL,
            yaw         REAL,
            phone       INTEGER,
            emotion     TEXT,
            risk_score  INTEGER,
            alert_level INTEGER
        )
    ''')
    conn.commit()
    conn.close()
    print("[DriveSense] Database ready!")

def log_event(ear, mar, pitch, yaw, phone, emotion, risk_score, alert_level):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO events 
        (timestamp, ear, mar, pitch, yaw, phone, emotion, risk_score, alert_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ear, mar, pitch, yaw,
        int(phone), emotion,
        risk_score, alert_level
    ))
    conn.commit()
    conn.close()

def get_recent_events(limit=100):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        SELECT * FROM events 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

def get_weekly_summary():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        SELECT 
            DATE(timestamp) as date,
            COUNT(*) as total_events,
            AVG(risk_score) as avg_risk,
            MAX(risk_score) as max_risk,
            SUM(CASE WHEN alert_level = 2 THEN 1 ELSE 0 END) as critical_alerts,
            SUM(CASE WHEN phone = 1 THEN 1 ELSE 0 END) as phone_events
        FROM events
        WHERE timestamp >= DATE('now', '-7 days')
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
    ''')
    rows = c.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    init_db()