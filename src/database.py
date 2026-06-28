import sqlite3
import hashlib
import os

DB_PATH = "data/parking_system.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Veritabanını ve tabloları oluştur"""
    os.makedirs("data", exist_ok=True)
    conn = get_connection()
    c = conn.cursor()

    # Kullanıcılar tablosu
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT UNIQUE NOT NULL,
            password  TEXT NOT NULL,
            role      TEXT NOT NULL DEFAULT 'user',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Analiz kayıtları tablosu
    c.execute("""
        CREATE TABLE IF NOT EXISTS analysis_logs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            username      TEXT NOT NULL,
            camera        TEXT NOT NULL,
            source_type   TEXT NOT NULL,
            total_spots   INTEGER,
            occupied      INTEGER,
            empty         INTEGER,
            occupancy_rate REAL,
            timestamp     TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Admin kullanıcısı yoksa oluştur
    admin_exists = c.execute(
        "SELECT id FROM users WHERE username = 'admin'"
    ).fetchone()

    if not admin_exists:
        c.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            ("admin", hash_password("admin123"), "admin")
        )
        print("Admin kullanicisi olusturuldu. Sifre: admin123")

    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username, password):
    """Kullanıcı doğrula, (id, username, role) döndür veya None"""
    conn = get_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE username = ? AND password = ?",
        (username, hash_password(password))
    ).fetchone()
    conn.close()
    return dict(user) if user else None

def get_all_users():
    conn = get_connection()
    users = conn.execute(
        "SELECT id, username, role, created_at FROM users ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(u) for u in users]

def add_user(username, password, role="user"):
    try:
        conn = get_connection()
        conn.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, hash_password(password), role)
        )
        conn.commit()
        conn.close()
        return True, "Kullanici basariyla eklendi."
    except sqlite3.IntegrityError:
        return False, "Bu kullanici adi zaten mevcut."

def delete_user(user_id):
    conn = get_connection()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

def log_analysis(user_id, username, camera, source_type, stats):
    """Analiz sonucunu kaydet"""
    conn = get_connection()
    conn.execute("""
        INSERT INTO analysis_logs
            (user_id, username, camera, source_type, total_spots, occupied, empty, occupancy_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, username, camera, source_type,
        stats["total"], stats["occupied"], stats["empty"], stats["occupancy_rate"]
    ))
    conn.commit()
    conn.close()

def get_logs(user_id=None, limit=100):
    """Analiz loglarını getir. Admin ise hepsini, kullanıcı ise sadece kendini."""
    conn = get_connection()
    if user_id:
        logs = conn.execute("""
            SELECT * FROM analysis_logs
            WHERE user_id = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (user_id, limit)).fetchall()
    else:
        logs = conn.execute("""
            SELECT * FROM analysis_logs
            ORDER BY timestamp DESC LIMIT ?
        """, (limit,)).fetchall()
    conn.close()
    return [dict(row) for row in logs]

def get_stats_by_hour(camera=None):
    """Saatlik ortalama doluluk istatistiği"""
    conn = get_connection()
    if camera:
        rows = conn.execute("""
            SELECT strftime('%H', timestamp) as hour,
                   AVG(occupancy_rate) as avg_rate,
                   COUNT(*) as count
            FROM analysis_logs
            WHERE camera = ?
            GROUP BY hour
            ORDER BY hour
        """, (camera,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT strftime('%H', timestamp) as hour,
                   AVG(occupancy_rate) as avg_rate,
                   COUNT(*) as count
            FROM analysis_logs
            GROUP BY hour
            ORDER BY hour
        """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

if __name__ == "__main__":
    init_db()
    print("Veritabani hazir.")