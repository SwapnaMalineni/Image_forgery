#!/usr/bin/env python3
"""
Create a new user in the app's SQLite database.

Usage:
  python scripts/create_user.py --email user@example.com --password Secret123 --username testuser --mobile 9999999999
  python scripts/create_user.py --email user@example.com --password Secret123 --db instance/database.db

The script looks for `instance/database.db` then `database.db` by default.
"""
import argparse
import os
import sqlite3
import sys

try:
    import bcrypt
except Exception:
    print("Please install the 'bcrypt' Python package in your environment (pip install bcrypt)")
    sys.exit(1)

DEFAULT_DB_CANDIDATES = [
    os.path.join('instance', 'database.db'),
    'database.db'
]


def find_db(path=None):
    if path:
        if os.path.exists(path):
            return path
        print(f"Provided DB path not found: {path}")
        return None
    for p in DEFAULT_DB_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def create_user(db_path, email, password, username, mobile):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Check if table exists
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user'")
        if not cur.fetchone():
            print('User table not found in DB. Is this the correct database?')
            conn.close()
            return
    except Exception as e:
        print('Error checking database:', e)
        conn.close()
        return

    # Check for existing email
    cur.execute('SELECT id FROM user WHERE email = ?', (email,))
    if cur.fetchone():
        print('A user with that email already exists.')
        conn.close()
        return

    # Hash password using bcrypt
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    try:
        cur.execute('INSERT INTO user (username, email, mobile, password) VALUES (?, ?, ?, ?)',
                    (username, email, mobile, hashed))
        conn.commit()
        print('User created successfully.')
    except Exception as e:
        print('Error inserting user:', e)
    finally:
        conn.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', help='Path to sqlite db file (overrides default)')
    ap.add_argument('--email', required=True)
    ap.add_argument('--password', required=True)
    ap.add_argument('--username', default='newuser')
    ap.add_argument('--mobile', default='0000000000')
    args = ap.parse_args()

    db = find_db(args.db)
    if not db:
        print('No database file found. Looked for:', DEFAULT_DB_CANDIDATES)
        sys.exit(1)

    create_user(db, args.email, args.password, args.username, args.mobile)
