#!/usr/bin/env python3
"""
Small helper to inspect users in the SQLite DB and verify a plaintext password
against the bcrypt hash stored in the `user` table.

Usage:
  python scripts/check_user.py list
  python scripts/check_user.py verify email@example.com PlaintextPassword

The script will try `instance/database.db` first, then `database.db` in the repo root.
"""
import sqlite3
import os
import sys
import argparse

try:
    import bcrypt
except Exception:
    print("bcrypt module not found. Make sure your virtualenv has bcrypt installed.")
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


def list_users(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute('SELECT id, username, email, mobile, password FROM user')
    except Exception as e:
        print('Error reading user table:', e)
        conn.close()
        return
    rows = cur.fetchall()
    if not rows:
        print('No users found in the database.')
    for r in rows:
        print(f'id={r[0]} username={r[1]} email={r[2]} mobile={r[3]} password_hash={r[4][:60]}...')
    conn.close()


def verify_password(db_path, email, password):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute('SELECT password FROM user WHERE email = ?', (email,))
    except Exception as e:
        print('Error querying user table:', e)
        conn.close()
        return
    row = cur.fetchone()
    conn.close()
    if not row:
        print('No user with that email found')
        return
    stored = row[0]
    try:
        ok = bcrypt.checkpw(password.encode('utf-8'), stored.encode('utf-8'))
    except Exception as e:
        print('Error while checking password:', e)
        return
    print('Password match' if ok else 'Password does NOT match')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['list', 'verify'])
    ap.add_argument('email', nargs='?')
    ap.add_argument('password', nargs='?')
    ap.add_argument('--db', help='Path to sqlite db file (overrides default)')
    args = ap.parse_args()

    db = find_db(args.db)
    if not db:
        print('No database file found. Looked for:', DEFAULT_DB_CANDIDATES)
        sys.exit(1)

    if args.cmd == 'list':
        list_users(db)
    elif args.cmd == 'verify':
        if not args.email or not args.password:
            print('verify requires email and password')
            sys.exit(1)
        verify_password(db, args.email, args.password)
