import sqlite3
import os

# Подключаемся к базе данных
conn = sqlite3.connect('db.sqlite3')
cursor = conn.cursor()

# Получаем список всех таблиц
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Существующие таблицы в базе данных:")
for table in tables:
    print(f"- {table[0]}")

conn.close()