import sqlite3
import time
import datetime
import random
import marshal
import numpy as np


def create_table():
    conn = sqlite3.connect('Face_Database.db')
    c = conn.cursor()
    print("Connected to SQLite")
    c.execute("CREATE TABLE IF NOT EXISTS new_employee ( name TEXT NOT NULL, vector BLOB NOT NULL);")
    c.close()
    conn.close()


def insertBLOB(name,vector):
    try:
        conn = sqlite3.connect('Face_Database.db')
        c = conn.cursor()
        print("Connected to SQLite")
        sql_fetch_blob_query = """SELECT vector from new_employee where name = ?"""
        c.execute(sql_fetch_blob_query, (name,))
        data = c.fetchall()
        if not data:
            vector = vector.tolist()
            data = marshal.dumps(vector[0])
            c.execute("INSERT INTO new_employee (name,vector) VALUES (?, ?)",(name,data))
            print("New Member Registered")
        else:
            print("Already Registered")
        conn.commit()
        c.close()

    except sqlite3.Error as error:
        print("Failed to insert blob data into sqlite table", error)

    finally:
        if (conn):
            conn.close()
            print("sqlite connection is closed")



def readAllBlobData():
    try:
        conn = sqlite3.connect('Face_Database.db')
        c = conn.cursor()
        print("Connected to SQLite")
        sql_fetch_blob_query = """SELECT * FROM new_employee"""
        c.execute(sql_fetch_blob_query)
        data = c.fetchall()
        list_embed = []
        list_emp = []
        array_embed = None
        if data:
            for row in data:
                list_emp.append(row[0])
                list_embed.append(marshal.loads(row[1]))

            array_embed = np.array(list_embed).reshape(len(list_emp),512)
            print(array_embed.shape)

        else:
            print("No Record Found")
        c.close()

    except sqlite3.Error as error:
        print("Failed to read blob data from sqlite table", error)
    finally:
        if (conn):
            conn.close()
            print("sqlite connection is closed")
            return array_embed, list_emp



def readBlobData(empId):
    try:
        conn = sqlite3.connect('Face_Database.db')
        c = conn.cursor()
        print("Connected to SQLite")
        sql_fetch_blob_query = """SELECT vector from new_employee where name = ?"""
        c.execute(sql_fetch_blob_query, (empId,))
        data = c.fetchall()
        if data:
            arr = marshal.loads(data[0][0])
            print(arr)
        else:
            print("Not Recognised")
        c.close()

    except sqlite3.Error as error:
        print("Failed to read blob data from sqlite table", error)
    finally:
        if (conn):
            conn.close()
            print("sqlite connection is closed")


def deleteBlob(name):
    try:
        conn = sqlite3.connect('Face_Database.db')
        c = conn.cursor()
        print("Connected to SQLite")
        sql_fetch_blob_query = """DELETE FROM new_employee WHERE name = ?"""
        c.execute(sql_fetch_blob_query, (name,))
        conn.commit()
        c.close()

    except sqlite3.Error as error:
        print("Failed to read blob data from sqlite table", error)
    finally:
        if (conn):
            conn.close()
            print("sqlite connection is closed")

