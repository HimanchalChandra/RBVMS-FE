import sqlite3
import time
import datetime
import random
import marshal
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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

def readall_data():
    try:
        conn = sqlite3.connect('student.db')
        c = conn.cursor()
        print("Connected to SQLite")
        sql_fetch_blob_query = """SELECT * from visualize"""
        c.execute(sql_fetch_blob_query)
        data = c.fetchall()

    except sqlite3.Error as error:
        print("Failed to read blob data from sqlite table", error)
    finally:
        if (conn):
            conn.close()
            print("sqlite connection is closed")
            return data

def bar_graph(day):
    test_string = day[4:]
    res = int(test_string)
    x = readall_data()
    names = []
    attend_percent = []
    no_of_ones_per_candidate = []
    for i in x:
        attend_percent.append((list(i[1:]).count(1)/(res-1)*100))
        names.append(i[0])
        no_of_ones_per_candidate.append(list(i[1:]).count(1))

    overall_percent = sum(attend_percent)/len(attend_percent)
    attendance_percent = [overall_percent,100-overall_percent]
    x = names
    percent = attend_percent

    x_pos = [i for i, _ in enumerate(x)]

    plt.bar(x_pos, percent, color='green')
    plt.xlabel("Users")
    plt.ylabel("Percentage Attendance")
    plt.title("Percentage Attendance of Each Individual")

    plt.xticks(x_pos, x)
    plt.tight_layout()
    plt.savefig('bar.png')
    im = Image.open("bar.png")
    im.convert('RGB').save("bar.jpg","JPEG")
    plt.close()

def pie_chart(day):
    test_string = day[4:]
    res = int(test_string)
    x = readall_data()
    names = []
    attend_percent = []
    no_of_ones_per_candidate = []
    for i in x:
        attend_percent.append((list(i[1:]).count(1)/(res-1)*100))
        names.append(i[0])
        no_of_ones_per_candidate.append(list(i[1:]).count(1))

    overall_percent = sum(attend_percent)/len(attend_percent)
    attendance_percent = [overall_percent,100-overall_percent]
    labels = 'Present','Absent'
    sizes = attendance_percent
    plt.title("Overall Attendance in the Organization")
    patches, texts, _ = plt.pie(sizes,autopct='%1.1f%%',startangle=90)
    plt.legend(patches, labels, loc="best")
    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('pie.png')
    im = Image.open("pie.png")
    im.convert('RGB').save("pie.jpg","JPEG")
    plt.close()

def create_attend_table():
    conn = sqlite3.connect('student.db')
    c = conn.cursor()
    print("Connected to SQLite")
    c.execute("CREATE TABLE IF NOT EXISTS visualize ( name TEXT NOT NULL, day_1 INT NULL, day_2 INT NULL, day_3 INT NULL, day_4 INT NULL, day_5 INT NULL, day_6 INT NULL, day_7 INT NULL, day_8 INT NULL, day_9 INT NULL, day_10 INT NULL, day_11 INT NULL, day_12 INT NULL, day_13 INT NULL, day_14 INT NULL, day_15 INT NULL);")
    c.close()
    conn.close()

def insert_data(name, data, day):
    try:
        conn = sqlite3.connect('student.db')
        c = conn.cursor()
        print("Connected to SQLite")
        sql_fetch_blob_query = """SELECT EXISTS(SELECT 1 FROM visualize WHERE name=?)"""
        c.execute(sql_fetch_blob_query, (name,))
        ver = c.fetchall()[0][0]
        if ver:
            print("done1")
            sql_update_query = """Update visualize set {} = {} where name = ?""".format(day,data)
            c.execute(sql_update_query,(name,))
        else:
            print("done2")
            c.execute("INSERT INTO visualize (name,{}) VALUES (?, ?)".format(day),(name,data))
        conn.commit()
        c.close()

    except sqlite3.Error as error:
        print("Failed to insert blob data into sqlite table", error)

    finally:
        if (conn):
            conn.close()
            print("sqlite connection is closed")

def delete_data(name):
    try:
        conn = sqlite3.connect('student.db')
        c = conn.cursor()
        print("Connected to SQLite")
        sql_fetch_blob_query = """DELETE FROM visualize WHERE name = ?"""
        c.execute(sql_fetch_blob_query, (name,))
        conn.commit()
        c.close()

    except sqlite3.Error as error:
        print("Failed to read blob data from sqlite table", error)
    finally:
        if (conn):
            conn.close()
            print("sqlite connection is closed")
