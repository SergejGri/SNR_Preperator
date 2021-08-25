import sqlite3
import os
import matplotlib.pyplot as plt

class DB:
    def __init__(self, d):
        self.conn = sqlite3.connect('curves.db')
        self.c = self.conn.cursor()
        self.d = d
        self.d = str(self.d)
        with self.conn:
            self.c.execute("CREATE TABLE IF NOT EXISTS curve_" + self.d + "(voltage REAL, T REAL, SNR REAL)")

    def add_data(self, voltage, SNR=None, T=None):
        self.conn = sqlite3.connect('curves.db')
        self.c = self.conn.cursor()

        self.c.execute("SELECT T, SNR FROM curve_" + self.d + " WHERE T=? OR SNR=?", (T, SNR))
        duplicates = self.c.fetchone()
        if duplicates:
            print('ignoring duplicates')
        else:
            with self.conn:
                self.c.execute("INSERT INTO curve_" + self.d + " VALUES (?, ?, ?)", (voltage, T, SNR))

    def read_data(self):
        self.conn = sqlite3.connect('curves.db')
        self.c = self.conn.cursor()
        self.c.execute("SELECT voltage, T, SNR FROM curve_" + self.d)
        rows = self.c.fetchall()
        list_voltage = [x[0] for x in rows]
        list_T = [x[1] for x in rows]
        list_SNR = [x[2] for x in rows]
        self.c.close()
        return list_voltage, list_T, list_SNR


def create_MAP(path_res, list_ds):
    fig = None
    for d in list_ds:
        db = DB(d)
        kV, T, SNR = db.read_data()

        fig = plt.gcf()
        plt.plot(T, SNR, label=f'{d}mm')
        plt.xlabel('Transmission a.u.')
        plt.ylabel('SNR')
        plt.legend()
    plt.show()
    fig.savefig(os.path.join(path_res, 'MAP.pdf'), dpi=600)

