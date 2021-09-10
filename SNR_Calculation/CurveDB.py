import sqlite3
import os
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt

class DB:
    def __init__(self, path_DB: str):
        self.path_DB = path_DB
        self.conn = sqlite3.connect(os.path.join(self.path_DB, 'curves.db'))
        self.c = self.conn.cursor()
        self.d = None

    def create_table(self, d):
        d = str(d)
        with self.conn:
            self.c.execute("CREATE TABLE IF NOT EXISTS curve_" + d + "(voltage REAL, T REAL, SNR REAL)")

    def add_data(self, d, voltage, SNR=None, T=None, mode=None):
        self.c = self.conn.cursor()
        d = str(d)
        if mode == 'raw':
            self.c.execute("CREATE TABLE IF NOT EXISTS curve_" + d + "(voltage REAL, T REAL, SNR REAL)")
            self.c.execute("SELECT T, SNR FROM curve_" + str(d) + " WHERE T=? OR SNR=?", (T, SNR))
            duplicates = self.c.fetchone()
            if duplicates:
                print(f'ignoring duplicates: {d}mm -> {duplicates}')
            else:
                with self.conn:
                    self.c.execute("INSERT INTO curve_" + d + " VALUES (?, ?, ?)", (voltage, T, SNR))
        if mode == 'fit':
            self.c.execute("CREATE TABLE IF NOT EXISTS curve_fit_" + d + "(voltage REAL, T REAL, SNR REAL)")
            self.c.execute("SELECT T, SNR FROM curve_fit_" + str(d) + " WHERE T=? OR SNR=?", (T, SNR))
            duplicates = self.c.fetchone()
            if duplicates:
                print(f'ignoring duplicates: {d}mm -> {duplicates}')
            else:
                with self.conn:
                    self.c.execute("INSERT INTO curve_fit_" + d + " VALUES (?, ?, ?)", (voltage, T, SNR))

    def read_data(self, d, mode=None):
        self.d = str(d)
        self.c = self.conn.cursor()
        if mode == 'raw':
            self.c.execute("SELECT voltage, T, SNR FROM curve_" + self.d)
        if mode == 'fit':
            self.c.execute("SELECT voltage, T, SNR FROM curve_fit_" + self.d)
        rows = self.c.fetchall()
        list_voltage = [x[0] for x in rows]
        list_T = [x[1] for x in rows]
        list_SNR = [x[2] for x in rows]
        self.c.close()
        return list_voltage, list_T, list_SNR


def create_MAP(path_res, list_ds, mode_fit=False):
    fig = None
    db = DB(path_res)
    for d in list_ds:
        kV, T, SNR = db.read_data(d)
        fig = plt.gcf()
        plt.plot(T, SNR, label=f'{d}mm', marker='o')
        if mode_fit:
            coefs = poly.polyfit(T, SNR, 3)
            x_new = np.linspace(T[0], T[-1], num=len(T) * 10)
            ffit = poly.polyval(x_new, coefs)
            plt.plot(x_new, ffit, '--', c='grey')
        plt.tight_layout()
        plt.xlabel('Transmission a.u.')
        plt.ylabel('SNR')
        plt.legend()
    plt.show()
    fig.savefig(os.path.join(path_res, 'MAP.pdf'), dpi=600)