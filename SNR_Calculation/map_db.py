import sqlite3
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


class DB:
    def __init__(self, path_DB: str):
        self.path_DB = path_DB
        self.conn = sqlite3.connect(os.path.join(self.path_DB, 'MAP.db'))
        self.c = self.conn.cursor()
        self.d = None

    def create_table(self, d):
        d = str(d)
        with self.conn:
            self.c.execute("CREATE TABLE IF NOT EXISTS curve_" + d + "(voltage REAL, T REAL, SNR REAL)")

    def add_data(self, obj: object):
        self.c = self.conn.cursor()
        curves = obj['curves']

        for _c in curves.keys():
            kV = list(curves[_c]['kV'])
            T = list(curves[_c]['T'])
            SNR = list(curves[_c]['SNR'])

            self.c.execute(f"CREATE TABLE IF NOT EXISTS {_c} (voltage REAL, T REAL, SNR REAL)")
            self.c.execute(f"SELECT T, SNR FROM " + _c + " WHERE T=? OR SNR=?", (T[0], SNR[0]))
            duplicates = self.c.fetchone()
            if duplicates:
                print(f'ignoring duplicates: {_c} -> {duplicates}')
            else:
                with self.conn:
                    for kv, t, snr in zip(kV, T, SNR):
                        self.c.execute(f"INSERT INTO {_c} VALUES (?, ?, ?)", (kv, t, snr))


    def add_data_v2(self, d, voltage, SNR=None, T=None, mode: str = None):
        '''
        :param mode: accepts 'raw' or 'fit' as input. If 'raw' is set, the passed values are going to be stored to the
        unfitted data otherwise to the tables with fitted curves.
        '''
        self.c = self.conn.cursor()
        d = str(d)

        if mode == 'raw':
            self.c.execute("CREATE TABLE IF NOT EXISTS curve_" + d + "(voltage REAL, T REAL, SNR REAL)")
            self.c.execute("SELECT T, SNR FROM curve_" + d + " WHERE T=? OR SNR=?", (T, SNR))
            duplicates = self.c.fetchone()
            if duplicates:
                print(f'ignoring duplicates: {d}mm -> {duplicates}')
            else:
                with self.conn:
                    self.c.execute("INSERT INTO curve_" + d + " VALUES (?, ?, ?)", (voltage, T, SNR))
        if mode == 'fit':
            self.c.execute("CREATE TABLE IF NOT EXISTS curve_fit_" + d + "(voltage REAL, T REAL, SNR REAL)")
            self.c.execute("SELECT T, SNR FROM curve_fit_" + d + " WHERE T=? OR SNR=?", (T, SNR))
            duplicates = self.c.fetchone()
            if duplicates:
                print(f'ignoring duplicates: {d}mm -> {duplicates}')
            else:
                with self.conn:
                    self.c.execute("INSERT INTO curve_fit_" + d + " VALUES (?, ?, ?)", (voltage, T, SNR))
        if mode == 'virtual':
            self.c.execute("CREATE TABLE IF NOT EXISTS curve_vir_" + d + "(voltage REAL, T REAL, SNR REAL)")
            self.c.execute("SELECT T, SNR FROM curve_vir_" + d + " WHERE T=? OR SNR=?", (T, SNR))
            duplicates = self.c.fetchone()
            if duplicates:
                print(f'ignoring duplicates: {d}mm -> {duplicates}')
            else:
                with self.conn:
                    self.c.execute("INSERT INTO curve_vir_" + d + " VALUES (?, ?, ?)", (voltage, T, SNR))


    def read_data(self, d, excl=None, mode=None):
        self.d = str(d)
        self.c = self.conn.cursor()
        table_exists = False
        if mode is None:
            print('No mode for reading specified!')
        if mode == 'raw':
            try:
                self.c.execute("SELECT voltage, T, SNR FROM curve_" + self.d)
                table_exists = True
            except:
                print('Table does not exists.')
        elif mode == 'fit':
            try:
                self.c.execute("SELECT voltage, T, SNR FROM curve_fit_" + self.d)
                table_exists = True
            except:
                print('Table does not exists.')
        elif mode=='virtual':
            try:
                self.c.execute("SELECT voltage, T, SNR FROM curve_vir_" + self.d)
                table_exists = True
            except:
                print('Table does not exists.')


        if table_exists:
            rows = self.c.fetchall()
            list_voltage = [x[0] for x in rows]
            list_T = [x[1] for x in rows]
            list_SNR = [x[2] for x in rows]
            if excl is not None:
                for i in excl:
                    idx = list_voltage.index(i)
                    del list_voltage[idx], list_T[idx], list_SNR[idx]

            self.c.close()
            return list_voltage, list_T, list_SNR

    def table_exists(self, name):
        # Check if there is an column with voltage entries --> existing table
        name = str(name)
        list_of_tables = []
        try:
            list_of_tables = self.c.execute("SELECT voltage FROM "+name+"").fetchall()
        except:
            print('test')
        if not list_of_tables:
            return False
        else:
            return True

'''
def create_MAP(path_res, list_ds, excl, mode):
    fig = None
    db = DB(path_res)
    for d in list_ds:
        kV, T, SNR = db.read_data(d, excl, mode=mode)
        fig = plt.gcf()
        plt.semilogy(T, SNR, label=f'{d}mm', marker='o')
        if mode=='fit':
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
'''



def data_points():
    db = DB(r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-10-01_Sergej_SNR-Stufenkeil_6W_130proj_0-10mm\MAP')
    ds = [0, 2, 4, 6, 8, 10]

    loc_SNR = []
    loc_T = []
    loc_kV = [np.linspace(40, 160, 20)]
    for d in ds:
        kV, T, SNR = db.read_data(d, mode='raw')
        loc_SNR.append(SNR)
        loc_T.append(T)
    T = np.array(loc_T)
    SNR = np.asarray(loc_SNR)
    kV = np.asarray(loc_kV)
    return kV, T, SNR


