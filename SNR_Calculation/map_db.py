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

    def add_data(self, object: object):
        self.c = self.conn.cursor()
        print('test')

        pass

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



def load_MAP(path_db, ds, excl, mode):
    db = DB(path_db)
    a = ds
    b = excl
    c = mode

    MAP = 1     # should be an 2D array
    return MAP





def gimme_mesh(n):
    return np.meshgrid(np.linspace(40, 160, n), np.linspace(0, 1, n+1))


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


def plot_mesh():
    _, y, z = data_points()
    x = np.linspace(40, 160, z.size)
    print(x.shape)
    print(y.shape)
    print(z.shape)
    grid_x, grid_y = np.mgrid[1:2:1j, 0:1:170j]

    points = np.stack([x.ravel(), y.ravel()], -1)  # shape (N, 2) in 2d

    z_griddata = interpolate.griddata(points, z.ravel(), (grid_x, grid_y), method='cubic')

    fig, ax = plt.subplots()
    ax.imshow(z_griddata, interpolation='bicubic')
    plt.plot(points[0], points[1], 'ro')
    plt.show()
    print('test')
