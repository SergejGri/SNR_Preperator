import sqlite3

class DB:
    def __init__(self, var):
        self.conn = sqlite3.connect('curves.db')
        self.c = self.conn.cursor()
        with self.conn:
            self.c.execute("CREATE TABLE IF NOT EXISTS curve_" + str(var) + "(voltage INTEGER, SNR REAL, T REAL)")


    def add_data(self, d, voltage, SNR=None, T=None):
        self.conn = sqlite3.connect('curves.db')
        self.c = self.conn.cursor()

        if SNR is not None and T is None:
            self.c.execute("SELECT voltage, SNR FROM curves WHERE voltage=? OR SNR=?", (voltage, SNR))
            duplicates = self.c.fetchone()
            if duplicates:
                print('ignoring duplicates')
            else:
                with self.conn:
                    self.c.execute("INSERT INTO curves VALUES (?, ?)", (voltage, SNR))

        elif SNR is None and T is not None:
            name = "curve_" + str(d)
            self.c.execute("SELECT voltage, T FROM " + name + " WHERE voltage=? OR T=?", (voltage, T))
            duplicates = self.c.fetchone()
            if duplicates:
                print('ignoring duplicates')
            else:
                with self.conn:
                    self.c.execute("INSERT INTO " + name + " VALUES (?, ?)", (voltage, T))
        else:
            print('...Error...')

    def get_data(self):
        self.conn = sqlite3.connect('curves.db')
        self.c = self.conn.cursor()
        self.c.execute("SELECT date, value FROM credit")
        rows = self.c.fetchall()
        date_list = [x[0] for x in rows]
        value_list = [x[1] for x in rows]
        self.c.close()
        return date_list, value_list