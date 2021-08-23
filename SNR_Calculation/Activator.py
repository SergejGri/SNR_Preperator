class Activator:
    def __init__(self, path_base: str):
        self.path_base = path_base
        self.curves = {}
        #self.d = d
        #self.d_mm = f'{self.d} mm'
        self.read_files()

    def read_files(self):
        for file in os.listdir(self.path_base):
            if os.path.isfile(os.path.join(self.path_base, file)) and file.endswith('.csv'):
                filename, int_filename, _, _ = SNRMapGenerator.get_properties(file)
                curve = np.genfromtxt(os.path.join(self.path_base, f'{filename}.csv'), delimiter=',')
                if int_filename is not None:
                    self.curves[f'{int_filename}'] = curve
                else:
                    self.curves[f'{filename}'] = curve