import Prepper as prep
import SNRMapGenerator as snrg
from externe_files import file

def get_d():
    thick_0 = [4, 8, 12, 16, 20, 24, 28, 32]
    thick_1 = [5, 9, 13, 17, 21, 25, 29, 33]
    thick_2 = [6, 10, 14, 18, 22, 26, 30, 34]
    thicknesses = [thick_0, thick_1, thick_2]
    return thicknesses


def calc_files_for_map():
    base_path_fin = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\MAP_v1'
    base_path_snr = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\SNR_evaluation_v2'
    base_path_T = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\Transmission_v2'

    # TODO: implement more robust detection of voltages/thicknesses independent on style of passed strings
    # '160kV == '160_kV' == '160' == '160kv'... // '6mm' == '6_mm' ...
    kV_filter = ['40_kV']
    d_filter = ['6', '16']

    curves = []

    thicknesses = get_d()
    for i in range(len(thicknesses)):
        for j in range(len(thicknesses[0])):
            _d = thicknesses[i][j]
            if _d not in d_filter:
                generator = snrg.SNRMapGenerator(path_snr=base_path_snr,
                                                 path_T=base_path_T,
                                                 path_fin=base_path_fin,
                                                 d=_d,
                                                 kV_filter=kV_filter)
                generator()




def activate_map():
    path_ = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\MAP_v1'
    curves = SNRMapGenerator.Activator(path_base=path_)
    return curves

''''
def plot(curves):
    for curves['10']
    if not os.path.exists(os.path.join(path_map, 'Plots')):
        os.mkdir(os.path.join(path_map, 'Plots'))
    for file in os.listdir(path_map):
        if file.endswith('.csv') and not file.split('.')[0] in excl_filter:
            filename = file.split('m')[0]
            data = np.genfromtxt(os.path.join(path_map, file), delimiter=',')
            max_kv = data[-1][0]
            data_x = data[:, 1]
            data_y = data[:, 2]
            plt.figure(figsize=(14.4, 8.8))
            plt.plot(data_x, data_y, marker='o', label=f'{filename} mm')
            plt.legend()
            plt.xlabel('Transmission a.u.')
            plt.ylabel('SNR')
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.join(path_map, 'Plots'), f'SNR_T_{filename}mm_{max_kv}maxkV.png'))
'''

def main():
    generator = SN


if __name__ == '__main__':
    main()
