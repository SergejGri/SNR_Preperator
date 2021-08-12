import os
from SNR_Calculation.Prepper import *
from SNR_Calculation.SNRMapGenerator import *




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



def create_report(_path):
    _date = datetime.datetime.now()
    _date = _date.strftime('%x')
    with open(os.path.join(_path, f'{_date}_report.txt'), 'w') as f:
        f.write(dict)
        f.close()


def get_dirs(pathh):
    list_dir = []
    for dirr in os.listdir(pathh):
        if os.path.isdir(os.path.join(pathh, dirr)) and dirr != 'darks':
            list_dir.append(dirr)
    return list_dir


def get_d():
    thick_0 = [4, 8, 12, 16, 20, 24, 28, 32]
    thick_1 = [5, 9, 13, 17, 21, 25, 29, 33]
    thick_2 = [6, 10, 14, 18, 22, 26, 30, 34]
    thicknesses = [thick_0, thick_1, thick_2]
    return thicknesses


def current_d(pattern):
    if pattern == '_0mm Al_':
        list_left_0mm = ['4', '12', '20', '28']
        list_right_0mm = ['8', '16', '24', '32']
        return list_left_0mm, list_right_0mm
    if pattern == '_1mm Al_':
        list_left_1mm = ['5', '13', '21', '29']
        list_right_1mm = ['9', '17', '25', '33']
        return list_left_1mm, list_right_1mm
    if pattern == '_2mm Al_':
        list_left_2mm = ['6', '14', '22', '30']
        list_right_2mm = ['10', '18', '26', '34']
        return list_left_2mm, list_right_2mm


def get_df():
    list_df = ['_0mm Al_', '_1mm Al_', '_2mm Al_']
    return list_df


def get_areas():
    list_areas = ['_1-area_', '_2-area_', '_3-area_', '_4-area_']
    return list_areas


def prep_data():
    path = r'\\132.187.193.8\junk\sgrischagin\2021-06-24-Sergej_AluKeil_0-1-2mm-AluFilter'
    res_path = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR'
    image_shape = (1536, 1944)
    header = 2048
    M = 21.2808

    calc = SNRCalculator(img_shape=image_shape,
                         header=header,
                         path=path,
                         magnification=M,
                         t_exp=1200,
                         res_path=res_path, mode_SNR=False)

    dirs = get_dirs(path)
    filters = get_df()
    areas = get_areas()

    for dir in dirs:
        for f in filters:
            for area in areas:
                list_l, list_r = current_d(f)
                for l, r in zip(list_l, list_r):
                    calc(dir=dir, df=f, dl=l, dr=r, area=area)


def calc_curves():
    path_snr = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\newApproach_0-1-2mmAlFilter'
    path_T = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\Transmission_v2'
    path_fin = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\MAP_v3'

    thickness = get_d()
    for i in range(len(thickness)):
        for j in range(len(thickness[i])):
            map = SNRMapGenerator(path_snr=path_snr, path_T=path_T, path_fin=path_fin, d=thickness[i][j])
            map()


def main():
    prep_data()
    calc_curves()



if __name__ == '__main__':
    main()
