import datetime

import SNR_Calculation.curve_db
from SNR_Calculation.Prepper import *
from SNR_Calculation.SNRMapGenerator import *
from SNR_Calculation.curve_db import *


def create_report(_path):
    _date = datetime.datetime.now()
    _date = _date.strftime('%x')
    with open(os.path.join(_path, f'{_date}_report.txt'), 'w') as f:
        f.write(dict)
        f.close()


def prep_data(path_base, path_result_prep):
    image_shape = (1536, 1944)
    header = 2048
    M = 20.0965
    watt = 4
    excl = ['70kV', '80kV']

    calc = SNRCalculator(img_shape=image_shape, header=header, path=path_base, magnification=M, nbins=300,
                         path_result=path_result_prep, watt=watt, exclude=excl, overwrite=False)

    dirs = SNRCalculator.get_dirs(path_base, excl)
    filters = SNRCalculator.get_df()


    for dir in dirs:
        for fltr in filters:
            calc(dir=dir, df=fltr)


def calc_curves(path_snr, path_T, path_fin):
    thickness = SNRCalculator.get_d()
    for i in range(len(thickness)):
        for j in range(len(thickness[i])):
            map = SNRMapGenerator(path_snr=path_snr, path_T=path_T, path_fin=path_fin, d=thickness[i][j])
            map()


def write_data_to_DB():
    path_to_data = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\SNR_evaluation_v6'
    for file in os.listdir(path_to_data):
        if file.endswith('.csv') or file.endswith('.CSV'):
            working_file = os.path.join(path_to_data, file)
            d = int(file.split('_mm')[0])
            db = DB(d)
            with open(working_file) as f:
                content = f.readlines()
                content = [x.strip() for x in content]
                for _c in range(len(content)):
                    line = content[_c]
                    kV = float(line.split(',')[0])
                    T = float(line.split(',')[1])
                    SNR = float(line.split(',')[2])
                    db.add_data(voltage=kV, T=T, SNR=SNR)


def main():
    path_to_raw_data = r'\\132.187.193.8\junk\sgrischagin\2021-08-09-Sergej_SNR_Stufelkeil_40-75kV'
    path_to_result_prep = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\SNR_evaluation_v6'
    #prep_data(path_to_raw_data, path_to_result_prep)


    path_snr_data = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\SNR_evaluation_v6\2021-8-20_SNR'
    path_T_data = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\SNR_evaluation_v6\2021-8-20_T'
    path_fin_of_T_SNR = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\SNR_evaluation_v6'
    #calc_curves(path_snr_data, path_T_data, path_fin_of_T_SNR)


    #write_data_to_DB(path_fin_of_T_SNR)
    ds = [4, 5, 6, 12, 13, 14]
    SNR_Calculation.curve_db.create_MAP(os.path.join(path_fin_of_T_SNR, 'curves'), ds)



if __name__ == '__main__':
    main()
