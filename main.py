import numpy as np
import timeit
from scipy import interpolate

import Activator as act
from SNR_Calculation.Prepper import *
from SNR_Calculation.SNRMapGenerator import *
from SNR_Calculation.CurveDB import *
import Plotter as PLT


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
    thickness = [0, 1, 2, 4, 5, 8, 9]
    for i in range(len(thickness)):
        map = SNRMapGenerator(path_snr=path_snr, path_T=path_T, path_fin=path_fin, d=thickness[i])
        map()


def write_data_to_DB(path):
    for file in os.listdir(path):
        if file.endswith('.csv') or file.endswith('.CSV'):
            working_file = os.path.join(path, file)
            d = int(file.split('_mm')[0])
            db = DB(path_DB=path)
            with open(working_file) as f:
                content = f.readlines()
                content = [x.strip() for x in content]
                for _c in range(len(content)):
                    line = content[_c]
                    kV = float(line.split(',')[0])
                    T = float(line.split(',')[1])
                    SNR = float(line.split(',')[2])
                    db.add_data(d, voltage=kV, T=T, SNR=SNR)


def compare_plot(db1, db2):
    pass


if __name__ == '__main__':
    start = timeit.default_timer()
    # path_to_raw_data = r'\\132.187.193.8\junk\sgrischagin\2021-08-09-Sergej_SNR_Stufelkeil_40-75kV'
    # path_to_result_prep = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\SNR_evaluation_v6'
    # prep_data(path_to_raw_data, path_to_result_prep)

    #path_snr_data = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\2021-8-30_Evaluation\SNR'
    #path_T_data = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\2021-8-30_Evaluation\Transmission'
    path_result = r'C:\Users\Rechenfuchs\PycharmProjects\SNR_Preperator'
    #calc_curves(path_snr_data, path_T_data, path_result)

    # write_data_to_DB(path=path_result)
    ds = [1, 4, 5, 8, 9]
    # create_MAP(path_result, ds, mode_fit=True)

    # obj = act.Activator(list_d=ds, path_db=path_result)

    test_arr = np.array([[0, 11, 34, 56, 75, 80, 99, 131, 165, 178],
                         [0.26, 0.35, 0.25, 0.27, 0.26, 0.31, 0.37, 0.52, 0.41, 0.45]])

    U0 = 100

    object = act.Activator(data_T=test_arr, path_db=path_result, U0=U0, ds=ds)
    stop = timeit.default_timer()
    print(f'Execution time: {round((stop - start), 2)}s')
    if not object.stop_exe:
        PLT.Plotter().create_plot(path_result=path_result, act_object=object, ds=ds, Y_style='log')

    #PLT.Plotter().compare_plot()




