import numpy as np
from PIL import Image
from Activator import Activator
import Activator as act
from SNR_Calculation.preperator import *
from SNR_Calculation.map_generator import *
from SNR_Calculation.map_db import *
import Plotter as PLT
from Plots import *


def prep_data(path_base, path_result_prep, ds):
    image_shape = (1536, 1944)
    header = 2048
    M = 18.3833
    watt = 6
    stk_rng = (30, 100)
    excl = ['40kV', '60kV', '80kV', '120kV', '140kV', '160kV', '180kV'] # must be a string. Example: '50kV' for excluding 50 kV folder

    calc = SNRCalculator(img_shape=image_shape, header=header, path=path_base, magnification=M, nbins=None,
                         path_result=path_result_prep, watt=watt, exclude=excl, overwrite=False, mode_T=True,
                         stack_range=stk_rng)

    dirs = SNRCalculator.get_dirs(path_base, excl)

    for dirr in dirs:
        calc(dir=dirr, subf=ds)



def fast_CT():
    stack_range = (50, 100)
    img_shape = (1536, 1944)
    header = 2048

    imgs = r'\\132.187.193.8\junk\sgrischagin\2021-10-04_Sergej_Res-phantom_135kV_1mean_texp195_15proj-per-angle\test_fast_CT'
    darks = r'\\132.187.193.8\junk\sgrischagin\2021-10-04_Sergej_Res-phantom_135kV_1mean_texp195_15proj-per-angle\darks'
    refs = r'\\132.187.193.8\junk\sgrischagin\2021-10-04_Sergej_Res-phantom_135kV_1mean_texp195_15proj-per-angle\refs'

    darks = file.volume.Reader(darks, mode='raw', shape=img_shape, header=header, dtype='u2').load_range((stack_range[0], stack_range[-1]))
    refs = file.volume.Reader(refs, mode='raw', shape=img_shape, header=header, dtype='u2').load_range((stack_range[0], stack_range[-1]))

    list_Ts = []
    list_angles = []
    data = file.volume.Reader(imgs, mode='raw', shape=img_shape, header=header, dtype='u2').load_all()
    for i in range(len(data)):
        if darks is None and refs is None:
            T = data[np.where(data > 0)].min()
        else:
            img = (data[i] - darks) / (refs - darks)
            T = img[np.where(img > 0)].min()
        list_Ts.append(T)
        list_angles.append(i)
    a = np.asarray(list_Ts)
    b = np.asarray(list_angles)
    del img, data, refs, darks
    gc.collect()
    return np.vstack([a, b])



def main():
    path_to_db = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-10-01_Sergej_SNR-Stufenkeil_6W_130proj_0-10mm\MAP\curves.db"
    # =============== RAW DATA PREPERATION ===============
    # path_to_raw_data = [r'\\132.187.193.8\junk\sgrischagin\2021-10-01_Sergej_SNR-Stufenkeil_6W_130proj_0-10mm']
    # ds = [0, 2, 4, 6, 8]
    # for _path in path_to_raw_data:
    #    _str = _path.split('sgrischagin\\')[1]
    #    path_to_result = os.path.join(r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR', _str, 'NEW_EXP_T')
    #    prep_data(_path, path_to_result, ds=ds)

    # =============== CREATE MAP ===============
    snr_data = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-10-01_Sergej_SNR-Stufenkeil_6W_130proj_0-10mm\2021-10-6_SNR'
    T_data = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-10-01_Sergej_SNR-Stufenkeil_6W_130proj_0-10mm\2021-10-6_T'
    result = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-10-01_Sergej_SNR-Stufenkeil_6W_130proj_0-10mm\MAP'
    ds = [0, 2, 4, 6, 8]
    usr_size = 10  # [micro meter] here you need to define the min. sizes which should be resolvable at given SNR level

    MAP_object = SNRMapGenerator(path_snr=snr_data, path_T=T_data, path_fin=result, d=ds)

    init_MAP = MAP_object(spatial_range=(150, 250), init_MAP=True)
    print('test')

    #secondary_MAP = MAP_object(spatial_range=usr_size)

    #PLT.Plotter().create_MAP_plot(path_result=result, object=init_MAP)
    #PLT.Plotter().create_MAP_plot(path_result=result, object=secondary_MAP)



    # 1) positioniere Objekt auf dem Drehteller und starte fast_CT
    arr_T = fast_CT()
    # 2) der ausgespuckte Array muss in den Activator gepackt werden um U_best zu bekommen

    acti = Activator(data_T=arr_T, path_db=path_to_db, U0=67, ds=[2, 4, 6, 8])
    acti()

    print('test')









if __name__ == '__main__':
    main()
















'''
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
'''




