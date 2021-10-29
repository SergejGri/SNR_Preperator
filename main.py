import numpy as np
from PIL import Image
import Activator as act
from SNR_Calculation.Calculator import *
from SNR_Calculation.Prepper import *
from SNR_Calculation.CurveDB import *
import Plotter as PLT
from Plots import *


def prep_data(path_base, path_result_prep):
    image_shape = (1536, 1944)
    header = 2048
    M = 18.3833
    watt = 6
    stk_rng = (30, 100)
    excl = ['40kV', '60kV', '80kV', '120kV', '140kV', '160kV', '180kV'] # must be a string. Example: '50kV' for excluding 50 kV folder
    subfolders = ['0', '1', '2', '3', '4', '6']

    calc = SNRCalculator(img_shape=image_shape, header=header, path=path_base, magnification=M, nbins=None,
                         path_result=path_result_prep, watt=watt, exclude=excl, overwrite=False, mode_T=True,
                         stack_range=stk_rng)

    dirs = SNRCalculator.get_dirs(path_base, excl)

    for dirr in dirs:
            calc(dir=dirr, subf=subfolders)


def calc_curves(path_snr, path_T, path_fin, res_range):
    # TODO: SNRMapGenerator need to be a single call object which returns an object

    map = SNRMapGenerator(path_snr=path_snr, path_T=path_T, path_fin=path_fin, d=ds, port=res_range)
    return map()



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
        img = (data[i] - darks) / (refs - darks)
        T = img[np.where(img > 0)].min()
        list_Ts.append(T)
        list_angles.append(i)
    a = np.asarray(list_Ts)
    b = np.asarray(list_angles)
    del img, data, refs, darks
    gc.collect()
    return np.vstack([a, b])


if __name__ == '__main__':
    ds = [0, 2, 4, 6, 8]

    path_to_raw_data = [r"\\132.187.193.8\junk\sgrischagin\2021-09-30_Sergej_SNR-Stufenkeil_6W_130proj_0-1-2-3-4mm"
                        ]
    # =============== RAW DATA PREPERATION ===============
    #for _path in path_to_raw_data:
    #    _str = _path.split('sgrischagin\\')[1]
    #    path_to_result = os.path.join(r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR', _str, 'with_px_map')
    #    prep_data(_path, path_to_result)







    # =============== CREATE INITIAL MAP ===============
    snr_data = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-10-01_Sergej_SNR-Stufenkeil_6W_130proj_0-10mm\2021-10-6_SNR'
    T_data = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-10-01_Sergej_SNR-Stufenkeil_6W_130proj_0-10mm\2021-10-6_T'
    result = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-10-01_Sergej_SNR-Stufenkeil_6W_130proj_0-10mm\MAP'

    usr_size = 10  # [micro meter] here you need to define the min. sizes which should be resolvable at given SNR level
    U0 = 135

    data_obj = SNRMapGenerator(path_snr=snr_data, path_T=T_data, path_fin=result, d=ds)

    init_MAP = data_obj.create_MAP(spatial_range=(150, 250))
    #secondary_MAP = data_obj.create_MAP(spatial_range=usr_size)
    #tertiary_MAP = data_obj.create_MAP(spatial_range=(usr_size + 10))

    #PLT.Plotter().create_MAP_plot(path_result=result, object=init_MAP)
    #PLT.Plotter().create_MAP_plot(path_result=result, object=secondary_MAP)
    #PLT.Plotter().create_MAP_plot(path_result=result, object=tertiary_MAP)

    PLT.Plotter().create_evolution_plot(path_snr=snr_data, path_T=T_data, path_result=result, object=data_obj, d=4)
    print('test')






    #object = act.Activator(data_T=T_arr, path_db=path_result, U0=U0, ds=ds)


    #if not object.stop_exe:
    #    PLT.Plotter().create_plot(path_result=path_result, act_object=object, ds=ds, Y_style='log')








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




