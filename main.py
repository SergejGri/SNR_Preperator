from snr_calc.preperator import *
from snr_calc.map_db import *
from Activator import Activator


def prep_data(base_path):
    # adjusted (without the outliner far right bottom)
    crop = (500, 1500), (595, 1395)
    M = 15.8429
    watt = 5
    ex_kVs = []
    ex_ds = [32]

    _str = base_path.split('sgrischagin\\')[1]
    path_to_result = os.path.join(r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR', _str, '20211206_5W')

    img_loader = ImageHolder(used_SCAP=True, remove_lines=True, load_px_map=True, crop_area=crop)
    calc_snr = SNRPrepperator(path=base_path, magnification=M, path_result=path_to_result, watt=watt, ex_kvs=ex_kVs,
                              ex_ds=ex_ds, image_holder=img_loader)
    calc_snr()





def main():
    # =============== RAW DATA PREPERATION ===============
    #   1) calculates SNR spectra from sets of projections, refs and darks
    #   2) saves the calculated spectra as a txt file + final plot in the same directory
    #   :param base_path: must be the directory of the kV folders

    #prep_data(base_path=r'\\132.187.193.8\junk\sgrischagin\2021-11-29-sergej-AluKeil-5W')



    snr_data = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-11-29-sergej-AluKeil-5W\20211204_5W\2021-12-3_SNR'
    T_data = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-11-29-sergej-AluKeil-5W\20211204_5W\2021-12-3_T'

    # TODO: implement excluding kvs as kv_ex
    acti = Activator(snr_files=snr_data, T_files=T_data, U0=110, snr_user=1.0, ssize=(30), kv_ex=[140], ds_ex=[32])
    acti(create_plot=True, detailed=False)


if __name__ == '__main__':
    main()


