from SNR_Calculation.preperator import *
from SNR_Calculation.map_db import *
from Activator import Activator


def prep_data(base_path):
    image_shape = (1944, 1536)
    header = 0
    M = 19.1262
    watt = 6

    crop = (500, 1500), (900, 1147)

    _str = base_path.split('sgrischagin\\')[1]
    path_to_result = os.path.join(r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR', _str, '20211126')

    calc_snr = SNRPrepperator(path=base_path, img_shape=image_shape, header=header, magnification=M, crop_area=crop,
                              path_result=path_to_result, watt=watt)
    calc_snr()





def main():
    # =============== RAW DATA PREPERATION ===============
    #   1) calculates SNR spectra from sets of projections, refs and darks
    #   2) saves the calculated spectra as a txt file + final plot in the same directory
    #   :param base_path: must be the directory of the kV folders

    prep_data(base_path=r'\\132.187.193.8\junk\sgrischagin\2021-11-26-sergej_Alukeil_6W')




    #snr_data = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-11-23-sergej_Al-StepWedge_6W\20211126\2021-11-26_SNR'
    #T_data = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-11-23-sergej_Al-StepWedge_6W\20211126\2021-11-26_T'

    # TODO: implement excluding kvs as kv_ex
    #acti = Activator(snr_files=snr_data, T_files=T_data, U0=100, snr_user=1.0, ssize=(150, 250), kv_ex=None)
    #acti(create_plot=True, detailed=False)


if __name__ == '__main__':
    main()


