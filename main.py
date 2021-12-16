from snr_calc.preperator import *
from Activator import Activator


class ParamHolder:
    def __init__(self):
        self.paths = {
        'snr_data': r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-11-29-sergej-AluKeil-5W\20211204_5W\2021-12-3_SNR',
        'T_data': r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-11-29-sergej-AluKeil-5W\20211204_5W\2021-12-3_T',

        'fCT_base_path': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\fCT-90kV-50proj',
        'fCT_imgs': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\fCT-90kV-50proj\imgs',
        'fCT_darks': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\fCT-90kV-50proj\darks',
        'fCT_refs': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\fCT-90kV-50proj\refs',

        'CT_base_path': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\CT_102kV-optkv',
        'CT_imgs': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\CT_102kV-optkv\imgs',
        'CT_darks': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\CT_102kV-optkv\darks',
        'CT_refs': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\CT_102kV-optkv\refs'}

        self.U0 = 90
        self.base_texp = 100
        self.snr_user = 10.0
        self.spatial_size = (25)
        self.ex_kv = [140]
        self.ex_ds = [32]

        self.vir_curve_step = 0.1



def prep_data(base_path):
    # adjusted (without the outliner far right bottom)
    crop = (500, 1500), (595, 1395)
    M = 15.8429
    watt = 5
    ex_kVs = []
    ex_ds = [32]

    _str = base_path.split('sgrischagin\\')[1]
    path_to_result = os.path.join(r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR', _str, '20211206_5W')

    img_loader = ImageLoader(used_SCAP=True, remove_lines=True, load_px_map=False, crop_area=crop)
    calc_snr = SNRPrepperator(path=base_path, magnification=M, path_result=path_to_result, watt=watt, ex_kvs=ex_kVs,
                              ex_ds=ex_ds, image_holder=img_loader)
    calc_snr()





def main():
    # =============== RAW DATA PREPERATION ===============
    #   1) calculates SNR spectra from sets of projections, refs and darks
    #   2) saves the calculated spectra as a txt file + final plot in the same directory
    #   :param base_path: must be the directory of the kV folders

    #prep_data(base_path=r'\\132.187.193.8\junk\sgrischagin\2021-11-29-sergej-AluKeil-5W')



    path_dict = {
        'snr_data': r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-11-29-sergej-AluKeil-5W\20211204_5W\2021-12-3_SNR',
        'T_data': r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-11-29-sergej-AluKeil-5W\20211204_5W\2021-12-3_T',

        'fCT_base_path': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\fCT-90kV-50proj',
        'fCT_imgs': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\fCT-90kV-50proj\imgs',
        'fCT_darks': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\fCT-90kV-50proj\darks',
        'fCT_refs': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\fCT-90kV-50proj\refs',

        'CT_base_path': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\CT_102kV-optkv',
        'CT_imgs': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\CT_102kV-optkv\imgs',
        'CT_darks': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\CT_102kV-optkv\darks',
        'CT_refs': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\CT_102kV-optkv\refs'}

    params = ParamHolder()
    acti = Activator(attributes=params)
    #acti = Activator(paths=path_dict, base_texp=100, U0=90, snr_user=10.0, spatial_size=(25), kv_ex=[140], ds_ex=[32])

    acti(create_plot=False, detailed=False, just_fCT=True)


if __name__ == '__main__':
    main()


