from snr_evaluator import *
from Activator import Activator
from image_loader import ImageLoader


class ParamHolder:
    def __init__(self):
        '''
        :param base_texp:   exposure time in s
        '''

        self.paths = {
        'snr_data': r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-11-29-sergej-AluKeil-5W\20211204_5W\2021-12-3_SNR',
        'T_data': r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-11-29-sergej-AluKeil-5W\20211204_5W\2021-12-3_T',

        'fCT_base_path': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\fCT-90kV-50proj',
        'fCT_imgs': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\fCT-90kV-50proj\imgs',
        'fCT_darks': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\fCT-90kV-50proj\darks',
        'fCT_refs': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\fCT-90kV-50proj\refs',

        'CT_base_path': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\CT_102kV-optkv',
        'CT_imgs': r'\\132.187.193.8\junk\sgrischagin\2021-12-22-sergej-CT-halbesPhantom-102kV-100ms-5W-M4p46\imgs',
        'CT_darks': r'\\132.187.193.8\junk\sgrischagin\2021-12-22-sergej-CT-halbesPhantom-102kV-100ms-5W-M4p46\darks',
        'CT_refs': r'\\132.187.193.8\junk\sgrischagin\2021-12-22-sergej-CT-halbesPhantom-102kV-100ms-5W-M4p46\refs',
        'CT_imgs_avg': r''
        }
        # parameter with prefixed _ are mandatory
        # if _base_texp value is passed, mode_avg will be activated automatically (see Activator.__init__)

        self.U0 = 90
        self.base_texp = 0.100
        self.snr_user = 0.05
        self.spatial_size = (50)
        self.excluded_kvs = [140]
        self.excluded_thicknesses = [32]
        self.virtual_curve_step = 0.1
        self.CT_steps = 1500
        self.imgs_per_angle = 120
        self.snr_bins = 30                  # must be a

    def printer(self, object):
        for property, value in vars(object).items():
            print(property, ":", value)


def main():
    # =============== RAW DATA PREPERATION ===============
    #   1) calculates SNR spectra from sets of projections, refs and darks
    #   2) saves the calculated spectra as a txt file + final plot in the same directory
    #   :param base_path: must be the directory of the kV folders

    #prep_data(base_path=r'\\132.187.193.8\junk\sgrischagin\2021-11-29-sergej-AluKeil-5W')

    params = ParamHolder()
    loader = ImageLoader(used_SCAP=False, remove_lines=False, load_px_map=False)
    #scanner = Scanner(p_snr=params.paths['snr_data'], p_T=params.paths['T_data'], d_ex=[32])
    map_generator = SNRMapGenerator(p_snr=r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-11-29-sergej-AluKeil-5W\20211204_5W\2021-12-3_SNR',
                                    p_T=r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR\2021-11-29-sergej-AluKeil-5W\20211204_5W\2021-12-3_T',
                                    ds=[0, 2, 4, 8, 12, 16, 20, 24, 28],
                                    kv_filter=[140])

    acti = Activator(image_loader=loader, map_generator=map_generator, attributes=params)
    acti(create_plot=True, only_fCT=False, detailed=False)



def calc_3D_snr():

    base_path = r'\\132.187.193.8\junk\sgrischagin\2021-12-22-sergej-CT-halbesPhantom-102kV-100ms-5W-M4p46'
    _str = base_path.split('sgrischagin\\')[1]
    path_to_result = os.path.join(r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR', _str, '20211222_5W')

    img_loader = ImageLoader(used_SCAP=False, remove_lines=False, load_px_map=False)
    eva = SNREvaluator(image_loader=img_loader, watt=5.0, voltage=102, magnification=4.45914, btexp=0.1, only_snr=False)



def prep_data(base_path):
    # adjusted (without the fat outliner far right bottom)
    crop = (500, 1500), (595, 1395)
    M = 15.8429
    watt = 5
    ex_kVs = []
    ex_ds = [32]

    _str = base_path.split('sgrischagin\\')[1]
    path_to_result = os.path.join(r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR', _str, '20211206_5W')

    img_loader = ImageLoader(used_SCAP=True, remove_lines=True, load_px_map=False, crop_area=crop)
    calc_snr = StepWedgeEvaluator(path=base_path, magnification=M, path_result=path_to_result, watt=watt, ex_kvs=ex_kVs,
                                  ex_ds=ex_ds, image_holder=img_loader)
    calc_snr()


if __name__ == '__main__':
    main()
