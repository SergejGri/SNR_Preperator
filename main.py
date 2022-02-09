from snr_evaluator import StepWedgeEvaluator
from snr_evaluator import SNREvaluator
from Activator import Activator


class ParamHolder:
    def __init__(self):
        '''
        :param base_texp:   exposure time in s
        '''

        # if _base_texp value is passed, mode_avg will be activated automatically (see Activator.__init__)
        self.U0 = 90
        self.base_texp = 0.100
        self.USR_SNR = 1
        self.spatial_size = 58
        self.min_kv = 50
        self.max_kv = 180
        self.ds = [0, 2, 4, 8, 12, 16, 20, 24, 28]
        self.excluded_kvs = None
        self.excluded_thicknesses = None
        self.virtual_curve_step = 0.1
        self.CT_steps = 50
        self.fCT_steps = 50
        self.imgs_per_angle = 120
        self.snr_bins = 30

        self.paths = {
        'MAP_snr_files': r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\20220120_stepWedge\2022-1-20_SNR',
        'MAP_T_files': r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\20220120_stepWedge\2022-1-20_T',

        'fCT_base_path': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\fCT-90kV-50proj',
        'fCT_imgs': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\fCT-90kV-50proj\imgs',
        'fCT_darks': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\fCT-90kV-50proj\darks',
        'fCT_refs': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\fCT-90kV-50proj\refs',

        'CT_base_path': r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\CT_102kV-optkv',
        'CT_imgs': r'\\132.187.193.8\junk\sgrischagin\2021-12-22-sergej-CT-halbesPhantom-102kV-100ms-5W-M4p46\imgs',
        'CT_darks': r'\\132.187.193.8\junk\sgrischagin\2021-12-22-sergej-CT-halbesPhantom-102kV-100ms-5W-M4p46\darks',
        'CT_refs': r'\\132.187.193.8\junk\sgrischagin\2021-12-22-sergej-CT-halbesPhantom-102kV-100ms-5W-M4p46\refs',
        'CT_avg_imgs': r'\\132.187.193.8\junk\sgrischagin\2021-12-22-sergej-CT-halbesPhantom-102kV-100ms-5W-M4p46\merged-CT\imgs',
        'CT_avg_refs': r'\\132.187.193.8\junk\sgrischagin\2021-12-22-sergej-CT-halbesPhantom-102kV-100ms-5W-M4p46\merged-CT\refs',
        'CT_avg_darks': r'\\132.187.193.8\junk\sgrischagin\2021-12-22-sergej-CT-halbesPhantom-102kV-100ms-5W-M4p46\merged-CT\darks',
        'result_path': rf'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\3D_SNR_eval_24012022_view_only_Cylinder_v1_ss{self.spatial_size}_without_bad_px'
        }

    def printer(self, object):
        for property, value in vars(object).items():
            print(property, ":", value)



def main():
    #snr_evaluator = SNREvaluator(watt=5, magnification=15.8429, voltage=102, btexp=1, only_snr=False, ex_kvs=[140],
    #                             ex_ds=[32], detector_pixel=74.8)
    #snr_evaluator.evaluate_step_wedge(base_path=r'\\132.187.193.8\junk\sgrischagin\2021-11-29-sergej-AluKeil-5W',
    #                                result_path=r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\20220120_stepWedge')


    acti = Activator(attributes=ParamHolder())
    acti(create_plot=True, detailed=False)



if __name__ == '__main__':
    main()
