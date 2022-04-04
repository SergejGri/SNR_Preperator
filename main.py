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
        self.base_texp = 0.05
        self.USR_SNR = 350
        self.new_mergings = True
        self.spatial_size = 150
        self.min_kv = 50
        self.max_kv = 180
        self.ds = [0, 2, 4, 8, 12, 16, 20, 24, 28]
        self.excluded_kvs = [180]
        self.excluded_thicknesses = None
        self.virtual_curve_step = 0.1
        self.CT_steps = 50
        self.fCT_steps = 50
        self.imgs_per_angle = 480
        self.realCT_steps = 1500
        self.snr_bins = 30

        self.paths = {
            # ACHTUNG NEUER STEP WEDGE EINGESPEIST
        'MAP_snr_files': r'C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\20220223_stepWedge\2022-2-23_SNR',
        'MAP_T_files': r'C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\20220223_stepWedge\2022-2-23_T',
        'plots': r'C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\plots',

        #'fCT_base_path': r'\\132.187.193.8\junk\sgrischagin\fCT-1W',
        'fCT_imgs': r'C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\Messreihe_P-5W_U-50kV_proj-480_theta-50\fCT-90kV_v2\imgs',
        'fCT_darks': r'C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\Messreihe_P-5W_U-50kV_proj-480_theta-50\fCT-90kV_v2\darks',
        'fCT_refs': r'C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\Messreihe_P-5W_U-50kV_proj-480_theta-50\fCT-90kV_v2\refs',

        #'CT_base': r'C:\Users\Sergej Grischagin\Desktop\final_evaluations\night_version\gated-CT-50kV-480proj-50angles',
        'CT_imgs': r'C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\Messreihe_P-5W_U-50kV_proj-480_theta-50\gated-CT-50kV-480proj-50angles\imgs',
        'CT_darks': r'C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\Messreihe_P-5W_U-50kV_proj-480_theta-50\gated-CT-50kV-480proj-50angles\darks',
        'CT_refs': r'C:\Users\Sergej Grischagin\Desktop\most_final_evaluations\Messreihe_P-5W_U-50kV_proj-480_theta-50\gated-CT-50kV-480proj-50angles\refs',
        #'CT_avg_refs': r'\\132.187.193.8\junk\sgrischagin\2021-12-22-sergej-CT-halbesPhantom-102kV-100ms-5W-M4p46\merged-CT-biggerview\imgs\refs',
        #'CT_avg_darks': r'\\132.187.193.8\junk\sgrischagin\2021-12-22-sergej-CT-halbesPhantom-102kV-100ms-5W-M4p46\merged-CT-biggerview\imgs\darks',
        'merged_imgs': rf'C:\Users\Sergej Grischagin\Desktop\final_evaluations\3D_SNR_eval_18032022_biggerview_nocylinder_v1_ss-{self.spatial_size}_SNRB-{self.USR_SNR}\merged_imgs',
        'result_path': rf'C:\Users\Sergej Grischagin\Desktop\final_evaluations\3D_SNR_eval_18032022_biggerview_nocylinder_v1_ss-{self.spatial_size}_SNRB-{self.USR_SNR}'
        }

    def printer(self, object):
        for property, value in vars(object).items():
            print(property, ":", value)



def main():
    #snr_evaluator = SNREvaluator(watt=5, magnification=15.8429, voltage=102, only_snr=False, ex_kvs=[140],
    #                             ex_ds=[32], detector_pixel=74.8)
    #snr_evaluator.evaluate_step_wedge(base_path=r'\\132.187.193.8\junk\sgrischagin\2021-11-29-sergej-AluKeil-5W',
    #                                result_path=r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\20220223_stepWedge')



    acti = Activator(attributes=ParamHolder())
    acti(create_plot=True, detailed=False)



if __name__ == '__main__':
    main()
