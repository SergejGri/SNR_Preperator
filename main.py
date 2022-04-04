from snr_evaluator import SNREvaluator
from Activator import Activator


class ParamHolder:
    def __init__(self):
        '''
        :param U0:                  voltage for fCT
        :param base_texp:           const. exposure time in s
        :param USR_SNR:             user desired SNR-Niveau
        :param new_mergings:        if "True", Images from path "CT_imgs" will be merged depending on calculated avgs in
                                    self.calc_texp() (see Activator)
        :param spatial_size:        area in which the avg of the snr-spectra is to be done
        :param min_kv:              min value of used valtage in the measurement
        :param max_kv:              see "min_kv"
        :param ds:                  only the thicknesses in this list are taken into account in the evaluation
        :param excluded_kvs:        excluded voltages in the evaluation
        :param excluded_ds:         excluded measured material thicknesses in the evaluation
        :param h_curve_step:        stepsize for virtual curves between actual data (curve number between 2mm and
                                    4mm curves)
        :param CT_steps:            number of angles in SNR_m calculation (see thesis chapter 5)
        :param fCT_steps:           see "CT_steps"
        :param imgs_per_angle:      raw and unmerged projections per CT_step
        :param realCT_steps:        number of projection angles of the "real" compare CT
        :param snr_bins:            number of projection in the stack for snr evaluation in SNR_m (see fig. 23 (Ebene 3)
                                    in thesis)
        :param paths:               required paths for evaluations
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
        self.excluded_ds = None
        self.h_curve_step = 0.1
        self.CT_steps = 50
        self.fCT_steps = 50
        self.imgs_per_angle = 480
        self.realCT_steps = 1500
        self.snr_bins = 30

        self.paths = {
        'MAP_snr_files': r'',
        'MAP_T_files': r'',
        'plots': r'',

        'fCT_imgs': r'',
        'fCT_darks': r'',
        'fCT_refs': r'',

        'CT_imgs': r'',
        'CT_darks': r'',
        'CT_refs': r'',
        'merged_imgs': rf'',
        'result_path': rf''
        }

    def printer(self, object):
        for property, value in vars(object).items():
            print(property, ":", value)



def main():
    snr_evaluator = SNREvaluator(watt=5, magnification=15.8429, voltage=102, only_snr=False, ex_kvs=[140],
                                 ex_ds=[32], detector_pixel=74.8)
    snr_evaluator.evaluate_step_wedge(base_path=r'',
                                    result_path=r'')



    acti = Activator(attributes=ParamHolder())
    acti(create_plot=True, detailed=False)



if __name__ == '__main__':
    main()
