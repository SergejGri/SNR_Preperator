from SNR_Calculation.Prepper import *
from SNR_Calculation.SNRMapGenerator import *




def SNR_eval(slice):

    d = 4
    dirs = ['40kV']
    dfs = SNRCalculator.get_df()
    areas = SNRCalculator.get_areas()

    img_shape = (1536, 1944)

    header = 2048
    t_exp = 1800
    watt = 4
    pixel_size = 74.8
    pixel_size_units = '$\mu m$'
    rng = (29, 99)
    dir = '40kV'
    x_min = None
    str_voltage = dir
    if '_' in str_voltage:
        voltage = int(str_voltage.split('_')[0])
    else:
        voltage = int(str_voltage.split('kV')[0])
    base_path = r'\\132.187.193.8\junk\sgrischagin\2021-08-09-Sergej_SNR_Stufelkeil_40-75kV'
    res_path_snr = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\SNR_evaluation_v3'

    results = []
    figure = None
    SNR_eval = SNR_Evaluator()
    filterer_l = ImageSeriesPixelArtifactFilterer()

    for dir in dirs:
        for f in dfs:
            for area in areas:
                list_l, list_r = SNRCalculator. current_d(f)
                for l, r in zip(list_l, list_r):
                imgs, ref_imgs, dark_imgs = SNRCalculator.prepare_imgs(base_path, dir, df, area)
                data = file.volume.Reader(imgs, mode='raw', shape=img_shape, header=header).load_range(rng)
                refs = file.volume.Reader(ref_imgs, mode='raw', shape=img_shape, header=header).load_range(rng)
                darks = file.volume.Reader(dark_imgs, mode='raw', shape=img_shape, header=header).load_range(rng)

                SNR_eval.estimate_SNR(data[slice], refs[slice], darks[slice],
                                      exposure_time=t_exp,
                                      pixelsize=pixel_size,
                                      pixelsize_units=pixel_size_units,
                                      series_filterer=filterer_l,
                                      save_path=os.path.join(res_path_snr,
                                                             f'SNR_{voltage}kV_{d}_mm_{t_exp}ms'))
                figure = SNR_eval.plot(figure, f'{d} mm')
                results.append(SNR_eval)

                SNR_eval.finalize_figure(figure,
                                         title=f'SNR @ {voltage}kV & {watt}W',
                                         smallest_size=x_min,
                                         save_path=os.path.join(res_path_snr, f'{voltage}kV_{d}mm'))



def main():
    # left area
    view = slice(0, 100), slice(60, 1450), slice(105, 855)
    SNR_eval(view)

    # right area
    #SNR_eval((slice(0, 100), slice(60, 1450), slice(1080, 1830)))


if __name__ == '__main__':
    main()