import os
from externe_files.SNR_spectra import SNR_Evaluator, ImageSeriesPixelArtifactFilterer
from SNR_Calculation.preperator import load_px_map
from SNR_Calculation import preperator
from SNR_Calculation.preperator import SNRCalculator
from externe_files import file, common, image




if __name__ == '__main__':
    ds = ['2', '4', '6', '8']
    kV = '80kV'
    watt = 6

    path_to_raw_data = r"\\132.187.193.8\junk\sgrischagin\2021-10-01_Sergej_SNR-Stufenkeil_6W_130proj_0-10mm"
    _str = path_to_raw_data.split('sgrischagin\\')[1]
    path_to_result = os.path.join(r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR', _str, 'SINGLE_EVAL_20211027_with_px_map', kV)

    img_shape = (1536, 1944)
    header = 2048
    px_size = 74.8
    M = 18.3833
    px_size = px_size / M
    units = '$\mu m$'
    stack_range = (30, 100)

    path_px_map = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\BAD-PIXEL-bin1x1-scans-MetRIC.tif"
    refs = r'\\132.187.193.8\junk\sgrischagin\2021-10-01_Sergej_SNR-Stufenkeil_6W_130proj_0-10mm\80kV\refs'
    darks = r'\\132.187.193.8\junk\sgrischagin\2021-10-01_Sergej_SNR-Stufenkeil_6W_130proj_0-10mm\80kV\darks'

    refs = file.volume.Reader(refs, mode='raw', shape=img_shape, header=header, dtype='u2').load_range((stack_range[0], stack_range[-1]))
    darks = file.volume.Reader(darks, mode='raw', shape=img_shape, header=header, dtype='u2').load_range((stack_range[0], stack_range[-1]))

    _slice = slice(0, -1), slice(60, 1460), slice(125, 1825)
    px_map_slice = _slice[1], _slice[2]

    SNR_eval = SNR_Evaluator()
    px_map = load_px_map()
    filterer = ImageSeriesPixelArtifactFilterer(bad_pixel_map=px_map[px_map_slice])

    results = []
    figure = None
    for k in range(len(ds)):
        imgs = os.path.join(path_to_raw_data, kV, ds[k])
        t_exp = preperator.get_t_exp(imgs)
        data = file.volume.Reader(imgs, mode='raw', shape=img_shape, header=header, dtype='u2').load_range((stack_range[0], stack_range[-1]))
        SNR_eval.estimate_SNR(data[_slice], refs[_slice], darks[_slice],
                              exposure_time=t_exp,
                              pixelsize=px_size,
                              pixelsize_units=units,
                              series_filterer=filterer,
                              save_path=os.path.join(path_to_result,
                                                     f'SNR_{kV}_{ds[k]}_mm_{t_exp}ms'))
        figure = SNR_eval.plot(figure, label=f'{ds[k]} mm', only_snr=True)
        results.append(SNR_eval)

    SNR_eval.finalize_figure(figure,
                             title=f'SNR Coin 2mm @ {kV} & {watt}W',
                             save_path=os.path.join(path_to_result, f'{kV}'))