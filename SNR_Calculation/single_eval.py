import os
import helpers as h
from externe_files import file
from externe_files.SNR_spectra import SNR_Evaluator
from externe_files.SNR_spectra import ImageSeriesPixelArtifactFilterer
from SNR_Calculation import preperator





if __name__ == '__main__':
    ds = ['22']
    kV = '70kV'
    watt = 6
    #_dtype = 'H'
    _dtype = '<u2'

    path_to_raw_data = r"\\132.187.193.8\junk\sgrischagin\2021-11-17-Sergej-StepWedge_6W"
    _str = path_to_raw_data.split('sgrischagin\\')[1]
    path_to_result = os.path.join(r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR', _str, 'SINGLE_EVAL_20211121_smallest_area', kV)

    img_shape = (1944, 1536)
    header = 0
    px_size = 74.8
    M = 18.6241
    px_size = px_size / M
    units = '$\mu m$'

    refs = os.path.join(path_to_raw_data, kV, 'refs')
    darks = os.path.join(path_to_raw_data, kV, 'darks')

    ref_imgs = file.volume.Reader(refs, mode='raw', shape=img_shape, header=header, dtype=_dtype).load_all()
    dark_imgs = file.volume.Reader(darks, mode='raw', shape=img_shape, header=header, dtype=_dtype).load_all()

    _view = slice(500, 1500), slice(645, 890)
    #_view = slice(500, 1500), slice(600, 1500)
    view = slice(None, None), *_view

    SNR_eval = SNR_Evaluator()
    px_map = h.load_bad_pixel_map()
    filterer = ImageSeriesPixelArtifactFilterer(bad_pixel_map=px_map)

    results = []
    figure = None
    for k in range(len(ds)):
        imgs = os.path.join(path_to_raw_data, kV, ds[k])
        t_exp = preperator.get_texp(kv=kV)
        data_imgs = file.volume.Reader(imgs, mode='raw', shape=img_shape, header=header, dtype=_dtype).load_all()

        print('\n'
              'calculating SNR with data from sources: \n'
              f'path to imgs: {imgs} \n'
              f'path to refs: {refs} \n'
              f'path to darks: {darks} \n')

        SNR_eval.estimate_SNR(data_imgs[view], ref_imgs[view], dark_imgs[view],
                              exposure_time=t_exp,
                              pixelsize=px_size,
                              pixelsize_units=units,
                              series_filterer=filterer,
                              u_nbins='auto',
                              save_path=os.path.join(path_to_result,
                                                     f'SNR_{kV}_{ds[k]}_mm_{int(t_exp*1000)}ms'))
        figure = SNR_eval.plot(figure, label=f'{ds[k]} mm', only_snr=True)
        results.append(SNR_eval)

    SNR_eval.finalize_figure(figure, title=f'SNR mm @ {kV} & {watt}W', save_path=os.path.join(path_to_result, f'{kV}'))
