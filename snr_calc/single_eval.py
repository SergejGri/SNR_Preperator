import os
import helpers as h
from ext import file
from ext.SNR_spectra import SNR_Evaluator
from ext.SNR_spectra import ImageSeriesPixelArtifactFilterer


if __name__ == '__main__':

    h.Cropper(path=r'\\132.187.193.8\junk\sgrischagin\2021-11-29-sergej-AluKeil-5W\180kV\8\1512-31243-High-250,0ms-Binx11_4.raw')


    ds = [2]
    kV = '180kV'
    watt = 6
    _dtype = '<u2'
    t_exp = 0.15

    path_to_raw_data = r"\\132.187.193.8\junk\sgrischagin\2021-11-26-sergej_Alukeil_6W\new_kvs"
    _str = path_to_raw_data.split('sgrischagin\\')[1]
    path_to_result = os.path.join(r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\SNR', _str, 'SINGLE_EVAL_20211129', kV)

    img_shape = (1944, 1536)
    header = 0
    px_size = 74.8
    M = 15.8429
    px_size = px_size / M
    units = '$\mu m$'
    crop = (500, 1500), (900, 1147)
    view = (0, 0), *crop

    refs = os.path.join(path_to_raw_data, kV, 'refs')
    darks = os.path.join(path_to_raw_data, kV, 'darks')

    ref_imgs = file.volume.Reader(refs, mode='raw', shape=img_shape, header=header, dtype=_dtype, crops=view).load_all()
    dark_imgs = file.volume.Reader(darks, mode='raw', shape=img_shape, header=header, dtype=_dtype, crops=view).load_all()



    SNR_eval = SNR_Evaluator()
    #px_map = h.load_bad_pixel_map()
    #filterer = ImageSeriesPixelArtifactFilterer(bad_pixel_map=px_map)
    filterer = ImageSeriesPixelArtifactFilterer()

    results = []
    figure = None
    for k in range(len(ds)):
        imgs = os.path.join(path_to_raw_data, kV, ds[k])

        print(f'using data as imgs: {imgs}\n')
        print(f'using data as refs: {ref_imgs}\n')
        print(f'using data as darks: {dark_imgs}\n')

        data_imgs = file.volume.Reader(imgs, mode='raw', shape=img_shape, header=header, dtype=_dtype, crops=view).load_all()


        SNR_eval.estimate_SNR(data_imgs, ref_imgs, dark_imgs,
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




