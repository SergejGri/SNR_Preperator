import os
from ct_operations import CT_multi_img
from ext import file
from ext.SNR_spectra import SNR_Evaluator
from ext.SNR_spectra import ImageSeriesPixelArtifactFilterer



def single():
    folders = ['0000', '0004', '0008', '0012', '0016']
    kV = '102kV'
    watt = 5
    _dtype = '<u2'
    t_exp = 0.1

    path_to_raw_data = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\TEST_102kV_CT_SNR_AVG_compare\single_proj"
    _str = path_to_raw_data.split('Auswertung_MA\\')[1]
    path_to_result = os.path.join(r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA', _str,
                                  'EVALUATION_single_img_per_angle_22122021')
    os.makedirs(path_to_result, exist_ok=True)

    img_shape = (1536, 1944)
    header = 2048
    px_size = 74.8
    M = 4.45885
    px_size = px_size / M
    units = '$\mu m$'
    # crop = (500, 1500), (900, 1147)
    # view = (0, 0), *crop
    view = None

    darks = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\TEST_102kV_CT_SNR_AVG_compare\darks'
    refs = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\TEST_102kV_CT_SNR_AVG_compare\refs'

    ref_imgs = file.volume.Reader(refs, mode='raw', shape=img_shape, header=header, dtype=_dtype).load_all()
    dark_imgs = file.volume.Reader(darks, mode='raw', shape=img_shape, header=header, dtype=_dtype).load_all()

    SNR_eval = SNR_Evaluator()
    filterer = ImageSeriesPixelArtifactFilterer()

    results = []
    figure = None
    for i in range(len(folders)):
        imgs = os.path.join(path_to_raw_data, folders[i])

        print(f'using data as imgs: {imgs}\n')
        print(f'using data as refs: {refs}\n')
        print(f'using data as darks: {darks}\n')

        data_imgs = file.volume.Reader(imgs, mode='raw', shape=img_shape, header=header, dtype=_dtype).load_all()

        SNR_eval.estimate_SNR(data_imgs, ref_imgs, dark_imgs,
                              exposure_time=t_exp,
                              pixelsize=px_size,
                              pixelsize_units=units,
                              series_filterer=filterer,
                              u_nbins='auto',
                              save_path=os.path.join(path_to_result, f'SNR_{kV}_{folders[i]}_mm_{int(t_exp * 1000)}ms'))
        figure = SNR_eval.plot(figure, label=f'{folders[i]}', only_snr=True)
        results.append(SNR_eval)

    SNR_eval.finalize_figure(figure, title=f'SNR mm @ {kV} & {watt}W', save_path=os.path.join(path_to_result, f'{kV}'))


def avg():
    folders = ['0000', '0004', '0008', '0012', '0016']
    kV = '102kV'
    watt = 5
    _dtype = '<u2'
    t_exp = 0.1


    dirr = CT_multi_img(path_ct=r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\TEST_102kV_CT_SNR_AVG_compare\four_proj',
                        path_refs=r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\TEST_102kV_CT_SNR_AVG_compare\refs',
                        path_darks=r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\TEST_102kV_CT_SNR_AVG_compare\darks',
                        )

    path_to_raw_data = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\TEST_102kV_CT_SNR_AVG_compare\single_proj"
    _str = path_to_raw_data.split('Auswertung_MA\\')[1]
    path_to_result = os.path.join(r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA', _str,
                                  'EVALUATION_single_img_per_angle_22122021')
    os.makedirs(path_to_result, exist_ok=True)

    img_shape = (1536, 1944)
    header = 2048
    px_size = 74.8
    M = 4.45885
    px_size = px_size / M
    units = '$\mu m$'
    # crop = (500, 1500), (900, 1147)
    # view = (0, 0), *crop
    view = None

    darks = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\TEST_102kV_CT_SNR_AVG_compare\darks'
    refs = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\TEST_102kV_CT_SNR_AVG_compare\refs'

    ref_imgs = file.volume.Reader(refs, mode='raw', shape=img_shape, header=header, dtype=_dtype).load_all()
    dark_imgs = file.volume.Reader(darks, mode='raw', shape=img_shape, header=header, dtype=_dtype).load_all()

    SNR_eval = SNR_Evaluator()
    filterer = ImageSeriesPixelArtifactFilterer()

    results = []
    figure = None
    for i in range(len(folders)):
        imgs = os.path.join(path_to_raw_data, folders[i])

        print(f'using data as imgs: {imgs}\n')
        print(f'using data as refs: {refs}\n')
        print(f'using data as darks: {darks}\n')

        data_imgs = file.volume.Reader(imgs, mode='raw', shape=img_shape, header=header, dtype=_dtype).load_all()

        SNR_eval.estimate_SNR(data_imgs, ref_imgs, dark_imgs,
                              exposure_time=t_exp,
                              pixelsize=px_size,
                              pixelsize_units=units,
                              series_filterer=filterer,
                              u_nbins='auto',
                              save_path=os.path.join(path_to_result, f'SNR_{kV}_{folders[i]}_mm_{int(t_exp * 1000)}ms'))
        figure = SNR_eval.plot(figure, label=f'{folders[i]}', only_snr=True)
        results.append(SNR_eval)

    SNR_eval.finalize_figure(figure, title=f'SNR mm @ {kV} & {watt}W', save_path=os.path.join(path_to_result, f'{kV}'))


def main():
    single()
    avg()


if __name__ == '__main__':
    main()





