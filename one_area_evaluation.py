import externe_files.SNR_spectra
from SNR_Calculation.Prepper import *
from SNR_Calculation.SNRMapGenerator import *
from SNR_Calculation.CurveDB import *
import datetime
import os


def prep_data(path_base, path_result_prep):
    image_shape = (1536, 1944)
    header = 2048
    M = 18.3768
    watt = 4
    excl = []

    calc = SNRCalculator(img_shape=image_shape, header=header, path=path_base, magnification=M, nbins=300,
                         path_result=path_result_prep, watt=watt, exclude=excl, overwrite=False)

    dirs = SNRCalculator.get_dirs(path_base, excl)
    filters = SNRCalculator.get_df()


    for dir in dirs:
        for fltr in filters:
            calc(dir=dir, df=fltr)


def _get_t_exp(path):
    t_exp = None
    for file in os.listdir(path):
        piece_l = file.split('expTime_')[1]
        piece_r = piece_l.split('__')[0]
        t_exp = int(piece_r)
        break
    return t_exp


def man_SNR_eval_single():
    c_date = datetime.datetime.now()
    _date = f'{c_date.year}-{c_date.month}-{c_date.day}_Evaluation'

    img_shape = (1536, 1944)
    header = 2048
    M = 18.3768
    watt = 6
    pixel_size = 4.07033
    pixel_size_units = '$\mu m$'
    nbins = 300
    path_map = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\BAD-PIXEL-bin1x1-scans-MetRIC.tif'

    dirs = ['140kV', '160kV', '180kV']
    thicknesses = ['2']
    for dir in dirs:
        figure = None
        for d in thicknesses:
            path_to_data = fr'\\132.187.193.8\junk\sgrischagin\2021-08-26-Sergej_SNR-Stufenkeil_130proj_6W\{dir}\{d}'
            path_to_refs = fr'\\132.187.193.8\junk\sgrischagin\2021-08-26-Sergej_SNR-Stufenkeil_130proj_6W\{dir}\refs'
            path_to_darks = fr'\\132.187.193.8\junk\sgrischagin\2021-08-26-Sergej_SNR-Stufenkeil_130proj_6W\{dir}\darks'
            path_to_result = fr'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\{_date}\SNR'
            if not os.path.isdir(path_to_result):
                os.makedirs(path_to_result)

            data = file.volume.Reader(path_to_data, mode='raw', shape=img_shape, header=header).load_all()
            refs = file.volume.Reader(path_to_refs, mode='raw', shape=img_shape, header=header).load_all()
            darks = file.volume.Reader(path_to_darks, mode='raw', shape=img_shape, header=header).load_all()


            view_left = slice(0, 129), slice(130, 1370), slice(250, 750)
            map_slice = slice(None, None), slice(130, 1370), slice(250, 750)

            SNR_eval = SNR_Evaluator()
            filterer_l = ImageSeriesPixelArtifactFilterer()
            print(f'working on {dir}')

            results = []
            t_exp = _get_t_exp(path_to_data)
            SNR_eval.estimate_SNR(data[view_left], refs[view_left], darks[view_left], exposure_time=t_exp,
                                  pixelsize=pixel_size, pixelsize_units=pixel_size_units, series_filterer=filterer_l,
                                  u_nbins=nbins,
                                  save_path=os.path.join(path_to_result, f'SNR_{dir}_{d}_mm_expTime_{t_exp}'))
            figure = SNR_eval.plot(figure, f'{d}mm')
            results.append(SNR_eval)
        SNR_eval.finalize_figure(figure,
                                 title=f'SNR @ {dir} & {watt}W',
                                 save_path=os.path.join(path_to_result, f'{dir}_{d}'))



def man_SNR_eval_multi():
    c_date = datetime.datetime.now()
    _date = f'{c_date.year}-{c_date.month}-{c_date.day}_Evaluation'

    img_shape = (1536, 1944)
    header = 2048
    M = 18.3768
    watt = 6
    pixel_size = 4.07033
    pixel_size_units = '$\mu m$'
    nbins = 300
    px_map = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\BAD-PIXEL-bin1x1-scans-MetRIC.tif'

    dirs = ['40kV', '60kV', '80kV', '100kV', '120kV', '140kV', '160kV', '180kV']
    thicknesses = ['4u8', '5u9']
    for dir in dirs:
        figure = None
        for d in thicknesses:
            if d == '4u8':
                d_l = 4
                d_r = 8
            elif d == '5u9':
                d_l = 5
                d_r = 9
            else:
                d_l = 'x'
                d_r = 'x'
            path_to_data = fr'\\132.187.193.8\junk\sgrischagin\2021-08-26-Sergej_SNR-Stufenkeil_130proj_6W\{dir}\{d}'
            path_to_refs = fr'\\132.187.193.8\junk\sgrischagin\2021-08-26-Sergej_SNR-Stufenkeil_130proj_6W\{dir}\refs'
            path_to_darks = fr'\\132.187.193.8\junk\sgrischagin\2021-08-26-Sergej_SNR-Stufenkeil_130proj_6W\{dir}\darks'
            path_to_result = fr'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\{_date}\SNR'
            if not os.path.isdir(path_to_result):
                os.makedirs(path_to_result)

            data = file.volume.Reader(path_to_data, mode='raw', shape=img_shape, header=header).load_all()
            refs = file.volume.Reader(path_to_refs, mode='raw', shape=img_shape, header=header).load_all()
            darks = file.volume.Reader(path_to_darks, mode='raw', shape=img_shape, header=header).load_all()



            view_left = slice(0, 129), slice(130, 1370), slice(250, 750)
            view_right = slice(0, 129), slice(130, 1370), slice(1200, 1700)
            map_l = slice(130, 1370), slice(250, 750)
            map_r = slice(130, 1370), slice(1200, 1700)


            SNR_eval = SNR_Evaluator()
            filterer_l = ImageSeriesPixelArtifactFilterer(bad_pixel_map=map_l)
            filterer_r = ImageSeriesPixelArtifactFilterer(bad_pixel_map=map_r)
            print(f'working on {dir}')

            results = []
            t_exp = _get_t_exp(path_to_data)
            SNR_eval.estimate_SNR(data[view_left], refs[view_left], darks[view_left], exposure_time=t_exp, pixelsize=pixel_size,
                                  pixelsize_units=pixel_size_units, series_filterer=filterer_l, u_nbins=nbins,
                                  save_path=os.path.join(path_to_result, f'SNR_{dir}_{d_l}_mm_expTime_{t_exp}'))
            figure = SNR_eval.plot(figure, f'{d_l}mm')
            results.append(SNR_eval)

            SNR_eval.estimate_SNR(data[view_right], refs[view_right], darks[view_right], exposure_time=t_exp, pixelsize=pixel_size,
                                  pixelsize_units=pixel_size_units, series_filterer=filterer_r, u_nbins=nbins,
                                  save_path=os.path.join(path_to_result, f'SNR_{dir}_{d_r}_mm_expTime_{t_exp}'))

            figure = SNR_eval.plot(figure, f'{d_r}mm')
            results.append(SNR_eval)
        SNR_eval.finalize_figure(figure,
                                 title=f'SNR @ {dir} & {watt}W',
                                 save_path=os.path.join(path_to_result, f'{dir}'))




def man_T_eval_single():
    c_date = datetime.datetime.now()
    _date = f'{c_date.year}-{c_date.month}-{c_date.day}_{c_date.hour}_Evaluation'

    img_shape = (1536, 1944)
    header = 2048
    M = 18.3768
    watt = 6
    pixel_size = 4.07033
    pixel_size_units = '$\mu m$'
    nbins = 300


    dirs = ['40kV', '60kV', '80kV', '100kV', '120kV', '140kV', '160kV', '180kV']
    thicknesses = ['2']
    for dir in dirs:
        for d in thicknesses:
            path_to_data = fr'\\132.187.193.8\junk\sgrischagin\2021-08-26-Sergej_SNR-Stufenkeil_130proj_6W\{dir}\{d}'
            path_to_refs = fr'\\132.187.193.8\junk\sgrischagin\2021-08-26-Sergej_SNR-Stufenkeil_130proj_6W\{dir}\refs'
            path_to_darks = fr'\\132.187.193.8\junk\sgrischagin\2021-08-26-Sergej_SNR-Stufenkeil_130proj_6W\{dir}\darks'
            path_to_T_result = fr'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\{_date}\Transmission'
            if not os.path.isdir(path_to_T_result):
                os.makedirs(path_to_T_result)

            data = file.volume.Reader(path_to_data, mode='raw', shape=img_shape, header=header).load_all()
            refs = file.volume.Reader(path_to_refs, mode='raw', shape=img_shape, header=header).load_all()
            darks = file.volume.Reader(path_to_darks, mode='raw', shape=img_shape, header=header).load_all()

            view = slice(0, 129), slice(130, 1370), slice(250, 750)

            img = (data[view] - darks[view]) / (refs[view] - darks[view])
            img = np.mean(img, axis=0)
            T_l = np.min(img)

            _file = f'{d}_mm.csv'
            print(f'WRITING FILES FOR: {_file}')
            _path_T = os.path.join(path_to_T_result, _file)

            int_vol = int(dir.split('kV')[0])
            with open(os.path.join(_path_T), 'a+') as f_l:
                f_l.write('{};{}\n'.format(int_vol, T_l))
                f_l.close()



def man_T_eval_multi():
    c_date = datetime.datetime.now()
    _date = f'{c_date.year}-{c_date.month}-{c_date.day}_{c_date.hour}_Evaluation'

    img_shape = (1536, 1944)
    header = 2048
    M = 18.3768
    watt = 6
    pixel_size = 4.07033
    pixel_size_units = '$\mu m$'
    nbins = 300

    dirs = ['40kV', '60kV', '80kV', '100kV', '120kV', '140kV', '160kV', '180kV']
    thicknesses = ['4u8', '5u9']
    for dir in dirs:
        for d in thicknesses:
            if d == '4u8':
                d_l = 4
                d_r = 8
            elif d == '5u9':
                d_l = 5
                d_r = 9
            else:
                d_l = 'x'
                d_r = 'x'
            path_to_data = fr'\\132.187.193.8\junk\sgrischagin\2021-08-26-Sergej_SNR-Stufenkeil_130proj_6W\{dir}\{d}'
            path_to_refs = fr'\\132.187.193.8\junk\sgrischagin\2021-08-26-Sergej_SNR-Stufenkeil_130proj_6W\{dir}\refs'
            path_to_darks = fr'\\132.187.193.8\junk\sgrischagin\2021-08-26-Sergej_SNR-Stufenkeil_130proj_6W\{dir}\darks'
            path_to_T_result = fr'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\{_date}\Evaluation'
            if not os.path.isdir(path_to_T_result):
                os.makedirs(path_to_T_result)

            data = file.volume.Reader(path_to_data, mode='raw', shape=img_shape, header=header).load_all()
            refs = file.volume.Reader(path_to_refs, mode='raw', shape=img_shape, header=header).load_all()
            darks = file.volume.Reader(path_to_darks, mode='raw', shape=img_shape, header=header).load_all()

            view_left = slice(0, 129), slice(130, 830), slice(250, 750)
            view_right = slice(0, 129), slice(130, 830), slice(1200, 1700)

            img = (data[view_left] - darks[view_left]) / (refs[view_left] - darks[view_left])
            img = np.mean(img, axis=0)
            T_l = np.min(img)

            img = (data[view_right] - darks[view_right]) / (refs[view_right] - darks[view_right])
            img = np.mean(img, axis=0)
            T_r = np.min(img)

            file_l = f'{d_l}_mm.csv'
            file_r = f'{d_r}_mm.csv'
            print(f'WRITING FILES FOR:\n'
                  f'{file_l} \n'
                  f'AND \n'
                  f'{file_r}')

            _path_T_l = os.path.join(path_to_T_result, file_l)
            _path_T_r = os.path.join(path_to_T_result, file_r)

            int_kV = int(dir.split('kV')[0])

            with open(os.path.join(_path_T_l), 'a+') as f_l, open(os.path.join(_path_T_r), 'a+') as f_r:
                f_l.write('{};{}\n'.format(int_kV, T_l))
                f_r.write('{};{}\n'.format(int_kV, T_r))
                f_l.close()
                f_r.close()




def main():
    man_T_eval_single()


if __name__ == '__main__':
    main()