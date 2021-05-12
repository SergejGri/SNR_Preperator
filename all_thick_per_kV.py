import os
import file
import fnmatch as fn
from evaluation.SNR_spectra import ImageSeriesPixelArtifactFilterer, show_image, SNR_Evaluator


def split_imgs(path, dir, files):
    list_1_section = []
    list_2_section = []
    list_3_section = []
    list_4_section = []

    for _, _, files in os.walk(files):
        for file in files:
            if '_0_' in file:
                list_1_section.append(file)
            elif '_00_' in file:
                list_2_section.append(file)
            elif '_000_' in file:
                list_3_section.append(file)
            else:
                list_4_section.append(file)
    return list_1_section, list_2_section, list_3_section, list_4_section


def mat_naming(filter):
    if filter == '0':
        list_left_0mm = ['4', '12', '20', '28']
        list_right_0mm = ['8', '16', '24', '32']
        return list_left_0mm, list_right_0mm
    if filter == '1':
        list_left_1mm = ['5', '13', '21', '29']
        list_right_1mm = ['9', '17', '25', '33']
        return list_left_1mm, list_right_1mm
    if filter == '2':
        list_left_2mm = ['6', '14', '22', '30']
        list_right_2mm = ['10', '18', '26', '34']
        return list_left_2mm, list_right_2mm







def calc_SNR_data(paths, filters, t_exp, px_size, px_units, smallest_size):
    finished = []
    results = []

    for path in paths:
        view_left = slice(0, 100), slice(15, 925), slice(530, 850)
        view_right = slice(0, 100), slice(15, 925), slice(1050, 1370)
        dark_folders = [os.path.join(path, folder) for folder in ['darks']]


        for path, dirs, files in os.walk(path):
            if 'darks' in dirs:
                dirs.remove('darks')
            for i in range(len(dirs)):
                result_path = os.path.join(r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\MAP', dirs[i])
                str_voltage = dirs[i]
                refs_folders = []
                refs_path = os.path.join(path, dirs[i], 'refs')
                refs_folders.append(refs_path)

                # split images depending on their name pattern
                # '_0_' == 4u8mm (without filter step wedge has a step width of 4mm)
                # '_00_' == 12u16mm
                # ...
                print(f'Working on {str_voltage}')
                list_images = split_imgs(path, dirs[i], os.path.join(path, dirs[i], 'imgs'))

                SNR_eval = SNR_Evaluator()
                filterer = ImageSeriesPixelArtifactFilterer()
                figure = None
                for k in range(len(list_images)):
                    list_left, list_right = mat_naming(filters)
                    data = file.volume.Reader(list_images[k], mode='raw', header=2048).load_all()
                    refs = file.volume.Reader(refs_folders[0],  mode='raw', header=2048).load_all()
                    darks = file.volume.Reader(dark_folders[0],  mode='raw', header=2048).load_all()
                    SNR_eval.estimate_SNR(data[view_left], refs[view_left], darks[view_left],
                                          exposure_time=t_exp,
                                          pixelsize=px_size,
                                          pixelsize_units=px_units,
                                          series_filterer=filterer,
                                          save_path=os.path.join(result_path, f'SNR_{str_voltage}_{list_left[k]}mm_{t_exp}ms'))
                    figure = SNR_eval.plot(figure, f'{list_left[k]} mm')
                    results.append(SNR_eval)
                    del data
                    del refs
                    del darks

                    data = file.volume.Reader(list_images[k]).load_all()
                    refs = file.volume.Reader(refs_folders[0]).load_all()
                    darks = file.volume.Reader(dark_folders[0]).load_all()
                    SNR_eval.estimate_SNR(data[view_right], refs[view_right], darks[view_right],
                                          exposure_time=t_exp,
                                          pixelsize=px_size,
                                          pixelsize_units=px_units,
                                          series_filterer=filterer,
                                          save_path=os.path.join(result_path, f'SNR_{str_voltage}_{list_right[k]}mm_{t_exp}ms'))
                    figure = SNR_eval.plot(figure, f'{list_right[k]} mm')
                    results.append(SNR_eval)
                    del data
                    del refs
                    del darks
                    print("Done with " + dirs[i] + ': ' + f"{list_left[k]}mm and " + f"{list_right[k]}mm")

                SNR_eval.finalize_figure(figure,
                                         title=f'{str_voltage}',
                                         smallest_size=smallest_size,
                                         save_path=os.path.join(result_path, f'{str_voltage}'))
                print('...finalizing figure...')
                finished.append(dirs[i])
                progress = round((len(finished) / 29) * 100)
                print('Progress: ' + f'{progress}%')
                print('Done with ' + f'{str_voltage}')


if __name__ == '__main__':
    base_paths = [r'\\132.187.193.8\junk\sgrischagin\2021-04-20_Sergej_SNR_AluKeil_0mmAlufilter',
                  r'\\132.187.193.8\junk\sgrischagin\2021-05-03_Sergej_SNR_AluKeil_1mmAlufilter',
                  r'\\132.187.193.8\junk\sgrischagin\2021-04-29_Sergej_SNR_AluKeil_2mmAlufilter']
    result_name = 'Alu_Keil'
    exposure_t = 150
    pixel_size = 74.8 / 2.95888
    pixel_size_units = '$\mu m$'
    smallest_size = 70
    filter_sizes = ['0', '1', '2'] # !!! its important that the order of passing paths and filter sizes are exactly matched !!!

    calc_SNR_data(base_paths,
                  filters=filter_sizes,
                  t_exp=exposure_t,
                  px_size=pixel_size,
                  px_units=pixel_size_units,
                  smallest_size=smallest_size)