import os
import file
from evaluation.SNR_spectra import ImageSeriesPixelArtifactFilterer, show_image, SNR_Evaluator





def split_imgs(path, dirs):
    list_1_section = []
    list_2_section = []
    list_3_section = []
    list_4_section = []
    for i in range(len(dirs)):
        for j in range(len(list_img_folders)):
            if '_0_' in list_img_folders:
                img_path = path + '\\' + dirs[i] + '\\' + list_img_folders[j]
                list_1_section.append(img_path)
            elif '_00_' in list_img_folders:
                img_path = path + '\\' + dirs[i] + '\\' + list_img_folders[j]
                list_2_section.append(img_path)
            elif '_000_' in list_img_folders:
                img_path = path + '\\' + dirs[i] + '\\' + list_img_folders[j]
                list_3_section.append(img_path)
            else:
                img_path = path + '\\' + dirs[i] + '\\' + list_img_folders[j]
                list_4_section.append(img_path)
    return list_1_section, list_2_section, list_3_section, list_4_section






def calc_SNR(path, result_path):
    finished = []
    results = []
    figure = None
    pixelsize = 74.8 / 2.95888
    smallest_size = 70
    list_left = ['5', '13', '21', '29']
    list_right = ['9', '17', '25', '33']
    # list_img_folders_1mmFilter = ['5u9mm', '13u17mm', '21u25mm', '29u33mm']
    list_img_folders = ['imgs']
    view_left = slice(0, 100), slice(15, 925), slice(530, 850)
    view_right = slice(0, 100), slice(15, 925), slice(1050, 1370)
    dark_folders = [os.path.join(path, folder) for folder in ['darks']]

    for path, dirs, files in os.walk(path):
        if 'darks' in dirs:
            dirs.remove('darks')
        for i in range(len(dirs)):
            result_path = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\allThick_per_voltage_v1_1mmFilter' + '\\' + dirs[i]
            str_voltage = dirs[i]
            images_folders = []
            refs_folders = []
            refs_path = base_path + '\\' + dirs[i] + '\\' + 'refs'
            refs_folders.append(refs_path)

            # split images depending on their name pattern
            # '_0_' == 4u8mm (without filter step wedge has a step width of 4mm)
            # '_00_' == 12u16mm
            # ...
            list_images = split_imgs(base_path, dirs)


            figure = None
            for k in range(len(images_folders)):
                print('Using voltage: ' + f'{str_voltage}')
                print('Using images: ' + f'{images_folders[k]}')
                print('Using refs: ' + f'{refs_folders[0]}')

                SNR_eval = SNR_Evaluator()
                filterer = ImageSeriesPixelArtifactFilterer()
                data = file.volume.Reader(images_folders[k], mode='raw', header=2048).load_all()
                refs = file.volume.Reader(refs_folders[0],  mode='raw', header=2048).load_all()
                darks = file.volume.Reader(dark_folders[0],  mode='raw', header=2048).load_all()

                SNR_eval.estimate_SNR(data[view_left], refs[view_left], darks[view_left], exposure_time=0.15, pixelsize=pixelsize, pixelsize_units='$\mu m$',
                                      series_filterer=filterer, save_path=os.path.join(result_path, f'SNR_{str_voltage}_{list_left[k]}_mm_150ms'))
                figure = SNR_eval.plot(figure, f'{list_left[k]} mm')
                results.append(SNR_eval)

                SNR_eval = SNR_Evaluator()
                filterer = ImageSeriesPixelArtifactFilterer()
                data = file.volume.Reader(images_folders[k]).load_all()
                refs = file.volume.Reader(refs_folders[0]).load_all()
                darks = file.volume.Reader(dark_folders[0]).load_all()

                SNR_eval.estimate_SNR(data[view_right], refs[view_right], darks[view_right], exposure_time=0.15, pixelsize=pixelsize, pixelsize_units='$\mu m$',
                                      series_filterer=filterer, save_path=os.path.join(result_path, f'SNR_{str_voltage}_{list_right[k]}_mm_150ms'))
                figure = SNR_eval.plot(figure, f'{list_right[k]} mm')
                results.append(SNR_eval)

                print("Done with " + dirs[i] + ': ' + f"{list_left[k]}mm and " + f"{list_right[k]}mm")

                del data
                del refs
                del darks

            SNR_eval.finalize_figure(figure, title=f'{str_voltage}', smallest_size=smallest_size, save_path=result_path + '\\' + f'{str_voltage}')
            print('...finalizing figure...')
            finished.append(dirs[i])
            progress = round((len(finished) / 29) * 100)
            print('Progress: ' + f'{progress}%')
            print('Done with ' + f'{str_voltage}')



if __name__ == '__main__':
    base_path = r'\\132.187.193.8\junk\sgrischagin\2021-05-03_Sergej_SNR_AluKeil_1mmAlufilter'
    result_path =
    calc_SNR(base_path)