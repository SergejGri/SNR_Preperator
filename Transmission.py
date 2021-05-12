import csv
import os
import numpy as np
from scipy import ndimage, misc
import file
from evaluation.SNR_spectra import ImageSeriesPixelArtifactFilterer, show_image, SNR_Evaluator

base_path = r'\\132.187.193.8\junk\sgrischagin\Dritte_Messung'
result_path = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\Transmission_v5_mean'


pixelsize = 74.8 / 2.95888
smallest_size = 70
list_left = ['4', '12', '20', '28']
list_right = ['8', '16', '24', '32']
list_img_folders = ['4u8mm', '12u16mm', '20u24mm', '28u32mm']
figure = None
finished = []
view_left = slice(0, 100), slice(15, 925), slice(530, 850)
view_right = slice(0, 100), slice(15, 925), slice(1050, 1370)

#dark_folders = [os.path.join(base_path, folder) for folder in ['darks']]

def transmission():
    for path, dirs, files in os.walk(base_path):
        if 'darks' in dirs:
            dirs.remove('darks')
        for i in range(len(dirs)):
            images_folders = []
            refs_folders = []
            refs_path = base_path + '\\' + dirs[i] + '\\' + 'refs'
            refs_folders.append(refs_path)

            for j in range(len(list_img_folders)):
                img_path = base_path + '\\' + dirs[i] + '\\' + list_img_folders[j]
                images_folders.append(img_path)

            min_val = []
            for k in range(len(images_folders)):
                kernel_size = 3
                print('working on ' + list_left[k] + 'mm and ' + list_right[k] + 'mm ' + ' / ' + dirs[i])
                working_file_l = result_path + '\\' + 'minima_' + list_left[k] + '_mm.csv'
                working_file_r = result_path + '\\' + 'minima_' + list_right[k] + '_mm.csv'
                filterer = ImageSeriesPixelArtifactFilterer()

                data = file.volume.Reader(images_folders[k]).load_all()
                refs = file.volume.Reader(refs_folders[0]).load_all()
                corr_data = ndimage.median_filter(data[view_left], size=kernel_size)
                corr_refs = ndimage.median_filter(refs[view_left], size=kernel_size)
                T_left = corr_data / corr_refs
                T_mean_left = np.mean(np.amin(T_left, axis=0))
                del data
                del refs
                del corr_data
                del corr_refs

                data = file.volume.Reader(images_folders[k]).load_all()
                refs = file.volume.Reader(refs_folders[0]).load_all()
                corr_data = ndimage.median_filter(data[view_right], size=kernel_size)
                corr_refs = ndimage.median_filter(refs[view_right], size=kernel_size)
                T_right = corr_data / corr_refs
                T_mean_right = np.mean(np.amin(T_right, axis=0))
                del data
                del refs
                del corr_data
                del corr_refs

                print('... writing files ...')
                with open(working_file_l, 'a+') as f_l, open(working_file_r, 'a+') as f_r:
                    str_voltage = dirs[i]
                    size = len(str_voltage)
                    voltage = str_voltage[:size-3]
                    f_l.write('{};{}\n'.format(int(voltage), T_mean_left))
                    f_r.write('{};{}\n'.format(int(voltage), T_mean_right))
                    f_l.close()
                    f_r.close()
            finished.append(dirs[i])
            progress = round((len(finished) / 29) * 100)
            print('Progress: ' + f'{progress}%')

# sort files
def read_data(path):
    for file in os.listdir(path):
        data = np.loadtxt(path + '\\' + file)
        sorted_data = data[data[:, 0].argsort()]
        np.savetxt(path + '\\sorted_' + file, sorted_data, delimiter='@@@')


if __name__ == '__main__':
    transmission()
    #path_to_T = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\Transmission_v4_mean'
    #read_data(path_to_T)