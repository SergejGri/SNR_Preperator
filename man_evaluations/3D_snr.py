import os
import numpy as np
from ext import file
import helpers as hlp



def interpolate_avg_num(angles: int):


    fCT_avg = fCT_data['avg_num']
    fCT_theta = fCT_data['theta']

    step = 360 / angles
    CT_theta = np.arange(0, 360, step)

    CT_avg = []
    istep = int(CT_theta.size / fCT_theta.size)

    for i in range(fCT_avg.size):
        for j in range(istep):
            CT_avg.append(fCT_avg[i])

    CT_avg = np.asarray(CT_avg)
    return CT_avg, CT_theta



def CT_multi_img():
    angles = 50
    imgs_per_angle = 120

    base_path = r'XXX'
    list_images = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]

    images = sorted(list_images)
    images_cp = images.copy()

    final_dir = os.path.join(base_path, f'merged-CT-{imgs_per_angle}-imgs-avg')
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
    working_dir = os.path.join(base_path, f'_TMP-FOLDER_')
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)


    CT_avg, CT_theta = interpolate_avg_num(angles=angles)

    for i in range(int(len(images)/imgs_per_angle)):
        nums = []
        iavg = CT_avg[i]

        for j in range(imgs_per_angle):
            os.rename(os.path.join(base_path, images_cp[j]), os.path.join(working_dir, images_cp[j]))
            num = hlp.extract_iternum_from_file(name=images_cp[j])
            nums.append(num)

        avg_img, filename = prep_substack(path=working_dir, fin_dir=final_dir, substack_avg=iavg, nums=nums)
        file.image.save(image=avg_img, filename=filename, suffix='raw', output_dtype=np.uint16)

        del images_cp[:imgs_per_angle]

    rm_info(path=final_dir)
    os.rmdir(working_dir)
    object.CT_data['CT_img'] = final_dir

    return final_dir



CT_multi_img()
