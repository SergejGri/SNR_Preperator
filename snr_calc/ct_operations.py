import os
import gc
import re
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
from ext import file
from snr_calc.preperator import ImageLoader
import helpers as hlp


def CT(path_ct, path_refs, path_darks):
    start = timer()

    img_holder = ImageLoader(used_SCAP=False, remove_lines=False, load_px_map=False)

    refs = img_holder.load_stack(path=path_refs)
    refs_avg = np.nanmean(refs, axis=0)
    darks = img_holder.load_stack(path=path_darks)
    darks_avg = np.nanmean(darks, axis=0)

    data = img_holder.load_stack(path=path_ct)

    list_T = []
    list_angles = []

    all_imgs = [f for f in os.listdir(path_ct) if os.path.isfile(os.path.join(path_ct, f))]
    all_imgs = sorted(all_imgs)

    for k in range(data.shape[0]):
        theta = hlp.extract_angle(name=all_imgs[k], num_of_projections=len(all_imgs))

        img = (data[k] - darks_avg) / (refs_avg - darks_avg)
        median = []
        h = 20
        w = 20
        for i in range(0, img.shape[0] - h, h):
            for j in range(0, img.shape[1] - w, w):
                rect = img[i:i + h, j:j + w]
                medn = np.median(rect)
                median.append(medn)
        transmission_min = min(median)

        del img
        gc.collect()
        list_T.append(transmission_min)
        list_angles.append(theta)

    T = np.asarray(list_T)
    theta = np.asarray(list_angles)
    del data, refs, darks
    gc.collect()

    end = timer()
    print(f'fast_ct evaluation took: {end - start} s.')
    return T, theta



def CT_avg_imgs(path_ct, path_refs, path_darks, avgs_num: int):
    start = timer()

    img_holder = ImageLoader(used_SCAP=False, remove_lines=False, load_px_map=False)

    refs = img_holder.load_stack(path=path_refs)
    refs_avg = np.nanmean(refs, axis=0)
    darks = img_holder.load_stack(path=path_darks)
    darks_avg = np.nanmean(darks, axis=0)

    data = img_holder.load_stack(path=path_ct)

    list_T = []
    list_angles = []

    all_imgs = [f for f in os.listdir(path_ct) if os.path.isfile(os.path.join(path_ct, f))]
    all_imgs = sorted(all_imgs)

    for k in range(data.shape[0]):
        theta = hlp.extract_angle(name=all_imgs[k], num_of_projections=len(all_imgs))

        img = (data[k] - darks_avg) / (refs_avg - darks_avg)
        median = []
        h = 20
        w = 20
        for i in range(0, img.shape[0] - h, h):
            for j in range(0, img.shape[1] - w, w):
                rect = img[i:i + h, j:j + w]
                medn = np.median(rect)
                median.append(medn)
        transmission_min = min(median)

        del img
        gc.collect()
        list_T.append(transmission_min)
        list_angles.append(round(theta, 4))

    T = np.asarray(list_T)
    theta = np.asarray(list_angles)
    del data, refs, darks
    gc.collect()

    end = timer()
    print(f'fast_ct evaluation took: {end - start} s.')
    return T, theta


def avg_multi_img_CT(object, base_path, imgs_per_angle):

    abc = object.fCT_data
    images = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
    images = sorted(images)
    images_cp = images.copy()

    final_dir = os.path.join(base_path, f'merged-CT-{imgs_per_angle}-imgs-avg')
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
    working_dir = os.path.join(base_path, f'_TMP-FOLDER_')
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    for i in range(int(len(images)/imgs_per_angle)):
        nums = []

        for j in range(imgs_per_angle):
            os.rename(os.path.join(base_path, images_cp[j]), os.path.join(working_dir, images_cp[j]))
            num = hlp.extract_iternum_from_file(name=images_cp[j])
            nums.append(num)

        del images_cp[:imgs_per_angle]

        filename = os.path.join(final_dir, f'ct-avgimg-{nums[0]}-{nums[-1]}.raw')

        avg_img = prep_substack(path=working_dir)
        file.image.save(image=avg_img, filename=filename, suffix='raw', output_dtype=np.uint16)

    rm_info(path=final_dir)
    os.rmdir(working_dir)



def rm_info(path):
    for fname in os.listdir(path):
        if fname.lower().endswith('.info'):
            os.remove(os.path.join(path, fname))


def prep_substack(path: str):
    img_loader = ImageLoader(used_SCAP=False, remove_lines=False, load_px_map=False)
    img_stack = img_loader.load_stack(path=path)
    avg_img = np.nanmean(img_stack, axis=0)
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))
    return avg_img


