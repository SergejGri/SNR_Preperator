import os
import gc
from timeit import default_timer as timer
import numpy as np

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
        theta = hlp.extract_angle(num_of_projections=len(all_imgs), img_name=all_imgs[k])

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
        theta = hlp.extract_angle(num_of_projections=len(all_imgs), img_name=all_imgs[k])

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


def split_multi_CT(base_path, imgs_per_angle):
    num_of_cts = [cti for cti in range(imgs_per_angle + 1)][1:]

    for j in range(len(num_of_cts)):
        single_ct_path = os.path.join(base_path, 'CT_', num_of_cts[j])
        if not os.path.exists(single_ct_path):
            os.makedirs(single_ct_path)

    pass