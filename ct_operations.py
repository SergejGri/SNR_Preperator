import os
import numpy as np
from ext import file
import helpers as hlp


def calc_T_for_stack(imgs: np.ndarray, refs: np.ndarray, darks: np.ndarray, detailed):
    """
    calculates minimum transmission per angle. Useful for stacks of images.
    """

    refs_avg = np.nanmean(refs, axis=0)
    darks_avg = np.nanmean(darks, axis=0)
    list_T = []
    list_theta = []
    STEP = round(360/imgs.shape[0], 2)

    angle = 0
    for k in range(imgs.shape[0]):
        if detailed:
            print(f'Evaluating CT: {round((k/(imgs.shape[0]-1))*100, 2)} %')
        img = (imgs[k] - darks_avg) / (refs_avg - darks_avg)

        img_min = [x for x in img.flatten() if x > 0.0]
        img2 = [x for x in img_min if x < 1.0]

        rng_min = np.nanmin(img2)
        if rng_min <= 0.0:
            rng_min = 0.0
        rng_max = np.nanmax(img2)
        if rng_max > 1.0:
            rng_min = 1.0

        sum = 0
        counter = 0
        for i in range(0, img.shape[0], 1):
            for j in range(0, img.shape[1], 1):
                if rng_min <= img[i, j] <= ((rng_max - rng_min) / 3 + rng_min):
                    sum += img[i, j]
                    counter += 1
        sum = sum / counter
        list_T.append(sum)
        list_theta.append(round(angle, 2))
        del img
        angle += STEP
    return np.asarray(list_T), np.asarray(list_theta)

def merging_multi_img_CT(object, base_path, l_images, imgs_per_angle, img_loader):
    images_cp = l_images.copy()
    CT_avg = object.CT_data['avg_num']
    angles_num = int(len(l_images) / imgs_per_angle)
    angles = np.linspace(0, 360, angles_num, endpoint=False)
    sub_bin = object.sub_bin
    bin = int(imgs_per_angle / sub_bin)

    fin_dir = object.paths['merged_imgs']

    stack_start = 0
    for i in range(angles.shape[0]):
        _tmp_avg = CT_avg[i]
        img_ids = []
        stack_end = stack_start + imgs_per_angle

        for j in range(imgs_per_angle):
            id = hlp.extract_iternum_from_file(name=images_cp[j])
            img_ids.append(id)

        img_substack = img_loader.load_stack(path=base_path, stack_range=(stack_start, stack_end))
        for k in range(0, img_substack.shape[0], bin):
            start = 0 + k
            end = start + _tmp_avg
            str1 = img_ids[start]
            str2 = img_ids[end-1]
            name = f'ct-avg-{str1}-{str2}.raw'
            avg_img = hlp.calculate_avg(img_substack[start:end])
            path_and_name = os.path.join(fin_dir, name)
            file.image.save(image=avg_img, filename=path_and_name, suffix='raw', output_dtype=np.uint16)

        del img_substack, avg_img
        print(f'Done with: angle: {round(angles[i],1)}')
        images_cp = images_cp[imgs_per_angle:]
        stack_start += imgs_per_angle
    hlp.rm_files(path=fin_dir, extension='info')
    return fin_dir

def remove_tmp_imgs(path):
    for f in os.listdir(path):
        if not f.endswith(".raw"):
            continue
        os.remove(os.path.join(path, f))

def make_infrastructure(path):
    final_dir = os.path.join(os.path.dirname(path), f'merged-CT')
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
    return final_dir

def make_num(num):
    num = str(num)
    length = len(num)
    if length == 1:
        num_str = '000' + num
    elif length == 2:
        num_str = '00' + num
    elif length == 3:
        num_str = '0' + num
    else:
        num_str = num
    return num_str

def build_substack_index(bin_size, stack, avg):
    idx = [[np.arange(bin_size)] for _ in range(int(stack/bin_size))]
    idx = np.asarray(idx)

    lidx = bin_size-avg
    if lidx == 0:
        idx = idx[:, :, :]
    else:
        idx = idx[:, :, :-lidx]

    '''j = 0
    while j < stack:
        i = 0
        sub_idx = []
        while i < bin_size:
            if i <= avg:
                sub_idx.append(j+i)
                i += 1
            else:
                j += bin_size
                break
        idx.append()'''
    return idx






