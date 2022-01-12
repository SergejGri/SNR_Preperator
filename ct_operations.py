import os
import gc
from timeit import default_timer as timer
import numpy as np
from ext import file
#from snr_calc.preperator import ImageLoader
import helpers as hlp


def CT(imgs: np.ndarray, refs: np.ndarray, darks: np.ndarray, detailed):
    """
    calculates minimum transmission per angle. Useful for stacks of images.
    """
    refs_avg = np.nanmean(refs, axis=0)
    darks_avg = np.nanmean(darks, axis=0)

    list_T = []
    for k in range(imgs.shape[0]):
        if detailed:
            print(f'Evaluating CT: {round((k/(imgs.shape[0]-1))*100, 2)} %')
        img = (imgs[k] - darks_avg) / (refs_avg - darks_avg)
        medians = []
        h = 20
        w = 20
        for i in range(0, img.shape[0] - h, h):
            for j in range(0, img.shape[1] - w, w):
                rect = img[i:i + h, j:j + w]
                medn = np.median(rect)
                medians.append(medn)
        transmission_min = min(medians)

        del img
        list_T.append(transmission_min)

    return np.asarray(list_T)



def CT_avg_imgs(path_ct, path_refs, path_darks):
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



def CT_multi_img(object, base_path, list_images, imgs_per_angle, img_loader):
    images = sorted(list_images)
    images_cp = images.copy()
    CT_avg = object.CT_data['avg_num']
    num_angles = int(len(list_images) / imgs_per_angle)
    angles = np.linspace(0, 360, num_angles, endpoint=False)
    sub_bin = object.sub_bin
    bin = int(imgs_per_angle / sub_bin)
    fin_dir = make_infrastructure(path=base_path)

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
            avg_img = calculate_avg(img_substack[start:end])
            path_and_name = os.path.join(fin_dir, name)
            file.image.save(image=avg_img, filename=path_and_name, suffix='raw', output_dtype=np.uint16)
            write_path_merge = os.path.join(os.path.dirname(fin_dir), 'merged_info.txt')
            write_path_avgs = os.path.join(os.path.dirname(fin_dir), 'avg_angle_info.txt')

            with open(write_path_merge, 'a+') as f_info, open(write_path_avgs, 'a+') as f_avg:
                f_info.write(f'AVG AT ANGLE: {round(angles[i], 2)} \n')
                f_info.write(f'MERGED: {images_cp[start]} UPTO {images_cp[end]} TO {name} \n')
                f_info.write('\n')
                f_avg.write(f'{name};{angles[i]};{_tmp_avg}\n')

        del img_substack, avg_img
        print(f'Done with: angle: {angles[i]}')

        images_cp = images_cp[imgs_per_angle:]
        stack_start += imgs_per_angle
    rm_info(path=fin_dir)
    return fin_dir


def calculate_avg(img_stack):
    if img_stack.shape[0] == 1:
        avg_img = img_stack
    else:
        avg_img = np.nanmean(img_stack, axis=0)
    return avg_img


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


'''
def CT_multi_img(object, base_path, list_images, imgs_per_angle):
    CT_avg = object.CT_data['avg_num']
    sub_bin = object.sub_bin
    angles = int(len(list_images)/imgs_per_angle)
    bin_size = int(imgs_per_angle / sub_bin)


    images = sorted(list_images)
    images_cp = images.copy()

    final_dir = os.path.join(base_path, f'merged-CT-{imgs_per_angle}-imgs-avg')
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
    working_dir = os.path.join(base_path, f'_TMP-FOLDER_')
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)


    for i in range(angles):
        nums = []
        iavg = CT_avg[i]
        idx = build_substack_index(bin_size=bin_size, stack=imgs_per_angle, avg=iavg)

        for j in range(imgs_per_angle):
            os.rename(os.path.join(base_path, images_cp[j]), os.path.join(working_dir, images_cp[j]))
            num = hlp.extract_iternum_from_file(name=images_cp[j])
            nums.append(num)

        # sub for loop: if iavg=2 then take avg(img and img1) throw img2 and img3

        avg_img, filename = prep_substack(path=working_dir, fin_dir=final_dir, idxs=idx, substack_avg=CT_avg[i], nums=nums)



        avg_img, filename = prep_substack(path=working_dir, fin_dir=final_dir, substack_avg=iavg, nums=nums)
        file.image.save(image=avg_img, filename=filename, suffix='raw', output_dtype=np.uint16)

        del images_cp[:imgs_per_angle]

    rm_info(path=final_dir)
    os.rmdir(working_dir)
    object.CT_data['CT_img'] = final_dir

    return final_dir
'''


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



def prep_substack(path: str, fin_dir: str, idxs: np.ndarray, substack_avg: int, nums: list):
    img_loader = ImageLoader(used_SCAP=False, remove_lines=False, load_px_map=False)
    img_substack = img_loader.load_stack(path=path)

    avg_end = nums[substack_avg - 1]
    j = 0
    for i in range(img_substack.shape[0]):
        _tmp_imgs_ = np.nanmean(img_substack[idxs], axis=0)
        #avg_sub_img = np.nanmean(img_substack[j:4], axis=0)
        filename = os.path.join(fin_dir, f'ct-avgimg-{nums[0]}-{avg_end}.raw')
        j += avg

    avg_img = np.nanmean(img_substack, axis=0)
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))
    return avg_img, filename



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



def rm_info(path):
    for fname in os.listdir(path):
        if fname.lower().endswith('.info'):
            os.remove(os.path.join(path, fname))


