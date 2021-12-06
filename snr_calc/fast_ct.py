

def fast_CT():
    stack_range = (50, 100)
    img_shape = (1536, 1944)
    header = 2048

    imgs = r'\\132.187.193.8\junk\sgrischagin\2021-10-04_Sergej_Res-phantom_135kV_1mean_texp195_15proj-per-angle\test_fast_CT'
    darks = r'\\132.187.193.8\junk\sgrischagin\2021-10-04_Sergej_Res-phantom_135kV_1mean_texp195_15proj-per-angle\darks'
    refs = r'\\132.187.193.8\junk\sgrischagin\2021-10-04_Sergej_Res-phantom_135kV_1mean_texp195_15proj-per-angle\refs'

    darks = file.volume.Reader(darks, mode='raw', shape=img_shape, header=header, dtype='u2').load_range((stack_range[0], stack_range[-1]))
    refs = file.volume.Reader(refs, mode='raw', shape=img_shape, header=header, dtype='u2').load_range((stack_range[0], stack_range[-1]))

    list_Ts = []
    list_angles = []
    data = file.volume.Reader(imgs, mode='raw', shape=img_shape, header=header, dtype='<u2').load_all()
    px_map = h.load_bad_pixel_map()

    for i in range(len(data)):
        if darks is None and refs is None:
            T = data[np.where(data > 0)].min()
        else:
            img = (data[i] - darks) / (refs - darks)
            T = img[np.where(img > 0)].min()
        list_Ts.append(T)
        list_angles.append(i)
    a = np.asarray(list_Ts)
    b = np.asarray(list_angles)
    del img, data, refs, darks
    gc.collect()
    return np.vstack([a, b])