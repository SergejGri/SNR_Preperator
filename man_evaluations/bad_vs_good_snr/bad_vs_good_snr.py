import os
from ext.SNR_spectra import SNR_Evaluator
from ext.SNR_spectra import ImageSeriesPixelArtifactFilterer
from image_loader import ImageLoader


def func():
    image_loader = ImageLoader(used_SCAP=True, remove_lines=False)
    EVA = SNR_Evaluator()
    filterer = ImageSeriesPixelArtifactFilterer()
    path_data = r'\\132.187.193.8\junk\sgrischagin\2021-11-29-sergej-AluKeil-5W\60kV'
    path_save = r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations\bad_vs_good_snr'
    view_good = slice(None, None), slice(450, 1650), slice(645, 888)
    view_bad = slice(None, None), slice(450, 1650), slice(250, 493)
    M = 15.8429
    pixel_size = 74.8 / M
    pixel_size_units = r'$\mu m$'
    texp = 0.37

    p_refs = os.path.join(path_data, 'refs')
    p_darks = os.path.join(path_data, 'darks')
    refs = image_loader.load_stack(path=p_refs, stack_range=(0, 70))
    darks = image_loader.load_stack(path=p_darks, stack_range=(0, 70))

    refs = refs[view_bad]
    darks = darks[view_bad]


    ds = [0, 4, 8, 12, 16, 28]
    figure = None
    results = []
    for i in range(len(ds)):
        p_imgs = os.path.join(path_data, str(ds[i]))
        data = image_loader.load_stack(path=p_imgs)
        data = data[view_bad]
        EVA.estimate_SNR(images=data, refs=refs, darks=darks, pixelsize=pixel_size,
                          pixelsize_units=pixel_size_units, exposure_time=texp, series_filterer=filterer,
                          save_path=os.path.join(path_save, 'badview', f'SNR-60kV-badview_{ds[i]}mm.pdf'))

        figure = EVA.plot(figure, f'{ds[i]}mm')
        results.append(EVA)
    EVA.finalize_figure(figure, save_path=os.path.join(path_save, 'badview', f'SNR-60kV-badview.pdf'))


func()