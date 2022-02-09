import os
from ext.SNR_spectra import SNR_Evaluator, ImageSeriesPixelArtifactFilterer
from ext import file
from image_loader import ImageLoader

def eval():
    view = slice(None, None), slice(200, 1700), slice(605, 1000)
    M = 15.8429
    watt = 5
    voltage = 60
    d = 2

    SNR_eval = SNR_Evaluator()
    filterer = ImageSeriesPixelArtifactFilterer()
    loader = ImageLoader(used_SCAP=True, remove_lines=False, load_px_map=False)


    path_darks = os.path.join(r'\\132.187.193.8\junk\sgrischagin\2021-11-29-sergej-AluKeil-5W\60kV\darks')
    path_refs = os.path.join(r'\\132.187.193.8\junk\sgrischagin\2021-11-29-sergej-AluKeil-5W\60kV\refs')
    path_images = os.path.join(rf'\\132.187.193.8\junk\sgrischagin\2021-11-29-sergej-AluKeil-5W\60kV\{d}')
    save_path = r'C:\Users\Sergej Grischagin\Desktop\test_images_for_thesis'

    refs = loader.load_stack(path=path_refs, stack_range=(0, 50))
    darks = loader.load_stack(path=path_darks, stack_range=(0, 50))
    imgs = loader.load_stack(path=path_images)


    imgs = imgs[view]
    refs = refs[view]
    darks = darks[view]



    results = []
    figure = None

    SNR_eval.estimate_SNR(images=imgs, refs=refs, darks=darks, pixelsize=74.8/M, exposure_time=0.37,
                          pixelsize_units='$\mu m$', series_filterer=filterer, save_path=save_path)


    figure = SNR_eval.plot(figure, f'{d} mm', only_snr=False)
    results.append(SNR_eval)

    print('finalizing figure...')
    SNR_eval.finalize_figure(figure, save_path=os.path.join(save_path, f'{voltage}kV'))


eval()