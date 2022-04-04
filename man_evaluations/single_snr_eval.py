import os
from ext.SNR_spectra import SNR_Evaluator, ImageSeriesPixelArtifactFilterer
from image_loader import ImageLoader
import numpy as np

def eval():
    SNR_eval = SNR_Evaluator()
    filterer = ImageSeriesPixelArtifactFilterer()
    loader = ImageLoader(used_SCAP=True, remove_lines=False, load_px_map=False)

    kv = 80

    path_darks = os.path.join(r'\\132.187.193.8\junk\sgrischagin\2021-11-29-sergej-AluKeil-5W\80kV\darks')
    path_refs = os.path.join(r'\\132.187.193.8\junk\sgrischagin\2021-11-29-sergej-AluKeil-5W\80kV\refs')
    path_images = os.path.join(r'\\132.187.193.8\junk\sgrischagin\2021-11-29-sergej-AluKeil-5W\80kV\2')
    refs = loader.load_stack(path=path_refs, stack_range=(0, 280))
    darks = loader.load_stack(path=path_darks, stack_range=(0, 280))
    #imgs = loader.load_stack(path=path_images)
    imgs = loader.load_stack(path=path_images, stack_range=(0, 20))
    img_num = imgs.shape[0]
    px_size = 74.8/15.8429
    view = slice(None, None), slice(375, 1458), slice(690, 882)
    refs = refs[view]
    darks = darks[view]
    imgs = imgs[view]


    t_exp = 0.27

    results = []
    figure = None



    SNR_eval.estimate_SNR(images=imgs, refs=refs, darks=darks, pixelsize=px_size, pixelsize_units='px', exposure_time=t_exp, series_filterer=filterer)
    print(np.mean(SNR_eval.SNR))
    figure = SNR_eval.plot(figure, f'{kv} kV')
    results.append(SNR_eval)


    print('finalizing figure...')
    SNR_eval.finalize_figure(figure, title=f'img num: {img_num}',
                             save_path=os.path.join(r"C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\man_evaluations", f'{kv}kV_imgsnum{img_num}.pdf'))


eval()