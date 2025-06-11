import os

import numpy as np
import matplotlib.pyplot as plt
from ext.SNR_spectra import SNR_Evaluator, ImageSeriesPixelArtifactFilterer
from ext import file
from image_loader import ImageLoader


def eval():
    view = slice(None, None), slice(200, 1700), slice(605, 1000)
    M = 15.8429
    watt = 5
    voltage = 60
    d = 16
    texp = 0.37

    SNR_eval = SNR_Evaluator()
    filterer = ImageSeriesPixelArtifactFilterer()
    loader = ImageLoader(used_SCAP=True, remove_lines=False, load_px_map=False)


    path_darks = os.path.join(r'XXX')
    path_refs = os.path.join(r'XXX')
    path_images = os.path.join(rf'XXX')
    save_path = r'XXX'

    refs = loader.load_stack(path=path_refs, stack_range=(0, 50))
    darks = loader.load_stack(path=path_darks, stack_range=(0, 50))
    imgs = loader.load_stack(path=path_images)


    imgs = imgs[view]
    refs = refs[view]
    darks = darks[view]



    results = []
    figure = None

    SNR_eval.estimate_SNR(images=imgs, refs=refs, darks=darks, pixelsize=74.8/M, exposure_time=texp,
                          pixelsize_units='$\mu m$', series_filterer=filterer, save_path=save_path)


    figure = SNR_eval.plot(figure, f'{d} mm', only_snr=False)
    results.append(SNR_eval)

    print('finalizing figure...')
    SNR_eval.finalize_figure(figure, save_path=os.path.join(save_path, f'{voltage}kV_{d}'))



def plot_file():
    data = np.genfromtxt(r'XXX',
                         skip_header=3, delimiter=' ')
    x = data[:, 0]
    x = 1/(2*x)
    SNR = data[:, 1]
    SPS = data[:, 2]
    NPS = data[:, 3]

    plt.plot(x, SNR, label='SNR')
    plt.plot(x, SPS, label='SPS')
    plt.plot(x, NPS, label='NPS')

    plt.gca().invert_xaxis()
    plt.yscale('log')
    plt.tight_layout()
    plt.legend()
    plt.savefig(r'XXX', dpi=600)
    plt.show()




plot_file()
