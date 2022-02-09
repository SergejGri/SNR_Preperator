import os
import numpy as np
from ext.SNR_spectra import SNR_Evaluator, ImageSeriesPixelArtifactFilterer
from image_loader import ImageLoader


def evaluate():
    path_imgs_30 = r'\\132.187.193.8\junk\sgrischagin\2021-12-22-sergej-CT-halbesPhantom-102kV-100ms-5W-M4p46\merged-CT'
    path_imgs_120 = r'\\132.187.193.8\junk\sgrischagin\2021-12-22-sergej-CT-halbesPhantom-102kV-100ms-5W-M4p46\imgs'
    path_refs = r'\\132.187.193.8\junk\sgrischagin\2021-12-22-sergej-CT-halbesPhantom-102kV-100ms-5W-M4p46\refs'
    path_darks = r'\\132.187.193.8\junk\sgrischagin\2021-12-22-sergej-CT-halbesPhantom-102kV-100ms-5W-M4p46\darks'

    view = slice(None, None), slice(50, 945), slice(866, 1040)
    loader = ImageLoader(used_SCAP=False, remove_lines=False, load_px_map=False)
    loader_nh = ImageLoader(used_SCAP=False, remove_lines=False, load_px_map=False)
    loader_nh.header = 0

    EVA = SNR_Evaluator()
    filterer = ImageSeriesPixelArtifactFilterer()

    imgs_120 = loader.load_stack(path=path_imgs_120, stack_range=(0, 120))
    refs_120 = loader.load_stack(path=)
    darks_120 =


    EVA.estimate_SNR(images=)


evaluate()