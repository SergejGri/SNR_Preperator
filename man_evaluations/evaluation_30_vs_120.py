import os
import numpy as np
from ext.SNR_spectra import SNR_Evaluator, ImageSeriesPixelArtifactFilterer
from image_loader import ImageLoader


def evaluate():
    # Please update these paths to your local or project-specific directories
    path_imgs_30 = 'PATH_TO_IMAGES_30'
    path_imgs_120 = 'PATH_TO_IMAGES_120'
    path_refs = 'PATH_TO_REFS'
    path_darks = 'PATH_TO_DARKS'

    view = slice(None, None), slice(50, 945), slice(866, 1040)
    loader = ImageLoader(used_SCAP=False, remove_lines=False, load_px_map=False)
    loader_nh = ImageLoader(used_SCAP=False, remove_lines=False, load_px_map=False)
    loader_nh.header = 0

    EVA = SNR_Evaluator()
    filterer = ImageSeriesPixelArtifactFilterer()

    imgs_120 = loader.load_stack(path=path_imgs_120, stack_range=(0, 120))
    refs_120 = loader.load_stack(path=path_refs)
    darks_120 = loader.load_stack(path=path_darks)

    EVA.estimate_SNR(images=imgs_120)
    # Add the rest of your logic here as needed


evaluate()
