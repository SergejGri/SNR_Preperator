import os
from ext.SNR_spectra import SNR_Evaluator, ImageSeriesPixelArtifactFilterer
from image_loader import ImageLoader

def eval():
    SNR_eval = SNR_Evaluator()
    filterer = ImageSeriesPixelArtifactFilterer()
    loader = ImageLoader(used_SCAP=False, remove_lines=False, load_px_map=False)


    path_darks = os.path.join(r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\fCT-90kV-50proj\darks')
    path_refs = os.path.join(r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\fCT-90kV-50proj\refs')
    path_images = os.path.join(r'\\132.187.193.8\junk\sgrischagin\2021-12-14-sergej-CT-halbesPhantom-100ms-5W-M4p46\fCT-90kV-50proj\imgs')
    imgs_refs = loader.load_stack(path=path_refs)
    imgs_darks = loader.load_stack(path=path_darks)
    imgs_imgs = loader.load_stack(path=path_darks)


    t_exp = 0.1

    results = []
    figure = None

    for i in range(imgs_imgs.shape[0]):

        print(f'working on {dir}: {_d} mm')

        SNR_eval, figure = SNR_eval.estimate_SNR(images=imgs_imgs, refs=imgs_refs, darks=imgs_darks, pixelsize=74
                                                 exposure_time=t_exp, filterer=filterer)



       pixelsize = 1.0, pixelsize_units = 'px', compute_N0 = False
        figure = SNR_eval.plot(figure, f'{_d} mm')
        results.append(SNR_eval)


    print('finalizing figure...')
    SNR_eval.finalize_figure(figure, title=f'{dir} @{self.watt}W',
                             save_path=os.path.join(psave_SNR, f'{voltage}kV'))


eval()