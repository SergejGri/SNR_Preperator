import numpy as np
from ext import file
from ext.SNR_spectra import ImageSeriesPixelArtifactFilterer
import helpers as hlp


class ImageLoader:
    def __init__(self, used_SCAP: bool = False, remove_lines: bool = False, load_px_map: bool = False,
                 crop_area: tuple = None):
        """
        ATTENTION:
        please note -> ref images MUST be loaded as the first image stack, when working with projections, refs and
        darks! Since the ratio between median intensity of the stack and the outlier pixel rows is most significant at
        ref images. If just projections are considered, the value for 'DEVIATION_THRESHOLD' in 'remove_detector_lines()'
        should be adjusted, since it's tuned for refs.


        :param used_SCAP: set value to True if you captured your images with the x-ray source in-house software SCAP.
        This is important, since captured images with 'Metric_Steuerung' Software are flipped and rotated in compare to
        SCAP images.
        :param remove_lines: if is True, detector slice line will be removed.
        """
        self.used_SCAP = used_SCAP
        self.remove_lines = remove_lines
        self.header = 2048
        self.shape = (1536, 1944)
        if self.used_SCAP:
            self.header = 0
            self.shape = (1944, 1536)
        if crop_area is not None:
            self.crop_area = crop_area
            self.view = (0, 0), *self.crop_area
        else:
            self.crop_area = (None, None)
            self.view = (None, None, None)
        self.idxs = []
        self.bad_px_map = load_px_map
        self.t_exp = None
        self.modified_px_map = None
        self.new_img_shape = None

        if self.bad_px_map:
            if self.used_SCAP:
                self.px_map = hlp.load_bad_pixel_map(crop=self.crop_area, scap=self.used_SCAP)
            else:
                self.px_map = hlp.load_bad_pixel_map(crop=self.crop_area, scap=self.used_SCAP)
            self.filterer = ImageSeriesPixelArtifactFilterer(bad_pixel_map=self.px_map)
        else:
            self.filterer = ImageSeriesPixelArtifactFilterer()


    def load_stack(self, path, stack_range = None):
        if not stack_range:
            images = file.volume.Reader(path, mode='raw', shape=self.shape, header=self.header, crops=self.view,
                                        dtype='<u2').load_all()
        else:
            images = file.volume.Reader(path, mode='raw', shape=self.shape, header=self.header, crops=self.view,
                                             dtype='<u2').load_range((stack_range[0], stack_range[-1]))

        if self.remove_lines:
            images = self.remove_detector_lines(images)
        return images


    def remove_detector_lines(self, img_stack):
        if len(self.idxs) < 1:
            DEVIATION_THRESHOLD = 0.15

            probe_img = img_stack[0]

            if self.used_SCAP:
                start, end = 0, probe_img.shape[1]
            else:
                start, end = 0, probe_img.shape[0]

            line_pos = 100
            line_plot = probe_img[line_pos-5:line_pos, start:end]
            line_plot = np.nanmean(line_plot, axis=0)
            line_median = np.nanmedian(line_plot)

            for i in range(len(line_plot)):
                px_val = line_plot[i]
                ratio = abs(1 - (px_val/line_median))

                if DEVIATION_THRESHOLD < ratio:
                    self.idxs.append(i)

        img_stack = np.delete(img_stack, self.idxs, axis=2)
        if self.bad_px_map:
            if self.px_map.shape[1] != img_stack.shape[2]: # only if the px_map was not updated yet, crop it to the new size
                self.px_map = np.delete(self.px_map, self.idxs, axis=1)
                self.filterer = ImageSeriesPixelArtifactFilterer(bad_pixel_map=self.px_map) # updating the filterer with new map
        self.new_img_shape = img_stack.shape
        return img_stack


    def load_filterer(self):
        if self.bad_px_map is True:
            filterer = ImageSeriesPixelArtifactFilterer(bad_pixel_map=self.modified_px_map)
            return filterer
        else:
            filterer = ImageSeriesPixelArtifactFilterer()
            return filterer