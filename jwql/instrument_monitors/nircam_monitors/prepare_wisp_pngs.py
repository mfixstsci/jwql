#! /usr/bin/env python

"""
Given a fits file, prepare an image from the data that can be provided to the model
"""

import argparse
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import os
from PIL import Image
import matplotlib.pyplot as plt


#file = 'jw01568001001_03101_00001_nrcb4_rate.fits'

#min_val = 0

def rescale_array(arr, max_val=None, new_max=255):
    """Rescales an array to the range 0-255."""
    min_val = np.nanmin(arr)
    if max_val is None:
        max_val = np.nanmax(arr)
    return ((arr - min_val) / (max_val - min_val)) * new_max


def add_options(parser=None, usage='', conflict_handler='resolve'):
    if parser is None:
        parser = argparse.ArgumentParser(usage=usage, conflict_handler=conflict_handler)

    parser.add_argument('filename', type=str, default='', help='File from which to create image')
    return parser


def run(filename, out_dir=None):
    data = fits.getdata(filename)
    #data = rescale_array(data, max_val=255)

    """
    IQR = np.percentile(data, 75) - np.percentile(data, 25)
    len_data = data.shape[-1] * data.shape[-2]
    bin_width_fd = 2 * IQR / np.power(len_data, 1/3)


    nbins = len(np.arange(0, 2, bin_width_fd))
    hist, edges = np.histogram(data, range=(0, 3), bins=nbins)


    peak_index = np.argmax(hist)

    # Get the value of the peak
    peak_value = hist[peak_index]
    toolow = np.where(hist > peak_value*0.1)[0]


    maximum_gray = edge[toolow[1]]
    """
    outfile_base = os.path.basename(filename).split('.')[0]

    mn, med, dev = sigma_clipped_stats(data)

    # Don't worry about any pixels more than 2-sigma from the peak value
    maximum_gray = med + dev * 1.
    minimum_gray = med #- dev * 2.

    #data = rescale_array(data, max_val=limit, new_max=255)
    #data[np.where(data > 255)] = 255

    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    adjusted_image = alpha * data + beta
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    img = Image.fromarray(adjusted_image)
    shrunk_img = img.resize(size=(256, 256))

    #h0 = fits.PrimaryHDU(shrunk_img)
    #hl = fits.HDUList([h0])
    #hl.writeto(f'{outfile_base}_adjusted.fits', overwrite=True)

    output_file = f'{outfile_base}.png'
    if out_dir is not None:
        output_file = os.path.join(out_dir, output_file)

    plt.imshow(shrunk_img, origin='lower')
    plt.axis('off')
    plt.savefig(output_file, bbox_inches='tight')
    return output_file


if __name__ == '__main__':
    parser = add_options()
    args = parser.parse_args()
    run(args.filename)
