#! /usr/bin/env python

"""
Given a fits file, prepare an image of the data that can be provided to the ML wisp
prediction model.
"""

import argparse
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import os
from PIL import Image
import matplotlib.pyplot as plt


def rescale_array(arr, max_val=None, new_max=255):
    """Rescales an array to the range 0-255.

    Parameters
    ----------
    arr : nump.ndarray
        2D image array

    max_val : float
        Maximum value to use in the input image

    new_max : float
        Maximum value in the rescaled image

    Returns
    -------
    arr : numpy.ndarray
        Rescaled image
    """
    min_val = np.nanmin(arr)
    if max_val is None:
        max_val = np.nanmax(arr)
    return ((arr - min_val) / (max_val - min_val)) * new_max


def add_options(parser=None, usage='', conflict_handler='resolve'):
    """
    Add command line options

    Parrameters
    -----------
    parser : argparse.parser
        Parser object

    usage : str
        Usage string

    conflict_handler : str
        Conflict handling strategy

    Returns
    -------
    parser : argparse.parser
        Parser object with added options
    """
    if parser is None:
        parser = argparse.ArgumentParser(usage=usage, conflict_handler=conflict_handler)

    parser.add_argument('filename', type=str, default='', help='File from which to create image')
    return parser


def run(filename, out_dir=None):
    """Main function. Read in fits file, create scaled and resized image. Save
    as png.

    Parameters
    ----------
    filename : str
        Name of fits file

    out_dir : str
        Output directory in which to save the final png file

    Returns
    -------
    output_file : str
        Full path to the output png file
    """
    data = fits.getdata(filename)

    # Get the basename of the input file. This will be used to create
    # the output png file name
    outfile_base = os.path.basename(filename).split('.')[0]

    # Calculate basic stats on the image
    mn, med, dev = sigma_clipped_stats(data)

    # Don't worry about any pixels more than 2-sigma from the peak value
    maximum_gray = med + dev * 1.
    minimum_gray = med

    # Calculate scaling factor and contrast adjustment
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    # Rescale the image
    adjusted_image = alpha * data + beta
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    # Resize image to 256x256 pixels
    img = Image.fromarray(adjusted_image)
    shrunk_img = img.resize(size=(256, 256))

    # Create output filename
    output_file = f'{outfile_base}.png'
    if out_dir is not None:
        output_file = os.path.join(out_dir, output_file)

    # Create image and save
    plt.imshow(shrunk_img, origin='lower')
    plt.axis('off')
    plt.savefig(output_file, bbox_inches='tight')
    return output_file


if __name__ == '__main__':
    parser = add_options()
    args = parser.parse_args()
    run(args.filename)
