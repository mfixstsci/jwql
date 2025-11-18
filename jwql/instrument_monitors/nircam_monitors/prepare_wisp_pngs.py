#! /usr/bin/env python

"""
Given a fits file, prepare an image of the data that can be provided to the ML wisp
prediction model.
"""

import argparse
import numpy as np
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import os
from PIL import Image
import matplotlib.pyplot as plt


def create_figure(image, outfile):
    """Create a figure of the scaled, resized image

    Parameters
    ----------
    image : PIL.Image.Image
        Image to be saved

    outfile : str
        Name of file it save the image into
    """
    plt.imshow(image, origin='lower')
    plt.axis('off')
    plt.savefig(outfile, bbox_inches='tight')
    plt.close('all')


def fill_nan_with_nearest_neighbor(arr):
    """
    Replaces NaN values in a 2D NumPy array with values interpolated
    from the nearest non-NaN neighbors.

    Parameters
    ----------
    arr : numpy.ndarray
        A 2D NumPy array potentially containing NaN values.

    Returns
    -------
    filled_arr : numpy.ndarray
        A new array with NaNs replaced by nearest neighbor interpolation.
    """
    kernel = Gaussian2DKernel(x_stddev=1, y_stddev=1)
    filled_arr = interpolate_replace_nans(arr, kernel)

    return filled_arr


def rescale_array(arr):
    """Rescales an array to the range 0-255.

    Parameters
    ----------
    arr : nump.ndarray
        2D image array

    Returns
    -------
    adjusted_image : numpy.ndarray
        Rescaled image
    """
    # Calculate basic stats on the image
    mn, med, dev = sigma_clipped_stats(arr)

    # Don't worry about any pixels more than 2-sigma from the peak value
    maximum_gray = med + dev * 1.
    minimum_gray = med

    # Calculate scaling factor and contrast adjustment
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    # Rescale the image
    adjusted_image = alpha * arr + beta
    mask = ~np.isnan(adjusted_image)
    adjusted_image[mask] = np.clip(adjusted_image[mask], 0, 255).astype(np.uint8)

    return adjusted_image


def resize_image(arr):
    """Resize the input image to the size expected by the ML model

    Parameters
    ----------
    arr : numpy.ndarray
        2D image to te resized

    Returns
    -------
    resized_image : PIL.Image.Image
        Resized image
    """
    img = Image.fromarray(arr)
    resized_image = img.resize(size=(256, 256))
    return resized_image


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

    # Replace NaN values with interpolated values from nearest neighbors
    data = fill_nan_with_nearest_neighbor(data)

    # Get the basename of the input file. This will be used to create
    # the output png file name
    outfile_base = os.path.basename(filename).split('.')[0]

    # Rescale and adjust contrast of the image
    adjusted_image = rescale_array(data)

    # Resize image to 256x256 pixels
    shrunk_img = resize_image(adjusted_image)

    # Create output filename
    output_file = f'{outfile_base}.png'
    if out_dir is not None:
        output_file = os.path.join(out_dir, output_file)

    # Create image and save
    create_figure(shrunk_img, output_file)
    return output_file


if __name__ == '__main__':
    parser = add_options()
    args = parser.parse_args()
    run(args.filename)
