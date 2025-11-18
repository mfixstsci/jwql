#! /usr/bin/env python

"""
Create a preview image from a fits file containing an observation.

This module creates and saves a "preview image" from a fits file that
contains a JWST observation. Data from the user-supplied ``extension``
of the file are read in, along with the ``PIXELDQ`` extension if
present. For each integration in the exposure, the first group is
subtracted from the final group in order to create a difference image.
The lower and upper limits to be displayed are defined as the
``clip_percent`` and ``(1. - clip_percent)`` percentile signals.
``matplotlib`` is then used to display a linear- or log-stretched
version of the image, with accompanying colorbar. The image is then
saved.

Authors:
--------

    - Bryan Hilbert

Use:
----

    This module can be imported as such:

    ::

        from jwql.preview_image.preview_image import PreviewImage
        im = PreviewImage(my_file, "SCI")
        im.clip_percent = 0.01
        im.scaling = 'log'
        im.output_format = 'jpg'
        im.make_image()
"""

from glob import glob
import logging
import os
import re
import socket
import warnings

from astropy import constants as const
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.visualization import LinearStretch, LogStretch, MinMaxInterval, SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import numpy as np
import pandas as pd

from jwst import datamodels

from jwql.utils import permissions
from jwql.utils.constants import ON_GITHUB_ACTIONS, ON_READTHEDOCS
from jwql.utils.utils import filesystem_path, get_config

# Use the 'Agg' backend to avoid invoking $DISPLAY
import matplotlib as mpl
#matplotlib.use('Agg') - uncomment before merging!!!!!!!!!!!!!
import matplotlib.pyplot as plt  # noqa
import matplotlib.colors as colors  # noqa
from matplotlib.colors import SymLogNorm  # noqa
from matplotlib.ticker import AutoMinorLocator, SymmetricalLogLocator  # noqa
from mpl_toolkits.axes_grid1 import make_axes_locatable  # no_qa

if not ON_READTHEDOCS:
    from jwst.datamodels import dqflags

if not ON_GITHUB_ACTIONS and not ON_READTHEDOCS:
    CONFIGS = get_config()


class PreviewImage():
    """An object for generating and saving preview images, used by
    ``generate_preview_images``.

    Attributes
    ----------
    clip_percent : float
        The amount to sigma clip the input data by when scaling the
        preview image.  Default is 0.01.
    cmap : str
        The colormap used by ``matplotlib`` in the preview image.
        Default value is ``viridis``.
    data : obj
        The data used to generate the preview image.
    dq : obj
        The DQ data used to generate the preview image.
    file : str
        The filename to generate the preview image from.
    output_format : str
        The format to which the preview image is saved.  Options are
        ``jpg`` and ``thumb``
    preview_output_directory : str or None
        The output directory to which the preview image is saved.
    scaling : str
        The scaling used in the preview image.  Default is ``log``.
    thumbnail_output_directory : str or None
        The output directory to which the thumbnail is saved.

    Methods
    -------
    difference_image(data)
        Create a difference image from the data
    find_limits(data, pixmap, clipperc)
        Find the min and max signal levels after clipping by
        ``clipperc``
    get_data(filename, ext)
        Read in data from the given ``filename`` and ``ext``
    make_figure(image, integration_number, min_value, max_value, scale, maxsize, thumbnail)
        Create the ``matplotlib`` figure
    make_image(max_img_size)
        Main function
    save_image(fname, thumbnail)
        Save the figure
    """

    def __init__(self, filename, extension):
        """Initialize the class.

        Parameters
        ----------
        filename : str
            Name of fits file containing data
        extension : str
            Extension name to be read in
        """
        self.clip_percent = 0.01
        self.cmap = 'viridis'
        self.file = filename
        self.output_format = 'jpg'
        self.preview_output_directory = None
        self.scaling = 'log'
        self.thumbnail_output_directory = None
        self.preview_images = []
        self.thumbnail_images = []

        # Read in file
        self.data, self.dq = self.get_data(self.file, extension)

    def determine_map_file(self, header):
        """Determine which file contains the map of non-science pixels given a
        file header

        Parameters
        ----------
        header : astropy.io.fits.header
            Header object from an HDU object
        """
        if header['INSTRUME'] == 'MIRI':
            # MIRI imaging files use the external MIRI non-science map. Note that MIRI_CORONCAL and
            # MIRI_LYOT observations also have 'mirimage' in the filename. We deal with this in
            # crop_to_subarray()
            if 'CORONMSK' not in header:
                self.nonsci_map_file = (os.path.join(CONFIGS['outputs'], 'non_science_maps', 'mirimage_non_science_map.fits'))
            elif header['CORONMSK'] == '4QPM_1065':
                self.nonsci_map_file = (os.path.join(CONFIGS['outputs'], 'non_science_maps', 'miri4qpm_1065_non_science_map.fits'))
            elif header['CORONMSK'] == '4QPM_1140':
                self.nonsci_map_file = (os.path.join(CONFIGS['outputs'], 'non_science_maps', 'miri4qpm_1140_non_science_map.fits'))
            elif header['CORONMSK'] == '4QPM_1550':
                self.nonsci_map_file = (os.path.join(CONFIGS['outputs'], 'non_science_maps', 'miri4qpm_1550_non_science_map.fits'))
            elif header['CORONMSK'] in ['LYOT', 'LYOT_2300']:
                self.nonsci_map_file = (os.path.join(CONFIGS['outputs'], 'non_science_maps', 'mirilyot_non_science_map.fits'))

        elif header['INSTRUME'] == 'NIRSPEC':
            if 'NRSIRS2' in header['READPATT']:
                # IRS2 mode arrays are very different sizes between uncal and i2d files. For the uncal,
                # use the external non-science map. The i2d files we can treat like i2d files from the
                # other NIR detectors.
                if header['DETECTOR'] == 'NRS1':
                    self.nonsci_map_file = (os.path.join(CONFIGS['outputs'], 'non_science_maps', 'nrs1_irs2_non_science_map.fits'))
                elif header['DETECTOR'] == 'NRS2':
                    self.nonsci_map_file = (os.path.join(CONFIGS['outputs'], 'non_science_maps', 'nrs2_irs2_non_science_map.fits'))
            else:
                self.nonsci_map_file = None
        else:
            self.nonsci_map_file = None

    def difference_image(self, data):
        """
        Create a difference image from the data. Use last group minus
        first group in order to maximize signal to noise. With 4D
        input, make a separate difference image for each integration.

        Parameters
        ----------
        data : obj
            4D ``numpy`` ``ndarray`` array of floats

        Returns
        -------
        result : obj
            3D ``numpy`` ``ndarray`` containing the difference image(s)
            from the input exposure
        """
        return data[:, -1, :, :] - data[:, 0, :, :]

    def find_limits(self, data):
        """
        Find the minimum and maximum signal levels after clipping the
        top and bottom ``clipperc`` of the pixels.

        Parameters
        ----------
        data : obj
            2D numpy ndarray of floats

        Returns
        -------
        results : tuple
            Tuple of floats, minimum and maximum signal levels
        """
        # Ignore any pixels that are NaN
        finite = np.isfinite(data)

        # If all pixels are NaN then we're sunk. Scale
        # from 0 to 1.
        if not np.any(finite):
            logging.info('No pixels with finite signal. Scaling from 0 to 1')
            return (0., 1.)

        # Combine maps of science pixels and finite pixels
        # self.dq can have values of 1 (science pixel) or 0 (non-science pixel)
        pixmap = (self.dq & finite > 0)

        # If all non-science pixels are NaN then we're sunk. Scale
        # from 0 to 1.
        if not np.any(pixmap):
            logging.info('No good science pixels with finite signal. Scaling from 0 to 1')
            return (0., 1.)

        sorted_pix = np.sort(data[pixmap], axis=None)

        # Determine how many pixels to clip off of the high and low ends
        nelem = np.sum(pixmap)
        numclip = np.int32(self.clip_percent * nelem)

        # Determine min and max scaling levels
        minval = sorted_pix[numclip]
        maxval = sorted_pix[-numclip - 1]
        return (minval, maxval)

    def get_data(self, filename, ext):
        """
        Read in the data from the given file and extension.  Also find
        how many rows/cols of reference pixels are present.

        Parameters
        ----------
        filename : str
            Name of fits file containing data
        ext : str
            Extension name to be read in

        Returns
        -------
        data : obj
            Science data from file. A 2-, 3-, or 4D numpy ndarray
        dq : obj
            2D ``ndarray`` boolean map of reference pixels. Science
            pixels flagged as ``True`` and non-science pixels are
            ``False``
        """
        if os.path.isfile(filename):
            extnames = []
            with fits.open(filename) as hdulist:
                for exten in hdulist:
                    try:
                        extnames.append(exten.header['EXTNAME'])
                    except KeyError:
                        pass
                if ext in extnames:
                    dimensions = len(hdulist[ext].data.shape)
                    if dimensions == 4:
                        data = hdulist[ext].data[:, [0, -1], :, :].astype(float)
                    else:
                        data = hdulist[ext].data.astype(float)
                    yd, xd = data.shape[-2:]
                    try:
                        self.units = f"{hdulist[ext].header['BUNIT']}  "
                    except KeyError:
                        self.units = ''
                else:
                    raise ValueError('WARNING: no {} extension in {}!'.format(ext, filename))

                # For files that have no DQ extension, we get a map of the non-science
                # pixels from a dedicated map file. Getting this info from the DQ extension
                # doesn't work for uncal and i2d files, nor MIRI rate files.
                self.determine_map_file(hdulist[0].header)

                if (('uncal' in filename) or ('i2d' in filename)):
                    # uncal files have no DQ extensions, so we can't get a map of non-science pixels from the
                    # data itself.
                    if 'miri' in filename:
                        if 'mirimage' in filename:
                            dq = self.nonsci_from_file()
                            dq = crop_to_subarray(dq, hdulist[0].header, xd, yd)
                            dq = expand_for_i2d(dq, xd, yd)
                        else:
                            # For MIRI MRS/LRS data, we don't worry about non-science pixels, so create a map where all
                            # pixels are good.
                            dq = np.ones((yd, xd), dtype="bool").astype(bool)
                    elif 'nrs' in filename:
                        if 'NRSIRS2' in hdulist[0].header['READPATT']:
                            # IRS2 mode arrays are very different sizes between uncal and i2d files. For the uncal,
                            # use the external non-science map. The i2d files we can treat like i2d files from the
                            # other NIR detectors.
                            if 'uncal' in filename:
                                dq = self.nonsci_from_file()
                                # ISR2 data are always full frame, so no need to crop to subarray
                                # and since we are guaranteed to have an uncal file, no need to expand for i2d
                            elif 'i2d' in filename:
                                dq = create_nir_nonsci_map()
                                dq = crop_to_subarray(dq, hdulist[0].header, xd, yd)
                                dq = expand_for_i2d(dq, xd, yd)
                        else:
                            # NIRSpec observations that do not use IRS2 use the "standard" NIR detector non-science map.
                            # i.e. 4 outer rows and columns are refernece pixels
                            dq = create_nir_nonsci_map()
                            dq = crop_to_subarray(dq, hdulist[0].header, xd, yd)
                            dq = expand_for_i2d(dq, xd, yd)
                    else:
                        # All NIRCam, NIRISS, and FGS observations also use the "standard" NIR detector non-science map.
                        dq = create_nir_nonsci_map()
                        dq = crop_to_subarray(dq, hdulist[0].header, xd, yd)
                        dq = expand_for_i2d(dq, xd, yd)
                elif 'rate' in filename:
                    # For rate/rateints images all we need to worry about is MIRI imaging files. For those we use
                    # the external non-science map, because the pipeline does not add the NON_SCIENCE flags
                    # to the MIRI DQ extensions until the data are flat fielded, which is after the rate
                    # files have been created.
                    if 'mirimage' in filename:
                        dq = self.nonsci_from_file()
                        dq = crop_to_subarray(dq, hdulist[0].header, xd, yd)
                        dq = expand_for_i2d(dq, xd, yd)
                    else:
                        # For everything other than MIRI imaging, we get the non-science map from the
                        # DQ array in the file.
                        dq = self.get_nonsci_map(hdulist, extnames, xd, yd)
                else:
                    # For all file suffixes other than uncal and rate/rateints, we get the non-science map
                    # from the DQ array in the file.
                    dq = self.get_nonsci_map(hdulist, extnames, xd, yd)

                # Collect information on aperture location within the
                # full detector. This is needed for mosaicking NIRCam
                # detectors later.
                try:
                    self.xstart = hdulist[0].header['SUBSTRT1']
                    self.ystart = hdulist[0].header['SUBSTRT2']
                    self.xlen = hdulist[0].header['SUBSIZE1']
                    self.ylen = hdulist[0].header['SUBSIZE2']
                except KeyError:
                    logging.warning('SUBSTR and SUBSIZE header keywords not found')

        else:
            raise FileNotFoundError('WARNING: {} does not exist!'.format(filename))

        if dq.shape != data.shape[-2:]:
            raise ValueError(f'DQ array does not have the same shape as the data in {filename}')

        # In some cases (e.g. MIRI suabrray TA files) all pixels will be flagged as non-science.
        # In cases where dq shows all non-science pixels, let's zero out the flags and use all
        # the pixels for image scaling.
        if np.sum(dq) == 0:
            dq = np.ones(dq.shape, dtype=int)

        return data, dq

    def get_nonsci_map(self, hdulist, extensions, xdim, ydim):
        """Create a map of non-science pixels for a given HDUList. If there is no DQ
        extension in the HDUList, assume all pixels are science pixels.

        Parameters
        ----------
        hdulist : astropy.io.fits.HDUList
            HDUList object from a fits file

        extensions : list
            List of extension names in the HDUList

        xdim : int
            Number of columns in data array. Only used if there is no DQ extension

        ydim : int
            Number of rows in the data array. Only used if there is no DQ extension

        Returns
        -------
        dq : numpy.ndarray
            2D boolean array giving locations of non-science pixels
        """
        if 'DQ' in extensions:
            dq = hdulist['DQ'].data

            # For files with multiple integrations (rateints, calints), chop down the
            # DQ array to a single frame, since the non-science pixels will be the same
            # in all integrations
            if len(dq.shape) == 3:
                dq = dq[0, :, :]
            elif len(dq.shape) == 4:
                dq = dq[0, 0, :, :]

            dq = (dq & (dqflags.pixel['NON_SCIENCE'] | dqflags.pixel['REFERENCE_PIXEL']) == 0)
        else:
            # If there is no DQ extension in the HDUList, then we create a dq map where we assume
            # that all of the pixels are science pixels
            dq = np.ones((ydim, xdim), dtype=bool)
        return dq

    def make_figure(self, image, integration_number, min_value, max_value,
                    scale, maxsize=8, thumbnail=False):
        """
        Create the matplotlib figure of the image

        Parameters
        ----------
        image : obj
            2D ``numpy`` ``ndarray`` of floats

        integration_number : int
            Integration number within exposure

        min_value : float
            Minimum value for display

        max_value : float
            Maximum value for display

        scale : str
            Image scaling (``log``, ``linear``)

        maxsize : int
            Size of the longest dimension of the output figure (inches)

        thumbnail : bool
            True to create a thumbnail image, False to create the full
            preview image

        Returns
        -------
        result : obj
            Matplotlib Figure object
        """

        # Check the input scaling
        if scale not in ['linear', 'log']:
            raise ValueError('WARNING: scaling option {} not supported.'.format(scale))

        # Set the figure size
        yd, xd = image.shape

        """
        ratio = yd / xd
        if xd >= yd:
            xsize = maxsize
            ysize = maxsize * ratio
        else:
            ysize = maxsize
            xsize = maxsize / ratio
        """

        # Use the data dimensions to determine the aspect ratio and figure size of the preview image
        aspect, colorbar_orient, figsize = \
            determine_figure_properties(xd, yd, threshold=self.threshold_for_nonsquare_pix,
                                        maxsize=self.maxsize)

        # Create figure and axis object
        if thumbnail:
            self.fig, ax = plt.subplots(figsize=(3, 3))
        else:
            self.fig, ax = plt.subplots(figsize=figsize)

        # Get color scale and tick values depending on the scaling
        if scale == 'log':
            # Shift data so everything is positive
            shiftdata = image - min_value + 1
            shiftmin = 1
            shiftmax = max_value - min_value + 1

            # Generate tick labels
            tickvals = np.logspace(np.log10(shiftmin), np.log10(shiftmax), 5)
            tlabelflt = tickvals + min_value - 1

            # Image object
            cax = ax.imshow(shiftdata,
                            norm=colors.LogNorm(vmin=shiftmin,
                                                vmax=shiftmax),
                            cmap=self.cmap,
                            aspect=aspect)

        elif scale == 'linear':
            # Generate tick labels
            tickvals = np.linspace(min_value, max_value, 5)
            tlabelflt = tickvals
            cax = ax.imshow(image, clim=(min_value, max_value), cmap=self.cmap, aspect=aspect)

        # Invert y axis in all cases
        plt.gca().invert_yaxis()

        # For preview images, add colorbar, and create tick labels for it
        if not thumbnail:
            # Adjust the number of digits after the decimal point
            # in the colorbar labels based on the signal range
            delta = tlabelflt[-1] - tlabelflt[0]
            if delta >= 100:
                dig = 0
            elif ((delta < 100) & (delta >= 10)):
                dig = 1
            elif ((delta < 10) & (delta >= 1)):
                dig = 2
            elif delta < 1:
                dig = 3
            else:
                dig = 2
            format_string = "%.{}f".format(dig)
            tlabelstr = [format_string % number for number in tlabelflt]

            xyratio = xsize / ysize
            if xyratio < 1.6:
                # For apertures that are taller than they are wide, square, or that are wider than
                # they are tall but still reasonably close to square, put the colorbar on the right
                # side of the image.

                # Some magic numbers arrived at through testing aspect ratios for all apertures
                if xyratio > 0.4:
                    cb_width = 0.05
                else:
                    cb_width = 0.05 * 0.4 / xyratio

                upper_x_anchor = 0.02
                if xyratio < 0.1:
                    upper_x_anchor = 0.12

                cbax = self.fig.add_axes([ax.get_position().x1 + upper_x_anchor,
                                          ax.get_position().y0,
                                          cb_width,
                                          ax.get_position().height
                                          ])
                cbar = self.fig.colorbar(cax, cax=cbax, orientation='vertical', ticks=tickvals)

                cbar.ax.yaxis.minorticks_off()
                cbar.ax.set_yticklabels(tlabelstr)
                cbar.ax.set_ylabel(self.units, labelpad=7, rotation=270)
            else:
                # For apertures that are significantly wider than they are tall, put the colorbar
                # under the image.

                # Again, some magic numbers controlling the positioning and height of the
                # colorbar, based on testing.
                lower_y_anchor = 0. - (xyratio / 14.5)
                cb_height = 0.07 * (np.log2(xyratio) - 1)

                cbax = self.fig.add_axes([ax.get_position().x0,
                                          ax.get_position().y0 + lower_y_anchor,
                                          ax.get_position().width,
                                          cb_height])
                cbar = self.fig.colorbar(cax, cax=cbax, ticks=tickvals, orientation='horizontal')

                cbar.ax.xaxis.minorticks_off()
                cbar.ax.set_xticklabels(tlabelstr)
                cbar.ax.set_xlabel(self.units, labelpad=7, rotation=0)

            # Set text sizes
            ax.set_xlabel('Pixels', fontsize=maxsize * 5. / 4)
            ax.set_ylabel('Pixels', fontsize=maxsize * 5. / 4)
            ax.tick_params(labelsize=maxsize)
            plt.rcParams.update({'axes.titlesize': 'small'})
            plt.rcParams.update({'font.size': maxsize * 5. / 4})
            plt.rcParams.update({'axes.labelsize': maxsize * 5. / 4})
            plt.rcParams.update({'ytick.labelsize': maxsize * 5. / 4})
            plt.rcParams.update({'xtick.labelsize': maxsize * 5. / 4})

        elif thumbnail:
            # If creating a thumbnail, make the axes invisible
            plt.axis('off')
            cax.axes.get_xaxis().set_visible(False)
            cax.axes.get_yaxis().set_visible(False)

        # If preview image, set a title
        if not thumbnail:
            filename = os.path.split(self.file)[-1]
            ax.set_title(filename + ' Int: {}'.format(int(integration_number)))

    def make_image(self, max_img_size=8.0, create_thumbnail=False):
        """The main function of the ``PreviewImage`` class.

        Parameters
        ----------
        max_img_size : float
            Image size in the largest dimension

        create_thumbnail : bool
            If True, a thumbnail image is created and saved.
        """

        shape = self.data.shape

        if len(shape) == 4:
            # Create difference image(s)
            diff_img = self.difference_image(self.data)
        elif len(shape) < 4:
            diff_img = self.data

        # If there are multiple integrations in the file,
        # work on one integration at a time from here onwards
        ndim = len(diff_img.shape)
        if ndim == 2:
            diff_img = np.expand_dims(diff_img, axis=0)
        nint, ny, nx = diff_img.shape

        # If there are 10 integrations or less, make image for every integration
        # If there are more than 10 integrations, then make image for every 10th integration
        # If there are more than 100 integrations, then make image for every 100th integration
        if nint <= 10:
            integration_range = range(nint)
        elif 11 <= nint <= 100:
            integration_range = range(0, nint, 10)
        else:
            integration_range = range(0, nint, 100)

        for i in integration_range:
            frame = diff_img[i, :, :]

            # Find signal limits for the display
            minval, maxval = self.find_limits(frame)

            # Set NaN values to zero, so that those pixels
            # do not appear as big white splotches in the jpgs
            # after matplotlib downsamples/averages
            frame = nan_to_zero(frame)

            # Create preview image matplotlib object
            indir, infile = os.path.split(self.file)
            suffix = '_integ{}.{}'.format(i, self.output_format)
            if self.preview_output_directory is None:
                outdir = indir
            else:
                outdir = self.preview_output_directory
            outfile = os.path.join(outdir, infile.split('.')[0] + suffix)
            self.make_figure(frame, i, minval, maxval, self.scaling.lower(),
                             maxsize=max_img_size, thumbnail=False)
            self.save_image(outfile, thumbnail=False)
            plt.close(self.fig)
            self.preview_images.append(outfile)

            # Create thumbnail image matplotlib object, only for the
            # first integration
            if i == 0 and create_thumbnail:
                if self.thumbnail_output_directory is None:
                    outdir = indir
                else:
                    outdir = self.thumbnail_output_directory
                outfile = os.path.join(outdir, infile.split('.')[0] + suffix)
                self.make_figure(frame, i, minval, maxval, self.scaling.lower(),
                                 maxsize=max_img_size, thumbnail=True)
                self.save_image(outfile, thumbnail=True)
                plt.close(self.fig)
                self.thumbnail_images.append(self.thumbnail_filename)

    def nonsci_from_file(self):
        """Read in a map of non-science/reference pixels from a fits file

        Parameters
        ----------
        filename : str
            Name of fits file to be read in.

        Returns
        -------
        map : numpy.ndarray
            2D boolean array of pixel values
        """
        map = fits.getdata(self.nonsci_map_file)
        return map.astype(bool)

    def save_image(self, fname, thumbnail=False):
        """
        Save an image in the requested output format and sets the
        appropriate permissions

        Parameters
        ----------
        image : obj
            A ``matplotlib`` figure object

        fname : str
            Output filename

        thumbnail : bool
            True if saving a thumbnail image, false for the full
            preview image.
        """
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)
        permissions.set_permissions(fname)

        # If the image is a thumbnail, rename to '.thumb'
        if thumbnail:
            self.thumbnail_filename = fname.replace('.jpg', '.thumb')
            os.rename(fname, self.thumbnail_filename)
            logging.info('\tSaved image to {}'.format(self.thumbnail_filename))
        else:
            logging.info('\tSaved image to {}'.format(fname))
            self.thumbnail_filename = None

    def set_scaling(self):
        """Determine the scaling (e.g. log, linear) to use for the preview image.
        NIRSpec WATA TA images and non-full-frame MIRI target acq images are set to linear,
        while everything else is set to log.
        """
        header = fits.getheader(self.file)
        self.scaling = 'log'
        if ((header['EXP_TYPE'] == 'NRS_WATA') & (header['SUBSIZE1'] == 32) & (header['SUBSIZE2'] == 32)):
            self.scaling = 'linear'
        if ((header['EXP_TYPE'] == 'MIR_TACQ') & (header['SUBARRAY'] != 'FULL')):
            self.scaling = 'linear'


def create_nir_nonsci_map():
    """Create a map of non-science pixels for a near-IR detector

    Returns
    -------
    arr : numpy.ndarray
        2D boolean array. Science pixels have a value of 1 and non-science pixels
        (reference pixels) have a value of 0.
    """
    arr = np.ones((2048, 2048), dtype=int)
    arr[0:4, :] = 0
    arr[:, 0:4] = 0
    arr[2044:, :] = 0
    arr[:, 2044:] = 0
    return arr.astype(bool)


def crop_to_subarray(arr, header, xdim, ydim):
    """Given a full frame array, along with a fits HDU header containing subarray
    information, crop the array down to the indicated subarray.

    Parameters
    ----------
    arr : numpy.ndarray
        2D array of data. Assumed to be full frame (2048 x 2048)

    header : astropy.io.fits.header
        Header from a single extension of a fits file

    xdim : int
        Number of columns in the corresponding data (not dq) array, in pixels

    ydim : int
        Number of rows in the corresponding data (not dq) array, in pixels

    Returns
    -------
    arr : numpy.ndarray
        arr, cropped down to the size specified in the header
    """
    # Pixel coordinates in the headers are 1-indexed. Subtract 1 to get them into
    # python's 0-indexed system
    try:
        xstart = header['SUBSTRT1'] - 1
        xlen = header['SUBSIZE1']
        ystart = header['SUBSTRT2'] - 1
        ylen = header['SUBSIZE2']
    except KeyError:
        # If subarray info is missing from the header, then we don't know which
        # part of the dq array to extract. Rather than raising an exception, let's
        # extract a portion of the dq array that is centered on the full frame
        # array, so that we can still create a preview image later.
        logging.info(f"No subarray location information in {header['FILENAME']}. Extracting a portion of the DQ array centered on the full frame.")
        arr_ydim, arr_xdim = arr.shape
        ystart = (arr_ydim // 2) - (ydim // 2)
        xstart = (arr_xdim // 2) - (xdim // 2)
        xlen = xdim
        ylen = ydim
    return arr[ystart: (ystart + ylen), xstart: (xstart + xlen)]


def expand_for_i2d(array, xdim, ydim):
    """Some file types, like i2d files, contain arrays with sizes that are different than
    those specified in the SUBSIZE header keywords. In those cases, we need to expand the
    input array from the official size to the actual size.

    Parameters
    ----------
    array : numpy.ndarray
        2D DQ array of booleans

    xdim : int
        Number of columns in the data whose dimensions we want ``array`` to have.
        (e.g. the dimensions of the i2d file)

    ydim : int
        Number of rows in the data whose dimensions we want ``array`` to have.
        (e.g. the dimensions of the i2d file)

    Returns
    -------
    new_array : numpy.ndarray
        2D array with dimensions of (ydim x xdim)
    """
    ydim_array, xdim_array = array.shape
    if ((ydim_array != ydim) or (xdim_array != xdim)):
        if (ydim_array != ydim):
            new_array_y = np.zeros((ydim, xdim_array), dtype=bool)  # Added rows/cols will be all zeros
            y_offset = abs((ydim - ydim_array) // 2)
            if (ydim_array < ydim):
                new_array_y[y_offset: (y_offset + ydim_array), :] = array
            elif (ydim_array > ydim):
                new_array_y = array[y_offset: (y_offset + ydim), :]
        else:
            new_array_y = array
        if (xdim_array != xdim):
            new_array_x = np.zeros((ydim, xdim), dtype=bool)  # Added rows/cols will be all zeros
            x_offset = abs((xdim - xdim_array) // 2)
            if (xdim_array < xdim):
                new_array_x[:, x_offset: (x_offset + xdim_array)] = new_array_y
            elif (xdim_array > xdim):
                new_array_x = new_array_y[:, x_offset: (x_offset + xdim)]
        else:
            new_array_x = new_array_y
        return new_array_x
    else:
        return array


def nan_to_zero(image):
    """Set any pixels with a value of NaN to zero

    Parameters
    ----------
    image : numpy.ndarray
        Array from which NaNs will be removed

    Returns
    -------
    image : numpy.ndarray
        Input array with NaNs changed to zero
    """
    nan = np.isnan(image)
    image[nan] = 0
    return image


class Level3PreviewImage():
    """An object for generating and saving a preview image from level 3 files.
    Used by``generate_preview_images``.
    """
    def __init__(self, filename, data_type=None, maxsize=8, preview_output_directory=None,
                 create_thumbnail=False, thumbnail_output_directory=None):
        self.filename = filename
        self.data_type = data_type
        self.maxsize = maxsize
        self.output_format = 'jpg'
        self.preview_output_directory = preview_output_directory
        self.thumbnail_output_directory = thumbnail_output_directory
        self.create_thumbnail = create_thumbnail
        self.preview_images = []
        self.thumbnail_images = []
        self.threshold_for_nonsquare_pix = 0.15
        self.figures = []
        self.wfss_source_ids = []

        # Read in the data
        self.get_data()

        # Fall back to the standard preview image and thumbnail output directory if a
        # directory is not given
        if preview_output_directory is None:
            self.preview_output_directory = os.path.join(CONFIGS["preview_image_filesystem"],
                                                         self.model.meta.filename[0:7])
        if thumbnail_output_directory is None:
            self.thumbnail_output_directory = os.path.join(CONFIGS["thumbnail_filesystem"],
                                                           self.model.meta.filename[0:7])


        print(self.filename)
        print(self.exp_type)


        if 'x1dints' in self.filename:
            # Time Series data from NIRCAM
            if self.exp_type == 'NRC_TSGRISM':
                print('nrc_tsgrism x1dints')
                self.tso_x1dints_plot()
            elif self.exp_type == 'MIR_LRS-SLITLESS':
                print('miri lrs slitless')
                # Make sure we support both old and new file formats
                #try:
                #    self.miri_lrs_slitless_x1dints_plot_old_format()
                #except IndexError:
                self.miri_lrs_slitless_x1dints_plot()
            elif self.exp_type == 'NIS_SOSS':
                print('niriss soss')
                self.niriss_soss_plot()
            elif self.exp_type == 'NRS_BRIGHTOBJ':
                print('nirspec bright obj')
                try:
                    self.miri_lrs_slitless_x1dints_plot_old_format()
                except IndexError:
                    self.miri_lrs_slitless_x1dints_plot()
        elif self.exp_type == 'NIS_AMI':
            if 'amilg' in self.filename:
                self.amilg_preview()
            elif 'ami-oi' in self.filename or 'aminorm-oi' in self.filename:
                self.ami_preview()
        elif 'whtlt.ecsv' in self.filename or 'phot.ecsv' in filename:
            self.tso_whitelight_curve()
        elif self.exp_type in ['NRS_FIXEDSLIT', 'MIR_LRS-FIXEDSLIT'] and ('cal' in self.filename or
                                                                          'crf' in self.filename or
                                                                          's2d' in self.filename):
            print('fixed slit 2d')
            self.fixed_slit_cal_crf_s2d()
        elif 'x1d' in self.filename and self.exp_type in ['NRS_FIXEDSLIT',
                                                          'NRS_IFU',
                                                          'MIR_LRS-FIXEDSLIT',
                                                          ]:
            print('fixed slit 1d')
            #self.miri_nirspec_fixed_slit_or_ifu_x1d()
            self.miri_lrs_fixed_slit_nirspec_ifu_x1d()
        elif 'x1d' in self.filename and self.exp_type == ['MIRI_MRS']:
            self.miri_mrs_x1d()
        elif ('x1d' in self.filename or 'c1d' in self.filename) and self.exp_type in ['NRC_WFSS', 'NIS_WFSS']:
            self.wfss_x1d()
        elif 's3d' in self.filename and self.exp_type in ['NRS_IFU', 'MIR_MRS']:
            print('ifu 3d')
            self.nirspec_miri_ifu_s3d()
        elif 'i2d' in self.filename:
            self.i2d_file()
        elif ('psfstack' in self.filename or 'psfalign' in self.filename or 'psfsub' in self.filename):
            self.psfstack_file()

        self.save_figures()

    def ami_preview(self):
        """Given a AmiOIModel datamodel instance, create a preview image. This function was
        taken from the jwst-pipeline notebooks repo
        """
        # Read the observables from the datamodel
        # Squared visibilities and uncertainties
        vis2 = self.model.vis2["VIS2DATA"]
        vis2_err = self.model.vis2["VIS2ERR"]

        # Closure phases and uncertainties
        t3phi = self.model.t3["T3PHI"]
        t3phi_err = self.model.t3["T3PHIERR"]

        # Construct baselines between the U and V coordinates of sub-apertures
        baselines = (self.model.vis2['UCOORD']**2 + self.model.vis2['VCOORD']**2)**0.5

        # Construct baselines between combinations of three sub-apertures
        u1 = self.model.t3['U1COORD']
        u2 = self.model.t3['U2COORD']
        v1 = self.model.t3['V1COORD']
        v2 = self.model.t3['V2COORD']
        u3 = -(u1 + u2)
        v3 = -(v1 + v2)
        baselines_t3 = []
        for k in range(len(u1)):
            B1 = np.sqrt(u1[k]**2 + v1[k]**2)
            B2 = np.sqrt(u2[k]**2 + v2[k]**2)
            B3 = np.sqrt(u3[k]**2 + v3[k]**2)

            # Use longest baseline of the three for plotting
            baselines_t3.append(np.max([B1, B2, B3]))
        baselines_t3 = np.array(baselines_t3)

        # Plot closure phases, squared visibilities against their baselines
        self.fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(8, 16))
        ax1.errorbar(baselines_t3, t3phi, yerr=t3phi_err, fmt="go")
        ax2.errorbar(baselines, vis2, yerr=vis2_err, fmt="go")
        ax1.set_xlabel(r"$B_{max}$ [m]", size=12)
        ax1.set_ylabel("Closure phase [deg]", size=12)
        ax1.set_title("Closure Phase", size=14)
        ax2.set_title("Squared Visibility", size=14)
        ax2.set_xlabel(r"$B_{max}$ [m]", size=12)
        ax2.set_ylabel("Squared Visibility", size=12)
        plt.suptitle(self.model.meta.filename, fontsize=16)
        ax1.set_ylim([-3.5, 3.5])
        ax2.set_ylim([0.5, 1.1])
        self.figures.append(self.fig)

    def amilg_preview(self):
        """Plot data, model, and residual from amilg.fits file
        """
        # Define colormap that shows NaNs as black
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='black')

        # Plot the data, model, residual
        norm = ImageNormalize(self.model.norm_centered_image[0],
                              interval=MinMaxInterval(),
                              stretch=SqrtStretch())
        fig, axs = plt.subplots(1, 3, figsize=(12, 5))
        axs[0].set_title('Normalized Data')
        im1 = axs[0].imshow(self.model.norm_centered_image[0], norm=norm, cmap=cmap)
        axs[1].set_title('Normalized Model')
        im2 = axs[1].imshow(self.model.norm_fit_image[0], norm=norm, cmap=cmap)
        axs[2].set_title('Normalized Residual (Data-Model)')
        im3 = axs[2].imshow(self.model.norm_resid_image[0], cmap=cmap)

        for im in [im1, im2, im3]:
            plt.colorbar(im, shrink=.95, location='bottom', pad=.05)
        for ax in axs:
            ax.axis('off')

        plt.suptitle(os.path.basename(firstfile))
        plt.tight_layout()
        self.figures.append(self.fig)

    def create_plot(self):
        """Create the 1D plot"""
        self.get_limits()

        # Create figure and axis object
        if thumbnail:
            self.fig, ax = plt.subplots(figsize=(3, 3))
        else:
            self.fig, ax = plt.subplots(figsize=(self.maxsize, self.maxsize))

        ax.plot(self.wavelength, self.signal, color='black')
        ax.set_xlabel(f'Wavelength ({self.wave_units})')
        ax.set_ylabel(f'Signal ({self.signal_units})')
        ax.set_xlim(self.min_wave, self.max_wave)
        ax.set_ylim(self.min_yval, self.max_yval)
        ax.set_title(os.path.filename(self.filename))
        plt.rcParams.update({'axes.titlesize': 'small'})
        plt.rcParams.update({'font.size': maxsize * 5. / 4})
        plt.rcParams.update({'axes.labelsize': maxsize * 5. / 4})
        plt.rcParams.update({'ytick.labelsize': maxsize * 5. / 4})
        plt.rcParams.update({'xtick.labelsize': maxsize * 5. / 4})

    def tso_x1dints_plot(self):
        """For TSO x1dints files, we plot the spectrum prior to any eclipse, on top
        of the spectrum during the eclipse. We assume that the first integraion occurs
        outside of any eclipse. Then find the spectrum with the overall minimum average,
        and assume that this is during the eclipse.
        """
        baseline = self.model.spec[0].spec_table.FLUX[1, :]
        baseline_waves = self.model.spec[0].spec_table.WAVELENGTH[1, :]
        minidx = 1
        minext = 0
        minmean = np.nanmean(baseline)

        n_extensions = len(self.model.spec)

        # Loop over spec extensions
        for ext in range(n_extensions):
            nints, npts = self.model.spec[ext].spec_table.FLUX.shape

            # Loop over integrations
            for i in range(nints):
                newmean = np.nanmean(self.model.spec[ext].spec_table.FLUX[i, :])
                if newmean < minmean:
                    minmean = newmean
                    minidx = i
                    minext = ext

        eclipse = self.model.spec[minext].spec_table.FLUX[minidx, :]
        eclipse_waves = self.model.spec[minext].spec_table.WAVELENGTH[minidx, :]

        # Will we ever need to make a thumnbail from these data?
        self.fig, ax = plt.subplots(figsize=(self.maxsize, self.maxsize))
        ax.plot(baseline_waves, baseline, color='blue', label='Baseline', alpha=0.5)
        ax.plot(eclipse_waves, eclipse, color='red', label='Lowest mean', alpha=0.5)
        ax.set_xlabel(f'Wavelength ({self.wavelength_units})')
        ax.set_ylabel(f'Flux ({self.flux_units})')
        ax.set_title(f'{self.model.meta.filename}: {self.model.meta.target.catalog_name}', fontsize=12)
        ax.legend(loc='upper right')
        self.figures.append(self.fig)

    def miri_lrs_slitless_x1dints_plot_old_format(self):
        """Create preview image for MIRI LRS-SLITLESS x1dints data that uses the old data
        format, pre-jwst-1.19.0
        """
        nints = len(self.model.spec)

        baseline_integ = 1
        baseline_flux = self.model.spec[baseline_integ].spec_table.FLUX
        baseline_waves = self.model.spec[baseline_integ].spec_table.WAVELENGTH

        # Find extension with the lowest mean flux
        low_med = 999999
        for integ in range(2, nints, 5):
            med = np.nanmedian(self.model.spec[integ].spec_table.FLUX)
            if med < low_med:
                low_med = med
                low_integ = integ
        low_flux = self.model.spec[low_integ].spec_table.FLUX
        low_waves = self.model.spec[low_integ].spec_table.WAVELENGTH

        self.fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(self.maxsize, self.maxsize))
        ax[0].plot(baseline_waves, baseline_flux, alpha=0.5, label=f'Int {baseline_integ}')
        ax[0].plot(low_waves, low_flux, alpha=0.5, label=f'Int {low_integ}')
        ax[0].set_xlabel(f'Wavelength ({self.wavelength_units})')
        ax[0].set_ylabel(f'Flux ({self.flux_units})')
        ax[0].legend()
        ax[0].set_title(self.model.meta.filename)

        # Pick several wavelengths and plot the corresponding fluxes over time
        fin = np.where(np.isfinite(baseline_flux))[0]
        minfin = np.min(fin)
        numfin = len(fin)
        wavelength = self.model.spec[0].spec_table.WAVELENGTH
        wave_idxs = [int(minfin + 0.25*i*numfin) for i in [1,2,3]]
        int_times = self.model.int_times['int_mid_MJD_UTC']
        time_units = self.model.int_times.columns['int_mid_MJD_UTC'].unit
        if time_units is None:
            time_units = 'd'

        #wave_idxs = [100, 200, 350]
        #print(wave_idxs)
        #print(len(baseline_flux))
        for wave_idx in wave_idxs:
            #print(wave_idx, self.model.spec[0].spec_table.shape)
            fluxes = [self.model.spec[i].spec_table.FLUX[wave_idx] for i in range(nints)]
            ax[1].plot(int_times, fluxes, label="{:.3f} um".format(wavelength[wave_idx]))
        ax[1].set_xlabel(f'MJD_UTC ({time_units})')
        ax[1].set_ylabel(f'Flux ({self.flux_units})')
        ax[1].legend()

        self.figures = [self.fig]

    def miri_lrs_slitless_x1dints_plot(self):
        """Create preview image for MIRI LRS-SLITLESS x1dints data
        """
        #nints = self.model.spec[0].spec_table.WAVELENGTH.shape[0]
        nspec = len(self.model.spec)

        baseline_integ = 1
        baseline_flux = self.model.spec[0].spec_table.FLUX[baseline_integ, :]
        baseline_waves = self.model.spec[0].spec_table.WAVELENGTH[baseline_integ, :]

        # Find extension with the lowest mean flux
        low_spec = 0
        low_med = 999999
        low_total_integ = 0
        total_prev_integ = 0

        for s_idx in range(nspec):
            nints = self.model.spec[s_idx].spec_table.WAVELENGTH.shape[0]

            for integ in range(2, nints, 5):
                med = np.nanmedian(self.model.spec[s_idx].spec_table.FLUX[integ, :])
                if med < low_med:
                    low_med = med
                    low_integ = integ
                    low_spec = s_idx
                    low_total_integ = total_prev_integ + integ

            # Running total of the number of integrations prior to the current spec
            total_prev_integ += nints

        low_flux = self.model.spec[low_spec].spec_table.FLUX[low_integ, :]
        low_waves = self.model.spec[low_spec].spec_table.WAVELENGTH[low_integ, :]

        self.fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(self.maxsize, self.maxsize))
        ax[0].plot(baseline_waves, baseline_flux, alpha=0.75, label=f'Integ {baseline_integ}')
        ax[0].plot(low_waves, low_flux, alpha=0.75, label=f'Integ {low_total_integ}')
        ax[0].set_xlabel(f'Wavelength ({self.wavelength_units})')
        ax[0].set_ylabel(f'Flux ({self.flux_units})')
        ax[0].legend()
        ax[0].set_title(self.model.meta.filename)

        # Pick several wavelengths and plot the corresponding fluxes over time
        fin = np.where(np.isfinite(baseline_flux))[0]
        minfin = np.min(fin)
        numfin = len(fin)
        wavelength = self.model.spec[0].spec_table.WAVELENGTH[0]
        wave_idxs = [int(minfin + 0.25*i*numfin) for i in [1,2,3]]
        int_times = self.model.int_times['int_mid_MJD_UTC']
        time_units = self.model.int_times.columns['int_mid_MJD_UTC'].unit
        if time_units is None:
            time_units = 'd'

        for wave_idx in wave_idxs:

            all_flux = np.array([])
            for s_idx in range(nspec):
                all_flux = np.concatenate([all_flux, self.model.spec[s_idx].spec_table.FLUX[:, wave_idx]])

            ax[1].plot(int_times, all_flux, alpha=0.5, label="{:.3f} um".format(wavelength[wave_idx]))

        ax[1].set_xlabel(f'MJD_UTC ({time_units})')
        ax[1].set_ylabel(f'Flux ({self.flux_units})')
        ax[1].legend()

        self.figures = [self.fig]

    def niriss_soss_plot(self):
        """Create preview image for NIRISS SOSS x1dints data

        NOTE: This function works for both level 3 and level 2 x1dint files.
        """
        nints = len(self.model.int_times['int_start_MJD_UTC'])

        # Find which orders are captured in which extensions
        orders = {}
        int_info = {}
        for ext in range(len(self.model.spec)):
            order = str(self.model.spec[ext].spectral_order)
            if order in orders:
                orders[order].append(ext)
            else:
                orders[order] = [ext]

        self.fig, ax = plt.subplots(ncols=2, nrows=len(orders), figsize=(self.maxsize*2, 6 + 3*(len(orders) - 1)))

        #    integ = str(self.model.spec[ext].int_num)
        #    if integ in int_info:
        #        int_info[integ].append(ext)
        #    else:
        #        int_info[integ] = [ext]
        # int_info['1'] = [0, 1, 2]  int_num is key, value is list of extension numbers

        #for ext in range(len(self.model.spec)):
        #    order = self.model.spec[ext].spectral_order
        #    integ = self.model.spec[ext].int_num
        #    if str(integ) in int_info:
        #        int_info[str(integ)].append(ext)
        #    else:
        #        int_info[str(integ)] = [ext]

        # Integration times for all integrations
        int_times = self.model.int_times['int_mid_MJD_UTC']
        time_axis = (int_times - np.nanmean(int_times)) * 24.0
        time_units = 'hr'

        # Minor tweak to axis label if we have a level 2 seg x1dints file
        x_ax_str = 'exposure'
        if '-seg' in self.model.meta.filename:
            x_ax_str = 'segment'

        # Integration number we'll consider "baseline", and plot
        baseline_integ = 1

        # Find integration with lowest mean value in 1st order
        low_integ = 2
        low_spec_idx = orders['1'][0]
        low_med = 999999
        spec_ints = []
        for spec_idx in orders['1']:
            n_spec_ints = self.model.spec[spec_idx].spec_table.WAVELENGTH.shape[0]
            spec_ints.append(n_spec_ints)

            # Define which integration to start with. In the initial spec
            # object, start with the integration after the baseline
            min_int = 0
            if spec_idx == orders['1'][0]:
                min_int = baseline_integ + 1

            for integ in range(min_int, n_spec_ints):
                data = self.model.spec[spec_idx].spec_table['FLUX'][integ, :]
                med = np.nanmedian(data)
                if med < low_med:
                    low_integ = integ
                    low_spec_idx = spec_idx
                    low_med = med

        # Find the spec extension that contains the low data
        match = np.where(low_spec_idx == np.array(orders['1']))[0][0]

        # Calculate the integration number corresponding to the low data
        low_total_integ = np.sum(np.array(spec_ints)[0:match]) + low_integ

        # For each order, get the flux and wavelength values corresponding to the baseline
        for i, order in enumerate(orders):

            # Get the baseline data, assumed to be an integration prior to the start of
            # any eclipse/variation in signal.
            baseline_data = self.model.spec[orders[order][0]].spec_table['FLUX'][baseline_integ, :]
            baseline_waves = self.model.spec[orders[order][0]].spec_table['WAVELENGTH'][baseline_integ, :]

            # Get the data from the integration with the lowest median signal. We assume this
            # is during the exlipse.
            low_data = self.model.spec[orders[order][match]].spec_table['FLUX'][low_integ, :]
            low_waves = self.model.spec[orders[order][match]].spec_table['WAVELENGTH'][low_integ, :]

            ax[i, 0].plot(baseline_waves, baseline_data, alpha=0.5, label=f'Integ {baseline_integ}')
            ax[i, 0].plot(low_waves, low_data, alpha=0.5, linestyle='dashed', label=f'Integ {low_total_integ}')
            if i == 0:
                ax[i, 0].set_title(f'{self.model.meta.filename}, Order {order}')
            else:
                ax[i, 0].set_title(f'Order {order}')
            ax[i, 0].set_xlabel(f'Wavelength ({self.wavelength_units})')
            ax[i, 0].set_ylabel(f'Flux ({self.flux_units})')
            ax[i, 0].legend()

            # Select three different wavelengths from each order, and get the signal through
            # time at those wavelengths
            fin = np.where(np.isfinite(baseline_data))[0]
            minfin = np.min(fin)
            numfin = len(fin)
            #wavelength = self.model.spec[orders[order][0]].spec_table.WAVELENGTH[baseline_integ, 0]
            wave_idxs = [int(minfin + 0.25*i*numfin) for i in [1,2,3]]

            waves = []
            for i_wave, wave_idx in enumerate(wave_idxs):

                # Create a list of wavelengths, for plot labeling
                waves.append(baseline_waves[wave_idx])

                fluxes_v_time = []
                for i_spec, spec_idx in enumerate(orders[order]):

                        # Get the data for the expected wavelength
                        fluxes_v_time.extend(self.model.spec[spec_idx].spec_table['FLUX'][:, wave_idx])

                ax[i, 1].plot(time_axis, fluxes_v_time, alpha=0.5, label="{:.3f} um".format(waves[i_wave]))
            ax[i, 1].set_xlabel(f'Time since mid-time of {x_ax_str} ({time_units})')
            ax[i, 1].set_ylabel(f'Flux ({self.flux_units})')
            ax[i, 1].set_title(f'Order {order}')
            ax[i, 1].legend()

        plt.tight_layout()
        self.figures = [self.fig]

    def niriss_soss_plot_old_format(self):
        """Create preview image for NIRISS SOSS x1dints data
        """
        nints = len(self.model.int_times['int_start_MJD_UTC'])

        # Find which orders are captured in which extensions
        orders = {}
        int_info = {}
        for ext in range(len(self.model.spec)):
            order = self.model.spec[ext].spectral_order
            if order in orders:
                orders[str(order)].append(ext)
            else:
                orders[str(order)] = [ext]

            integ = self.model.spec[ext].int_num
            if str(integ) in int_info:
                int_info[str(integ)].append(ext)
            else:
                int_info[str(integ)] = [ext]

        baseline_integ = '1'
        baseline_order_idx = int_info[baseline_integ]
        baseline_data = [self.model.spec[e].spec_table.FLUX for e in baseline_order_idx]
        baseline_waves = [self.model.spec[e].spec_table.WAVELENGTH for e in baseline_order_idx]
        baseline_orders = [self.model.spec[e].spectral_order for e in baseline_order_idx]

        # Find integration with lowest mean value
        low_integ = '2'
        low_med = 999999
        for integ in range(2, nints):
            idx = int_info[str(integ)][0]
            med = np.nanmedian(self.model.spec[idx].spec_table.FLUX)
            if med < low_med:
                low_integ = str(integ)

        low_order_idx = int_info[low_integ]
        low_data = [self.model.spec[e].spec_table.FLUX for e in low_order_idx]
        low_waves = [self.model.spec[e].spec_table.WAVELENGTH for e in low_order_idx]
        low_orders = [self.model.spec[e].spectral_order for e in low_order_idx]

        self.fig, ax = plt.subplots(figsize=(self.maxsize, self.maxsize))
        for wave, data, order in zip(baseline_waves, baseline_data, baseline_orders):
            ax.plot(wave, data, alpha=0.5, label=f'Int {baseline_integ}, order {order}')
        for wave, data, order in zip(low_waves, low_data, low_orders):
            ax.plot(wave, data, alpha=0.5, linestyle='dashed', label=f'Int {low_integ}, order {order}')
        ax.set_xlabel(self.wavelength_units)
        ax.set_ylabel(self.flux_units)
        ax.legend()

        self.figures = [self.fig]


    def get_data(self):

        if '.fits' in self.filename:
            self.model = datamodels.open(self.filename)
            self.exp_type = self.model.meta.exposure.type

            #self.num_ext = len(self.model.spec) #- this may not be used
            try:
                self.wavelength_units = self.model.spec[0].spec_table.columns["wavelength"].unit
                self.flux_units = self.model.spec[0].spec_table.columns["flux"].unit
            except AttributeError:
                self.wavelength_units = None
                self.flux_units = None

            # I think everything in JWST is in Microns
            if self.wavelength_units is None:
                self.wavelength_units = 'Microns'

        elif 'whtlt.ecsv' in self.filename or 'phot.ecsv' in self.filename:
            # Top (usually) 15 lines contain metadata
            self.metadata = {'exp_type': 'None', 'filter': None, 'pupil': None,
                             'subarray': None, 'number_of_integrations': None,
                             'target_name': None}
            data_start_line = 0
            with open(self.filename, 'r') as f:
                for i, line in enumerate(f):
                    if line.strip().startswith('#'):
                        for key in self.metadata:
                            if key in line:
                                try:
                                    colon_loc = line.rindex(':')
                                    brace_loc = line.rindex('}')
                                    self.metadata[key] = line[colon_loc+2:brace_loc]
                                except ValueError:
                                    pass
                        # phot.ecsv files have units in the metadata
                        if 'net_aperture_sum' in line:
                            strt = line.rfind('unit: ')
                            ending = line.rfind(', datatype')
                            self.flux_units = line[strt+6: ending]
                    else:
                        data_start_line = i
                        break
            self.model = pd.read_csv(self.filename, header=data_start_line, delimiter=' ')
            self.num_ext = 1
            self.exp_type = self.metadata['exp_type']
        """
        # This is based on a nirspec fixed slit file
        # jw06550001001_03104_00003_nrs2_x1d.fits (MultiSpecModel)
        for ext in range(num_ext):
            self.wavelength = model.spec[ext].spec_table.WAVELENGTH
            self.signal = model.spec[ext].spec_table.FLUX


        # MIRI MRS jw01522091001_03101_00002_mirifushort_x1d.fits
        # has len(m.spec) of 1. Only a single 1d spectrum. So same commands as above

        # Also MIRI MRS jw01033001001_03103_00002_mirimage_x1d.fits
        # Same. Only 1 1d spectrum
        """




    def get_limits(self):
        """Determine the plot limits based on characteristics of the data.
        Let's just ignore the shortest and longest X% wavelengths and use
        the max and min of the rest? Would be nice to be able to ignore band
        edge effects, where the signal often jumps up.
        """
        ignore = 0.10  # 10%
        finite = np.where(np.isfinite(self.signal))[0]
        self.min_wave = np.min(finite)
        self.max_wave = np.max(finite)
        num_ignore = int((self.max_wave - self.min_wave) * ignore)
        min_idx = min_wave + num_ignore
        max_idx = max_wave - num_ignore

        self.max_yval = np.nanmax(self.signal[min_idx: max_idx])
        self.min_yval = np.nanmin(self.signal[min_idx: max_idx])

    def save_figures(self):
        """
        Save matplotlib figures as preview images and possibly thumbnail images,
        and set the appropriate permissions
        """
        for i, figure in enumerate(self.figures):

            # Preview image filename is the name of the input file with jpg at the end
            #fname = os.path.join(self.preview_output_directory, self.model.meta.filename.split('.')[0] + '.jpg')
            fname = os.path.basename(self.filename).split('.')[0] + '.jpg'

            # If we are dealing with WFSS data, where we have multiple figures, use the source
            # ID values to create unique jpg filenames.
            if len(self.wfss_source_ids) > 0:
                fname = fname.replace('.jpg', f'_source{self.wfss_source_ids[i]}.jpg')

            # Save preview image
            self.preview_images.append(fname)
            img_type = 'preview'
            outname = os.path.join(self.preview_output_directory, fname)
            figure.savefig(outname, bbox_inches='tight', pad_inches=0.05)
            permissions.set_permissions(outname)
            logging.info(f'\tSaved {img_type} image to {outname}')
            print(f'\tSaved {img_type} image to {outname}')

        # If we're saving a thumbnail, change the output file name and make the
        # figure smaller. Make axes and labels invisible. In cases where there are
        # mulitple preview images, and thumbnails are to be made, only create a
        # thumbnail for the first one.
        if self.create_thumbnail:
            figure = self.figures[0]
            figure = hide_axes(figure)
            orig_width, orig_height = figure.get_size_inches()
            figure.set_size_inches(3, 3, forward=True)
            img_type = 'thumbnail'

            fname = os.path.basename(self.filename).split('.')[0] + '.jpg'
            if len(self.wfss_source_ids) > 0:
                fname = fname.replace('.jpg', f'_source{self.wfss_source_ids[0]}.jpg')

            outname = os.path.join(self.thumbnail_output_directory, fname)
            figure.savefig(outname, bbox_inches='tight', pad_inches=0)
            permissions.set_permissions(outname)

            thumb_outname = outname.replace('.jpg', '.thumb')
            os.rename(outname, thumb_outname)
            self.thumbnail_images.append(os.path.basename(thumb_outname))
            logging.info(f'\tSaved {img_type} image to {thumb_outname}')
            print(f'\tSaved {img_type} image to {thumb_outname}')

        # Close the figures
        for figure in self.figures:
            plt.close(figure)

    def i2d_file(self):
        """Create a preview image from an i2d file.
        """
        vmin = np.nanpercentile(self.model.data, 1)
        vmax = np.nanpercentile(self.model.data, 99)

        # Use to determine linearly scaled signals
        #clipped = sigma_clip_ignore_nan(self.model.data)

        # Find sigma-clipped standard deviation of the data
        #dev = np.std(clipped)

        # Set the threshold for the linearly-scaled portion of the map
        #linthresh = dev

        # Define colormap that shows NaNs as black
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='black')

        # Get basic figure properties
        yd, xd = self.model.data.shape
        aspect, colorbar_orient, figsize = \
            determine_figure_properties(xd, yd, threshold=self.threshold_for_nonsquare_pix,
                                        maxsize=self.maxsize)

        # Use SymLogNorm with a linear region around zero
        #norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)

        #self.fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, constrained_layout=True)
        self.fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        #im = ax.imshow(self.model.data, cmap=cmap, norm=norm, aspect=aspect)
        ax.set_title(self.model.meta.filename)
        #ax.set_xlabel("Pixels")
        #ax.set_ylabel("Pixels")
        #ax.invert_yaxis()

        # Choose symmetric ticks around zero for the colorbar
        #pos_ticks = np.logspace(np.log10(vmax) - 3, np.log10(vmax*1.01), 4)
        #neg_ticks = np.logspace(np.log10(-vmin) - 3, np.log10(-vmin), 4)
        #ticks = np.concatenate([-neg_ticks[::-1], np.array([0]), pos_ticks])

        # Shared colorbar
        #cbar = self.fig.colorbar(im, ax=ax, orientation=colorbar_orient, fraction=0.15, pad=0.05)
        #cbar.set_ticks(ticks)
        #cbar.set_ticklabels([f"{t:.1e}" for t in ticks])  # scientific notation
        #cbar.locator = SymmetricalLogLocator(linthresh=dev, base=10)
        #cbar.update_ticks()
        # ----- Tick generation -----


        """
        ticks = []

        # Always include full range endpoints
        #ticks.extend([vmin, vmax])

        # Linear region ticks
        nlin_ticks = 3
        lin_ticks = np.linspace(-linthresh, linthresh, nlin_ticks)

        # Negative log region (only if needed)
        nlog_neg_ticks = 3
        if vmin < -linthresh:
            neg_log_min = np.log10(-vmin)
            neg_log_max = np.log10(linthresh)
            neg_ticks = -np.logspace(neg_log_max, neg_log_min, nlog_neg_ticks, base=10)
            ticks.extend(neg_ticks[1:])  # Skip the first so we don't repeat

        # Positive log region
        nlog_pos_ticks = 3
        if vmax > linthresh:
            pos_log_min = np.log10(linthresh)
            pos_log_max = np.log10(vmax)
            pos_ticks = np.logspace(pos_log_min, pos_log_max, nlog_pos_ticks, base=10)
            ticks.extend(pos_ticks[1:])  # Skip the first so we don't repeat

        # Combine all and filter
        ticks = np.unique(np.concatenate((ticks, lin_ticks)))
        ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
        ticks = np.sort(ticks)



        ticks = [0]

        # Positive side
        if vmax > linthresh:
            # Find the midpoint in log space between linthresh and vmax
            log_linthresh = np.log10(linthresh)
            log_vmax = np.log10(vmax)
            log_mid = (log_linthresh + log_vmax) / 2
            mid_positive = 10 ** log_mid
            ticks.extend([linthresh, mid_positive, vmax])
        else:
            ticks.append(vmax)

        if vmin < -linthresh:
            # Find the midpoint in log space between -linthresh and vmin
            log_linthresh = np.log10(linthresh)
            log_vmin = np.log10(abs(vmin))
            log_mid = (log_linthresh + log_vmin) / 2
            mid_negative = -(10 ** log_mid)
            ticks = [vmin, mid_negative, -linthresh] + ticks
        else:
            ticks.insert(0, vmin)

        # Apply
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.2g}" for t in ticks])
        cbar.ax.set_ylabel("Signal (SymLog scale)")

        # Remove unlabeled minor ticks
        cbar.ax.tick_params(which="minor", length=0)
        # Critical: Remove all minor ticks
        #cbar.ax.yaxis.set_minor_locator(plt.NullLocator())
        """

        shiftdata, shiftmin, shiftmax, tickvals, tlabelflt = shift_data_get_ticks(self.model.data, vmin, vmax, num_ticks=5)
        tlabelstr = formatted_tick_labels(tlabelflt)



        """
        shiftdata = image - min_value + 1
        shiftmin = 1
        shiftmax = max_value - min_value + 1

        # Generate tick labels
        tickvals = np.logspace(np.log10(shiftmin), np.log10(shiftmax), 5)
        tlabelflt = tickvals + min_value - 1
        """

        # Image object
        cax = ax.imshow(shiftdata,
                            norm=colors.LogNorm(vmin=shiftmin,
                                                vmax=shiftmax),
                            cmap=cmap,
                            aspect=aspect)

        # Invert y axis in all cases
        ax.invert_yaxis()

        # For preview images, add colorbar, and create tick labels for it
        if 1>0:
            ysize, xsize = self.model.data.shape

            """
            # Adjust the number of digits after the decimal point
            # in the colorbar labels based on the signal range
            delta = tlabelflt[-1] - tlabelflt[0]
            if delta >= 100:
                dig = 0
            elif ((delta < 100) & (delta >= 10)):
                dig = 1
            elif ((delta < 10) & (delta >= 1)):
                dig = 2
            elif delta < 1:
                dig = 3
            else:
                dig = 2
            format_string = "%.{}f".format(dig)
            tlabelstr = [format_string % number for number in tlabelflt]
            """
            xyratio = xsize / ysize
            print('xyratio: ', xyratio)
            divider = make_axes_locatable(ax)
            #if xyratio < 1.6:
            if colorbar_orient == 'vertical':
                # For apertures that are taller than they are wide, square, or that are wider than
                # they are tall but still reasonably close to square, put the colorbar on the right
                # side of the image.
                cax_colorbar = divider.append_axes("right", size="5%", pad=0.5)

                cbar = self.fig.colorbar(cax, cax=cax_colorbar, ticks=tickvals, orientation='vertical')

                cbar.ax.yaxis.minorticks_off()
                cbar.ax.set_yticklabels(tlabelstr)
                cbar.ax.set_ylabel(self.model.meta.bunit_data, labelpad=15, rotation=270)


                """
                # Some magic numbers arrived at through testing aspect ratios for all apertures
                if xyratio > 0.4:
                    cb_width = 0.05
                else:
                    cb_width = 0.05 * 0.4 / xyratio

                upper_x_anchor = 0.02
                if xyratio < 0.1:
                    upper_x_anchor = 0.12

                cbax = self.fig.add_axes([ax.get_position().x1 + upper_x_anchor,
                                          ax.get_position().y0,
                                          cb_width,
                                          ax.get_position().height
                                          ])
                cbar = self.fig.colorbar(cax, cax=cbax, orientation='vertical', ticks=tickvals)

                cbar.ax.yaxis.minorticks_off()
                cbar.ax.set_yticklabels(tlabelstr)
                cbar.ax.set_ylabel(self.units, labelpad=7, rotation=270)
                """
            elif colorbar_orient == 'horizontal':
                # For apertures that are significantly wider than they are tall, put the colorbar
                # under the image.
                cax_colorbar = divider.append_axes("bottom", size="5%", pad=0.5)






                # Again, some magic numbers controlling the positioning and height of the
                # colorbar, based on testing.
                #lower_y_anchor = 0. - (xyratio / 14.5)
                #cb_height = 0.07 * (np.log2(xyratio) - 1)


                # Check if colorbar is within bounds
                #cb_y_position = ax.get_position().y0 + lower_y_anchor
                #print(f"Colorbar y position: {cb_y_position}")
                #if cb_y_position < 0:
                #    print("WARNING: Colorbar is below figure bounds!")




                #cbax = self.fig.add_axes([ax.get_position().x0,
                #                          ax.get_position().y0 + lower_y_anchor,
                #                          ax.get_position().width,
                #                          cb_height])
                cbar = self.fig.colorbar(cax, cax=cax_colorbar, ticks=tickvals, orientation='horizontal')

                cbar.ax.xaxis.minorticks_off()
                cbar.ax.set_xticklabels(tlabelstr)
                cbar.ax.set_xlabel(self.model.meta.bunit_data, labelpad=7, rotation=0)

            # Set text sizes
            maxsize = self.maxsize
            ax.set_xlabel('Pixels', fontsize=maxsize * 5. / 4)
            ax.set_ylabel('Pixels', fontsize=maxsize * 5. / 4)
            ax.tick_params(labelsize=maxsize)
            plt.rcParams.update({'axes.titlesize': 'small'})
            plt.rcParams.update({'font.size': maxsize * 5. / 4})
            plt.rcParams.update({'axes.labelsize': maxsize * 5. / 4})
            plt.rcParams.update({'ytick.labelsize': maxsize * 5. / 4})
            plt.rcParams.update({'xtick.labelsize': maxsize * 5. / 4})














        #cbar.set_label(self.model.meta.bunit_data)
        plt.show()
        #self.fig.close()
        self.figures.append(self.fig)

    def tso_whitelight_curve_prev_version(self):
        """Plot the whitelight curve. This function covers NRC_TSGRISM, MIR_LRS-SLITLESS and NIS_SOSS,
        as well as the phot.ecsv files from NRC_TSGRISM.
        """
        # There are two options for the column name that contains the MJD dates
        # of the data
        time_key = 'MJD'
        if 'NRC' in self.metadata['exp_type']:
            time_key = 'MJD_UTC'

        flux_key = 'whitelight_flux'
        if 'phot.ecsv' in self.filename:
            flux_key = 'net_aperture_sum'

        self.fig, ax = plt.subplots(figsize=(self.maxsize*1.375, self.maxsize))
        ax.plot(self.model[time_key], self.model[flux_key], color='black')
        ax.set_xlabel('Time (MJD)')
        ax.set_ylabel(flux_key)
        ax.set_title(f'{os.path.basename(self.filename)}: {self.metadata["target_name"]}, {self.metadata["number_of_integrations"]} ints')
        #plt.show()
        #self.fig.close()
        self.figures.append(self.fig)

    def tso_whitelight_curve(self):
        """
        Create and display white light curve.
        """
        # There are two options for the column name that contains the MJD dates
        # of the data
        time_key = 'MJD'
        if self.metadata['exp_type'] in ['NRC_TSGRISM', 'NIS_SOSS']:
            #if 'NRC' in self.metadata['exp_type'] or 'NIS' in self.metadata['exp_type']:
            time_key = 'MJD_UTC'

        # NIS_SOSS will have a separate whitelight column for each spectral order
        # NRC_TSGRISM will have a single whitelight_flux column
        flux_keys = [c for c in self.model.columns if 'whitelight' in c]
        print(flux_keys)

        #flux_key = 'whitelight_flux'
        if 'phot.ecsv' in self.filename:
            flux_keys = ['net_aperture_sum']

        # --------------------------Set up figures--------------------------
        #self.fig, axes = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[1, 2])
        self.fig, axes = plt.subplots(figsize=(self.maxsize, self.maxsize))
        #self.fig.subplots_adjust(hspace=0.3, wspace=0.2)
        #axlc, axslc = axes
        axlc = axes

        n_spec = len(self.model[time_key])  # Number of spectra (integrations).
        all_times = self.model[time_key]
        time_axis = (all_times - np.nanmean(all_times)) * 24.0

        for flux_key in flux_keys:
            if 'order' in flux_key:
                order_str = 'order ' + flux_key.split('_')[-1]
            else:
                order_str = ''

            # ---------------------Obtain white light curve---------------------
            wlc_flux = self.model[flux_key]

            # Calculate light curve scatter from first ~100 points.
            wlc_flux_scatter = sigma_clip(wlc_flux, sigma=2, maxiters=2, masked=False)
            sigma_wlc = np.sqrt(np.nanvar(wlc_flux_scatter[2:100]))
            sigma_wlc_ppm = round(sigma_wlc * 1e6, 0)

            # Plot white light curve
            if 'whitelight' in flux_key:
                label = f"White light curve, " + order_str + f"(r.m.s.={round(sigma_wlc * 1e6, 0)} ppm)"
            elif flux_key == 'net_aperture_sum':
                label = f"Photometry curve, (r.m.s.={round(sigma_wlc * 1e6, 0)} ppm)"
            axlc.plot(time_axis, wlc_flux, marker='o', markersize=2,
                      label=label)

        axlc.set_title(f"{os.path.basename(self.filename)}: White Light Curve", fontsize=12)
        axlc.legend(loc="lower right")
        axlc.set_xlabel("Time since mid-exposure, (hr)", fontsize=15)
        axlc.set_ylabel("Normalized flux", fontsize=15)

        # Add secondary x-axis for integration indices
        integration_indices = np.arange(n_spec)
        tick_positions = np.linspace(time_axis.min(),
                                     time_axis.max(),
                                     len(integration_indices))
        axlc_secondary = axlc.secondary_xaxis('top')
        axlc_secondary.set_xlabel("Integration Index", fontsize=12)
        axlc_secondary.set_xticks(tick_positions[::len(tick_positions) // 10])
        axlc_secondary.set_xticklabels([f"{int(idx)}" for idx in
                                        integration_indices[::len(integration_indices) // 10]])
        self.figures = [self.fig]


    def find_wfss_source_ext(self, cal_hdu, source_id):
        """Given a source ID number, find the extension in the cal file where the source is located

        Parameters
        ----------
        cal_hdu : astropy.io.fits.HDUList
            *_cal.fits file HDUList

        Returns
        -------
        wh_cal : int
            Extension number where the source_id exists
        """
        # Get a list of SOURCE_IDs for all extensions. Set all non-SCI extension values to -999
        cal_source_ids = np.array([cal_hdu[ext].header['SOURCEID'] if cal_hdu[ext].header['EXTNAME'] == 'SCI'
                                  else -999 for ext in range(1, len(cal_hdu))])
        cal_orders = np.array([cal_hdu[ext].header['SPORDER'] if cal_hdu[ext].header['EXTNAME'] == 'SCI'
                                  else -999 for ext in range(1, len(cal_hdu))])

        # Get extension numbers for order 1 and 2 spectra. Add 1 for the primary header
        cal_ex_o1 = np.where((cal_source_ids == source_id) & (cal_orders == 1))[0] + 1
        cal_ex_o2 = np.where((cal_source_ids == source_id) & (cal_orders == 2))[0] + 1

        if len(cal_ex_o1) > 0:
            cal_ex_o1 = cal_ex_o1[0]
        else:
            cal_ex_o1 = -999
        if len(cal_ex_o2) > 0:
            cal_ex_o2 = cal_ex_o2[0]
        else:
            cal_ex_o2 = -999
        return (cal_ex_o1, cal_ex_o2)

    def get_wfss_cal_data(self, hdulists, source_num):
        """Find and extract the 2D spectrum from the WFSS cal file for a given source ID

        Parameters
        ----------
        hdulists : list
            List of HDULists of the cal files

        source_num : int
            Source ID number

        Returns
        -------
        cal_data : numpy.ndarray
            2D extracted spectrum corresponding to source_num
        """
        # Find the cal file extension where the 2D spectrum is located.
        # Get the cal data if the source is present
        cal_data = None
        cal_width = 0
        cal_name = ''
        cal_ext = -999
        cal_order = -999
        for hdulist in hdulists:

            ext_o1, ext_o2 = self.find_wfss_source_ext(hdulist, source_num)

            for ext in [ext_o1, ext_o2]:
                if ext != -999:
                    data = hdulist[ext].data

                    # Check the size of the 2D cutout. In order to avoid showing a
                    # dither where the source is close to the edge and the data quality
                    # isn't good, look through all cal files and keep the longest spectrum.
                    width = data.shape[-1]
                    dispersion_direction = hdulist[ext].header['DISPAXIS']
                    if dispersion_direction == 2:
                        width = data.shape[-2]

                    # If the 2D array is longer than the previous longest, keep it.
                    if width > cal_width:
                        # If the dispersion direction is along columns, transpose in
                        # order to make plotting easier.
                        if dispersion_direction == 2:
                            data = np.transpose(data)

                        # NIRISS data are dispersed in the -x direction, so flip the
                        # 2D cutout horizontally so that the wavelength direction will
                        # match that in the 1D figure
                        if hdulist[0].header['INSTRUME'] == 'NIRISS':
                            data = np.fliplr(data)

                        name = hdulist[0].header['FILENAME']
                        units = hdulist[ext].header['BUNIT']

                        cal_width = width
                        cal_data = data
                        cal_name = name
                        cal_ext = ext
                        cal_units = units

                        if ext == ext_o1:
                            cal_order = 1
                        elif ext == ext_o2:
                            cal_order = 2

        return cal_data, cal_name, cal_ext, cal_order, cal_units

    def wfss_calc_yrange(self, spec, ignore_frac=0.2, padding=0.1):
        """Try to do something sort of intelligent to come up with a plot range
        for the 1d spectral plots. Often the flux values go unrealistically high
        at the red and blue ends. Would be nice to not let that drive the plot range.

        Parameters
        ----------
        """
        fin = np.isfinite(spec)
        finite_flux = spec[fin]

        if len(finite_flux) > 0:
            # Ignore the bluest and reddest points
            n_ignore = int(len(finite_flux) * ignore_frac)
            chopped = finite_flux[n_ignore: len(finite_flux)-n_ignore]
            ylower = np.nanmin(chopped) - np.abs(np.nanmin(chopped) * padding)
            yupper = np.nanmax(chopped) * (1. + padding)
        else:
            ylower = 0
            yupper = 1
        return ylower, yupper

    def wfss_x1d(self):
        """Create a compound preview image for a source in WFSS data that shows:

           1. A cutout of the object in the direct (i2d) file
           2. Image of the 2D extracted spectrum from one of the cal files
           3. Plot of the 1D extracted spectrum (1st and 2nd order if present)
        """
        n_ext = len(self.model.spec)
        bright_idx = self.find_brightest_wfss_sources(nbrightest=2)
        self.wfss_source_ids = []

        # Get the corresponding cal file data
        if '-' in self.model.meta.filename:
            if 'x1d' in self.model.meta.filename:  # Stage 3 x1d file
                cal_files = [ext.filename for ext in self.model.spec]
            elif 'c1d' in self.model.meta.filename:  # Stage 3 c1d file
                x1dname = self.filename.replace('c1d', 'x1d')
                x1dmodel = datamodels.open(x1dname)
                cal_files = [ext.filename for ext in x1dmodel.spec]
            cal_files = list(set(cal_files))

            # cal files are located in a different directory from the level 3 files
            cal_files = [filesystem_path(filename, check_existence=True) for filename in cal_files]
        else:  # Stage 2 x1d file
            suffix = self.filename.split('_')[-1].split('.fits')[0]
            cal_files = [self.filename.replace(suffix, 'cal')]

        cal_hdus = [fits.open(cal_file) for cal_file in cal_files]

        # Define colormap that shows NaNs as black
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='black')

        for idx in bright_idx:
            # Dictionary to hold all the 1D spectra
            spec1d = {}

            source_id = self.model.spec[0].spec_table.SOURCE_ID[idx]

            # Find the cal file extension where the 2D spectrum is located.
            # Get the cal data if the source is present
            cal_data, cal_name, cal_ext, cal_order, cal_units = self.get_wfss_cal_data(cal_hdus, source_id)

            # Determine the source type and which flux units to use
            source_type = self.model.spec[0].spec_table.SOURCE_TYPE[idx]

            if source_type == 'EXTENDED':
                flux_col = 'SURF_BRIGHT'
            elif source_type == 'POINT':
                flux_col = 'FLUX'
            else:
                raise ValueError(f'Unrecognized source_type: {source_type}')

            # Get 1D spectrum from 0th extension and populate the dictionary
            spec1d['order'] = [self.model.spec[0].spectral_order]
            spec1d['flux'] = [np.array(self.model.spec[0].spec_table[flux_col][idx, :])]
            spec1d['wavelength'] = [np.array(self.model.spec[0].spec_table['WAVELENGTH'][idx, :])]

            # Convert Jy to f_lambda
            if flux_col == 'FLUX':
                spec1d['flux'] = [Fnu_to_Flam(spec1d['wavelength'][0], spec1d['flux'][0])]
                #spec1d['flux'] = [Fnu_to_Flam(ele['wavelength'], ele['flux']) for ele in spec1d['flux']]

            # Get the units for later plotting
            flux_units = self.model.spec[0].spec_table.columns[flux_col].unit
            wavelength_units = self.model.spec[0].spec_table.columns['WAVELENGTH'].unit
            if flux_col == 'FLUX':
                flux_units = 'F_lambda (erg/cm2/s/A)'

            # Loop over other extensions and look for the same source_id.
            for exten in range(1, n_ext):
                if source_id in self.model.spec[exten].spec_table['SOURCE_ID']:
                    row = self.model.spec[exten].spec_table[self.model.spec[exten].spec_table['SOURCE_ID'] == source_id][0]

                    if flux_col == 'FLUX':
                        fluxes = Fnu_to_Flam(np.array(row['WAVELENGTH']), np.array(row[flux_col]))

                    spec1d['order'].append(self.model.spec[exten].spectral_order)
                    spec1d['flux'].append(fluxes)
                    spec1d['wavelength'].append(np.array(row['WAVELENGTH']))

            # Remove any entries where e.g. the 2nd order data are all NaN
            spec1d = self.remove_wfss_nan_data(spec1d)

            orders = set(spec1d['order'])
            if len(orders) > 2:
                raise ValueError((f'Extracted {len(orders)} orders: {orders}. '
                                  'Plotting only supported for up to 2 orders.'))

            # Create the figure
            self.fig = plt.figure(figsize=(12, 12))
            ax_i2d = plt.subplot2grid((3, 3), (0, 0))
            ax_2dspec = plt.subplot2grid((3, 3), (0, 1), colspan=2)
            ax_1d_o1 = plt.subplot2grid((3, 3), (1, 0), colspan=3)
            ax_1d_o2 = plt.subplot2grid((3, 3), (2, 0), colspan=3)

            # If 2nd order spectra are present, set the plot limits
            if len(orders) == 2:
                second = np.where(np.array(spec1d['order']) == 2)[0][0]
                ylower_o2, yupper_o2 = self.wfss_calc_yrange(spec1d['flux'][second])
                ax_1d_o2.set_ylim(ylower_o2, yupper_o2)
            else:
                ax_1d_o2.set_ylim(0, 1)
                ax_1d_o2.set_xlim(0, 1)
                ax_1d_o2.annotate('No order 2 data found', xy=(0.4, 0.5), color='black')

            # Set title and labels for second order plot regardless of whether the spectrum
            # is present or not
            ax_1d_o2.set_title(f'{self.model.meta.filename}, Source {source_id}, Order 2')
            ax_1d_o2.set_xlabel(f'Wavelength ({wavelength_units})')
            ax_1d_o2.set_ylabel(f'{flux_units}')

            # Overplot all 1d spectra for both orders
            num_o1 = len(np.where(np.array(spec1d['order']) == 1)[0])
            num_o2 = len(np.where(np.array(spec1d['order']) == 2)[0])
            for i in range(len(spec1d['order'])):
                if spec1d['order'][i] == 1:
                    ax_tmp = ax_1d_o1
                    alpha = 1. / num_o1
                    if alpha < 1:
                        alpha *= 2
                elif spec1d['order'][i] == 2:
                    ax_tmp = ax_1d_o2
                    alpha = 1. / num_o2
                    if alpha < 1:
                        alpha *= 2
                else:
                    raise ValueError(f'Encountered unsupported spectral order {spec1d["order"][i]}')

                # Plot the 1D spectrum
                ax_tmp.plot(spec1d['wavelength'][i], spec1d['flux'][i], alpha=alpha, ds='steps-mid')

            first = np.where(np.array(spec1d['order']) == 1)[0][0]
            ylower_o1, yupper_o1 = self.wfss_calc_yrange(spec1d['flux'][first])

            # Set 1st order 1D plot title and labels
            ax_1d_o1.set_title(f'{self.model.meta.filename}, Source {source_id}, Order 1')
            ax_1d_o1.set_xlabel(f'Wavelength ({wavelength_units})')
            ax_1d_o1.set_ylabel(f'{flux_units}')
            ax_1d_o1.set_ylim(ylower_o1, yupper_o1)

            # Show the 2D extracted spectrum
            # Clip brightest and dimmest 1% of pixels to find min and max values
            # Throw out the outermost 4 rows on each side when determining colormap
            # scaling, just in case they are reference pixels
            cal_vmin = np.nanpercentile(cal_data[4:-4, 4:-4], 1)
            cal_vmax = np.nanpercentile(cal_data[4:-4, 4:-4], 99)

            # Use to determine linearly scaled signals
            clipped = sigma_clip_ignore_nan(cal_data)

            # Find sigma-clipped standard deviation of the data
            dev = np.std(clipped)

            # Use SymLogNorm with a linear region around zero
            norm = SymLogNorm(linthresh=dev/1000., vmin=cal_vmin, vmax=cal_vmax)
            im2d = ax_2dspec.imshow(cal_data, norm=norm, cmap=cmap, origin='lower', aspect='auto')
            ax_2dspec.set_xlabel('Pixel')
            ax_2dspec.set_ylabel('Pixel')
            ax_2dspec.set_title(f'{cal_name}, Ext {cal_ext}, Order {cal_order}')

            # Add colorbar
            # Create a separate axes for the colorbar, right next to the image
            divider = make_axes_locatable(ax_2dspec)
            cax_2d = divider.append_axes("right", size="5%", pad=0.05)  # thickness=5%, gap=0.05
            self.fig.colorbar(im2d, cax=cax_2d, label=cal_units)


            #cbar_2dspec = self.fig.colorbar(im2d, ax=ax_2dspec, orientation="vertical") #, fraction=0.15, pad=0.05)
            #cbar_2dspec.set_label(cal_units)

            # Show the i2d cutout
            i2d_file = os.path.join(os.path.dirname(self.filename), self.model.meta.direct_image)
            ax_i2d, im_i2d = self.plot_i2d_plus_source(i2d_file, source_id, ax_i2d)

            self.fig.tight_layout(pad=2.0)
            self.figures.append(self.fig)
            self.wfss_source_ids.append(source_id)
        """

            self.identify_wfss_cal_files()
            self.get_wfss_2d_cutout()


        open x1d file:
        model.meta.asn.table_name - name of association file
        model.meta.direct_image - name of i2d file (direct image)
        model.meta.target.catalog_name - target name


        model.spec[0].spec_table columns include extracted x,y.  from the dispersed image?
        Also has SOURCE_ID. And SOURCE_TYPE which will be helpful because we should plot FLUX for point sources, and SURF_BRIGHT for extended sources


        model.spec[0].filename - gives cal filename associated with the spectra in this extension
        model.spec[0].spectral_order - gives the spectral order associated with the extension
        """

    def plot_i2d_plus_source(self, i2dname, source_id, ax, box_hw=10):
        """Open the i2d & catalog files, and show an annotated image of the source associated
        with the source number.

        Parameters
        ----------
        i2dname : str
            Name of source i2d file

        source_id : int
            Source ID number

        ax : matplotlib.axes._axes.Axes
            Figure axis that will be used to show the image

        box_hw : int
            Half-width of the box to show containing the source

        Returns
        -------
        ax : matplotlib.axes._axes.Axes
            Figure axis with annotated image added
        """
        catname = i2dname.replace('i2d.fits', 'cat.ecsv')

        # Get source location from the source catalog
        cat = Table.read(catname)
        cat_line = cat[cat['label'] == source_id]
        xcentroid = cat_line['xcentroid'][0]
        ycentroid = cat_line['ycentroid'][0]
        xstart = int(xcentroid - box_hw)
        ystart = int(ycentroid - box_hw)
        xstop = int(xcentroid + box_hw)
        ystop = int(ycentroid + box_hw)

        # Define colormap that shows NaNs as black
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='black')

        # Plot the image
        with fits.open(i2dname) as i2d:
            if ystart < 0:
                ystart == 0
            if xstart < 0:
                xstart = 0
            if xstop > i2d['SCI'].data.shape[-1]:
                xstop = i2d['SCI'].data.shape[-1]
            if ystop > i2d['SCI'].data.shape[-2]:
                ystop = i2d['SCI'].data.shape[-2]

            cutout = i2d[1].data[ystart: ystop, xstart: xstop]

            # Clip brightest and dimmest 1% of pixels to find min and max values
            vmin = np.nanpercentile(cutout, 1)
            vmax = np.nanpercentile(cutout, 99)

            # Use to determine linearly scaled signals
            clipped = sigma_clip_ignore_nan(cutout)

            # Find sigma-clipped standard deviation of the data
            dev = np.std(clipped)

            # Use SymLogNorm with a linear region around zero
            norm = SymLogNorm(linthresh=dev/1000., vmin=vmin, vmax=vmax)
            imi2d = ax.imshow(cutout, norm=norm, cmap=cmap, origin='lower', aspect='auto')

        # Shift the tick values to the full frame coordinate system
        ax.set_xticks(np.arange(0, 2*box_hw+1, 5))
        ax.set_yticks(np.arange(0, 2*box_hw+1, 5))
        new_xticks = ax.get_xticks() + xstart
        new_yticks = ax.get_yticks() + ystart
        ax.set_xticklabels(new_xticks)
        ax.set_yticklabels(new_yticks)
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Pixel')

        # Add annotation showing location and source ID
        ax.scatter(xcentroid - xstart, ycentroid - ystart, s=20, facecolors='None', edgecolors='magenta', alpha=0.9)
        ax.annotate(source_id,
                    (xcentroid-xstart+0.5, ycentroid-ystart+0.5),
                    fontsize=10,
                    color='magenta')
        ax.set_title(i2dname, fontsize=10)

        # Add colorbar
        # Create a separate axes for the colorbar, right next to the image
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)  # thickness=5%, gap=0.05
        self.fig.colorbar(imi2d, cax=cax, label=i2d[1].header['BUNIT'])

        #cbar_i2d = self.fig.colorbar(imi2d, ax=ax, orientation="vertical", label=i2d[1].header['BUNIT'], pad=0.01)

        return ax, imi2d

    def find_brightest_wfss_sources(self, nbrightest=10):
        """Determine the `nsources` brightest sources in the WFSS file. Do this using the
        mean value. Work on only the model.spec[0].spec_table. Note that if there are dithers,
        the source may not be present in all extensions.

        Parameters
        ----------
        nbrightest : int
            Number of brightest sources to keep

        Returns
        -------
        brightest : numpy.ndarray
            1D array of index numbers corresponding to the `nsources` brightest sources.
        """
        num_ext = len(self.model.spec)

        num_sources = self.model.spec[0].spec_table.shape[0]

        medians = []
        for source in range(num_sources):


            # Not sure how to compare flux vs surf_bright to determine which sources are brightest
            # Can probably just look at surf_bright for everyone and find the brightest there.
            #source_type = self.model.spec[0].spec_table.SOURCE_TYPE[source]
            #if source_type == 'EXTENDED':
            #    flux_col = 'SURF_BRIGHT'
            #elif source_type == 'POINT':
            #    flux_col = 'FLUX'
            ##flux_units = self.model.spec[0].spec_table.columns[flux_col].unit


            # Ignore sources where the source_id is empty, which indicates a larger problem.
            if self.model.spec[0].spec_table['SOURCE_TYPE'][source] != '':
                medians.append(np.nanmedian(self.model.spec[0].spec_table['SURF_BRIGHT'][source, :]))

        idxs = np.argsort(medians)[::-1]
        brightest = idxs[0:nbrightest]
        return brightest

    def remove_wfss_nan_data(self, data):
        """Look through wfss data included in a dictionary, and remove instances where the fluxes
        are all nan. This will most commonly happen where the 1st order data are present, but then it
        turns out the 2nd order data are empty. Keys in the dictionary are: order, flux, and wavelength.
        The dictionary contains data only for a single source, but multiple exposures/dithers/orders
        """
        new_dict = {'order': [], 'flux': [], 'wavelength': []}
        for index in range(len(data['order'])):
            # If there are any finite flux values, keep the array
            if ~np.all(~np.isfinite(np.array(data['flux'][index]))):
                new_dict['order'].append(data['order'][index])
                new_dict['flux'].append(data['flux'][index])
                new_dict['wavelength'].append(data['wavelength'][index])
        return new_dict

    def locate_other_mrs_x1ds(self):
        """
        """
        dirname, fname  = os.path.split(self.filename)
        x1dfiles = sorted(glob(os.path.join(dirname, '*x1d.fits')))

        # Remove self.filename so that we don't double count it
        #x1dfiles.remove(self.filename)

        # Sort each channel to be in short, medium, long order, to
        # make the plot more readable
        sorted_files = sorted(x1dfiles, key=lambda f: (
            int(re.search(r"ch(\d+)", f).group(1)),
            -ord(re.search(r"-(short|medium|long)", f).group(1)[0])  # crude way
        ))
        return sorted_files

    def miri_mrs_x1d(self):
        """Plot spectrum from x1d file from NIRSpec fixed slit or IFU file,
        or MIRI LRS fixed slit or IFU (MRS) mode. In these cases, there is only
        one self.model.spec extension, and the flux and wavelength data within
        that extension is 1-dimensional.
        """
        self.fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(self.maxsize, self.maxsize * 1.5))

        flux = self.model.spec[0].spec_table.RF_FLUX

        # MRS x1d files have both regular ('flux') and residual-fringe (RF) corrected ('rf_flux') spectra.
        # The RF-corrected spectra will have NaN values if RF correction was disabled or failed to converge.
        # Plot the RF corrected spectrum if available, otherwise plot the regular spectrum.
        if np.nansum(flux == 0):
            flux = self.model.spec[0].spec_table.FLUX

        waves = self.model.spec[0].spec_table.WAVELENGTH
        targname = self.model.meta.target.proposer_name
        channel_band = f'{self.model.meta.instrument.channel} {self.model.meta.instrument.band}'

        # Determine plot range
        clipped = sigma_clip_ignore_nan(flux, sigma=9)
        vmax = np.nanmax(clipped) * 1.1
        vmin = np.nanmin(clipped)
        if vmin >= 0:
            vmin = vmin * 0.9
        else:
            vmin = vmin * 1.1

        ax[0].plot(waves, flux, color='blue', label=channel_band)
        ax[0].set_xlabel(f'Wavelength ({self.wavelength_units})')
        ax[0].set_ylabel(f'Flux ({self.flux_units})')
        ax[0].legend()
        ax[0].set_ylim(vmin, vmax)
        #self.fig.suptitle(f'{self.model.meta.filename}')
        ax[0].set_title(f'{self.model.meta.filename}\n{targname}')

        # Locate any other related x1d files that are related. If others are found, plot
        # all the x1d files together. If no other files are found, then create only the
        # single plot above
        x1dfiles = self.locate_other_mrs_x1ds()

        if len(x1dfiles) > 1:
            vmin, vmax = np.nan, np.nan
            for x1dfile in x1dfiles:

                mod = datamodels.open(x1dfile)
                flux = mod.spec[0].spec_table.RF_FLUX
                waves = mod.spec[0].spec_table.WAVELENGTH
                channel_band = f'{mod.meta.instrument.channel} {mod.meta.instrument.band}'

                # MRS x1d files have both regular ('flux') and residual-fringe (RF) corrected ('rf_flux') spectra.
                # The RF-corrected spectra will have NaN values if RF correction was disabled or failed to converge.
                # Plot the RF corrected spectrum if available, otherwise plot the regular spectrum.
                if np.nansum(flux == 0):
                    flux = mod.spec[0].spec_table.RF_FLUX

                # Determine plot range
                clipped = sigma_clip_ignore_nan(flux, sigma=9)
                vfilemax = np.nanmax(clipped) * 1.1
                vfilemin = np.nanmin(clipped)
                if vfilemin >= 0:
                    vfilemin = vfilemin * 0.9
                else:
                    vfilemin = vfilemin * 1.1
                vmin = np.nanmin([vmin, vfilemin])
                vmax = np.nanmax([vmax, vfilemax])

                ax[1].plot(waves, flux, label=channel_band)
                ax[1].set_xlabel(f'Wavelength ({self.wavelength_units})')
                ax[1].set_ylabel(f'Flux ({self.flux_units})')
                ax[1].set_ylim(vmin, vmax)
                ax[1].legend()
                ax[1].set_title(self.model.meta.target.catalog_name)
        else:
            # In this case, no other x1d files are present
            ax[1].remove()
        self.figures.append(self.fig)

    def miri_lrs_fixed_slit_nirspec_ifu_x1d(self):
        """Plot spectrum from x1d file from NIRSpec fixed slit or IFU file,
        or MIRI LRS fixed slit or IFU (MRS) mode. In these cases, there is only
        one self.model.spec extension, and the flux and wavelength data within
        that extension is 1-dimensional.
        """
        self.fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(self.maxsize, self.maxsize))

        flux = self.model.spec[0].spec_table.FLUX
        waves = self.model.spec[0].spec_table.WAVELENGTH
        targname = self.model.meta.target.proposer_name

        # Determine plot range
        clipped = sigma_clip_ignore_nan(flux, sigma=9)
        vmax = np.nanmax(clipped) * 1.1
        vmin = np.nanmin(clipped)
        if vmin >= 0:
            vmin = vmin * 0.9
        else:
            vmin = vmin * 1.1

        ax.plot(waves, flux, color='blue')
        ax.set_xlabel(f'Wavelength ({self.wavelength_units})')
        ax.set_ylabel(f'Flux ({self.flux_units})')
        #ax.legend()
        ax.set_ylim(vmin, vmax)
        #self.fig.suptitle(f'{self.model.meta.filename}')
        ax.set_title(f'{self.model.meta.filename}\n{targname}')

        self.figures.append(self.fig)


    def fixed_slit_cal_crf_s2d(self):
        """Create preview image for NIRSpec or MIRI fixed slit cal, crf, and s2d files
        """
        # Modify fig size based on number of nods
        figure_size = {'1': (12, 4),
                       '2': (12, 6),
                       '3': (12, 8),
                       '5': (12, 12)
                       }

        # MIRI slits are presented vertically in the data, while NIRSpec slits are
        # presented horizontally. If we have a MIRI file, we need to essentially swap
        # x and y figure size, orientation, etc
        inst = self.model.meta.instrument.name.lower()

        if 'cal' in self.model.meta.filename or 'crf' in self.model.meta.filename:
            num_nods = len(self.model.exposures)
            figsize = figure_size[str(num_nods)]
            aspect = "auto"
            colorbar_orient = "horizontal"

            if inst == 'miri':
                figsize = (figsize[::-1])
                colorbar_orient = "vertical"

            arrs = [self.model.exposures[i].data for i in range(num_nods)]
            titles = [f'Nod {i}' for i in range(num_nods)]
            titles[0] = f'{self.model.meta.filename}: Nod 0'

            # Clip brightest and dimmest 1% of pixels to find min and max values
            vmin = np.nanpercentile(self.model.exposures[0].data, 1)
            vmax = np.nanpercentile(self.model.exposures[0].data, 99)

            # Use to determine linearly scaled signals
            clipped = sigma_clip_ignore_nan(self.model.exposures[0].data)

        elif 's2d' in self.model.meta.filename:
            num_nods = 1
            figsize = figure_size[str(num_nods)]
            aspect = "auto"
            colorbar_orient = "horizontal"

            if inst == 'miri':
                figsize = (figsize[::-1])
                colorbar_orient = "vertical"

            arrs = [self.model.data]
            titles = [self.model.meta.filename]

            # Clip brightest and dimmest 1% of pixels to find min and max values
            vmin = np.nanpercentile(self.model.data, 1)
            vmax = np.nanpercentile(self.model.data, 99)

            # Use to determine linearly scaled signals
            clipped = sigma_clip_ignore_nan(self.model.data)

        # Find sigma-clipped standard deviation of the data
        dev = np.std(clipped)

        # Define colormap that shows NaNs as black
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='black')

        # Use SymLogNorm with a linear region around zero
        norm = SymLogNorm(linthresh=dev/1000., vmin=vmin, vmax=vmax)

        # Create figure with 3 stacked subplots
        self.fig, axes = plt.subplots(nrows=num_nods, ncols=1, figsize=figsize, constrained_layout=True)

        if inst == 'miri':
            print('HERE')
            print(aspect, colorbar_orient)
            self.fig, axes = plt.subplots(nrows=1, ncols=num_nods, figsize=figsize, constrained_layout=True)

        # For s2d files, there will be only one set of axes. We need to put this into
        # a numpy array so that we can iterate over it below.
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for ax, arr, title in zip(axes, arrs, titles):
            im = ax.imshow(arr, cmap=cmap, norm=norm, aspect=aspect)
            ax.set_title(title)
            ax.set_xlabel("Pixels")
            ax.set_ylabel("Pixels")
            ax.invert_yaxis()

        # Choose symmetric ticks around zero for the colorbar
        pos_ticks = np.logspace(np.log10(vmax) - 3, np.log10(vmax*1.01), 4)
        neg_ticks = np.logspace(np.log10(-vmin) - 3, np.log10(-vmin), 4)
        ticks = np.concatenate([-neg_ticks[::-1], np.array([0]), pos_ticks])

        # Shared colorbar
        cbar = self.fig.colorbar(im, ax=axes[-1], orientation=colorbar_orient, fraction=0.2, pad=0.05)
        cbar.set_ticks(ticks)
        cbar.ax.xaxis.minorticks_off()
        cbar.set_ticklabels([f"{t:.1e}" for t in ticks])  # scientific notation

        # Looks like the units don't always make it into the datamodel (e.g. NIRSpec fixedslit crf)
        try:
            units = self.model.meta.bunit_data
        except AttributeError:
            units = fits.getheader(self.filename, 1)['BUNIT']
        cbar.set_label(units)
        #plt.show()
        #self.fig.close()
        self.figures.append(self.fig)

    def nirspec_miri_ifu_s3d(self):
        nframes, yd, xd = self.model.data.shape
        frames_to_view = [1]  # make this realistic before merging - waiting on word from NIRSpec team via Kayli Glidic

        # Get basic figure properties
        aspect, colorbar_orient, figsize = \
            determine_figure_properties(xd, yd, threshold=self.threshold_for_nonsquare_pix,
                                        maxsize=self.maxsize)

        # Determine the vmin and vmax values for scaling based on a mean frame
        # To save time, take the mean over only a subset of frames. Focus on the
        # shorter wavelngths where the signals are most likely higher
        strt_frame = nframes // 9
        end_frame = strt_frame + 100
        if end_frame > nframes:
            end_frame = nframes

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            slice_mean = np.nanmean(self.model.data[strt_frame:end_frame, :, :], axis=0)
        vmin = np.nanpercentile(slice_mean, 1)
        vmax = np.nanpercentile(slice_mean, 99)
        if vmin < -vmax:
            vmin = -vmax

        # Define colormap that shows NaNs as black
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='black')

        for frame in frames_to_view:
            self.fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            slice_full = ax.imshow(self.model.data[frame, :, :],
                                   norm=ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch()),
                                   origin='lower', cmap=cmap, aspect=aspect)
            ax.invert_yaxis()
            ax.set_xlabel('Pixel')
            ax.set_ylabel('Pixel')
            ax.set_title(f'{self.model.meta.filename}: slice {frame}')




            #shiftdata, shiftmin, shiftmax, tickvals, tlabelflt = shift_data_get_ticks(self.model.data[frame,:,:], vmin, vmax, num_ticks=5)
            #tlabelstr = formatted_tick_labels(tlabelflt)

            #divider = make_axes_locatable(ax)
            #cax_colorbar = divider.append_axes("right", size="5%", pad=0.5)
            #cbar = self.fig.colorbar(slice_full, cax=cax_colorbar, ticks=tickvals, orientation='vertical')
            #cbar.ax.yaxis.minorticks_off()
            #cbar.ax.set_yticklabels(tlabelstr)
            #cbar.ax.set_ylabel(self.model.meta.bunit_data, labelpad=15, rotation=270)

            #print(vmin, np.nanmin(self.model.data[frame, :, :]))





            plt.colorbar(slice_full, ax=ax, orientation=colorbar_orient,
                         fraction=0.05, pad=0.04).set_label(self.model.meta.bunit_data, fontsize=15)
            self.figures.append(self.fig)

    def output_name(self):
        """
        """
        suffix = '_integ{}.{}'.format(i, self.output_format)
        if self.preview_output_directory is None:
            outdir = indir
        else:
            outdir = self.preview_output_directory
        outfile = os.path.join(outdir, infile.split('.')[0] + suffix)
        self.save_image(outfile, thumbnail=False)
        plt.close(self.fig)
        self.preview_images.append(outfile)

    def psfstack_file(self):
        """Create preview image for the psfstack files associated with coronaraphic observations
        """
        nframes, yd, xd = self.model.data.shape

        # Discussions with coronagraph folks suggest that showing just one frame from the
        # psfstack should be fine in terms of preview images. JDAViz/other tools can be
        # used to look for shifts in the PSF location after examining the PSF-subtracted
        # i2d image.
        frames_to_view = [0]

        vmin = np.nanpercentile(self.model.data[0, :, :], 1)
        vmax = np.nanpercentile(self.model.data[0, :, :], 99)

        # Use to determine linearly scaled signals
        clipped = sigma_clip_ignore_nan(self.model.data[0, :, :])

        # Find sigma-clipped standard deviation of the data
        dev = np.std(clipped)

        # Define colormap that shows NaNs as black
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='black')

        # Get basic figure properties
        aspect, colorbar_orient, figsize = \
            determine_figure_properties(xd, yd, threshold=self.threshold_for_nonsquare_pix,
                                        maxsize=self.maxsize)

        # Use SymLogNorm with a linear region around zero
        norm = SymLogNorm(linthresh=dev/1000., vmin=vmin, vmax=vmax)

        for frame in frames_to_view:
            # Create figure
            self.fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize,
                                        constrained_layout=True)
            im = ax.imshow(self.model.data[frame, :, :], norm=norm, origin='lower', aspect=aspect,
                           cmap=cmap)
            ax.set_title(f'{self.model.meta.filename}, Frame: {frame}')
            ax.set_xlabel("Pixels")
            ax.set_ylabel("Pixels")

            cbar = self.fig.colorbar(im, ax=ax, orientation=colorbar_orient, shrink=0.8)

            # If there are limited ticks, create them manually
            default_tick_labels = [t.get_text() for t in cbar.ax.get_yticklabels()]
            if len(default_tick_labels) < 2:
                minval = 1e-4
                if vmin > 0:
                    minval = vmin
                new_ticks = np.logspace(np.log10(minval), np.log10(vmax), 5)
                cbar.set_ticks(new_ticks)
                new_ticks_str = [f"{value:.1e}" for value in new_ticks]
                cbar.set_ticklabels(new_ticks_str)
            #print(cbar.ax.get)


            cbar.set_label(self.model.meta.bunit_data)
            self.figures.append(self.fig)


def compile_segments(data_products):
    """
    Compiles extracted 1D spectra, corresponding timestamps,
    and wavelengths from a list of X1D data products output from
    the stage 2 pipeline. (Designed for NIRSpec BOTS data)

    Parameters
    ----------
    data_products : list of str
        A list of data products (X1DINT files).

    Returns
    -------
    all_spec_1D : numpy.ndarray
        A 2D array where each row corresponds to a spectrum from a single
        integration, and columns represent flux values at each wavelength.
    all_times : numpy.ndarray
        A 1D array containing the mid-integration times (e.g., BJD_TDB) for
        each spectrum in `all_spec_1D`.
    """

    data_products = [data_products] if isinstance(data_products, str) else data_products

    # Return empty arrays if the input list is empty.
    if not data_products:
        return None, None

    for i, product in enumerate(data_products):

        x1d = datamodels.open(product)

        n_spec, n_pix = x1d.spec[0].spec_table.WAVELENGTH.shape
        seg_spec_1D = np.zeros([n_spec, n_pix])
        wave_um = x1d.spec[0].spec_table.WAVELENGTH[0, :]

        for j in range(n_spec):
            seg_spec_1D[j, :] = x1d.spec[0].spec_table.FLUX[j, :]

        if i == 0:
            all_spec_1D = seg_spec_1D
            all_times = x1d.int_times.int_mid_BJD_TDB
        if i > 0:
            all_spec_1D = np.concatenate((all_spec_1D, seg_spec_1D), axis=0)
            all_times = np.concatenate((all_times,
                                        x1d.int_times.int_mid_BJD_TDB),
                                       axis=0)

    # We also trim several columns at the start and end of the spectra.
    # These belong to the reference pixels and are marked 'nan'.
    print("Trimming first/last 5 reference pixels with nan-values ...")
    all_spec_1D = all_spec_1D[:, 5:-5]
    wave_um = wave_um[5:-5]

    return all_spec_1D, all_times, wave_um


def determine_figure_properties(xdim, ydim, threshold=0.15, maxsize=8):
    """Based on the dimensions of the 2D data, determine which aspect
    to use for plt.imshow. Arrays that are far from square will use "auto",
    in which pixels are allowed to be non-square, and the array is manipulated
    to fill the figure. Arrays that are close to square will use "equal", which
    enforces square pixels.

    Default colorbar orientation is vertical, to the right
    of the array. But if the array is wider than it is tall, then the colorbar
    moves to the horizontal orentation, below the array.

    Also provide a default figure size. This may be overridden later, especially
    depending on the aspect value.

    Parameters
    ----------
    xdim : int
        Number of columns in the data array

    ydim : int
        Number of rows in the data array

    threshold : float
        Threshold value for the ratio of xdim/ydim. Ratio values below
        this value, or above the inverse of this value, will cause the
        aspect value to be 'auto'

    maxsize : float
        Length of the largest dimension, in units accepted by plt.subplots

    Returns
    -------
    aspect : str
        Aspect type for the array
    """
    aspect_value = determine_aspect(xdim, ydim, threshold=threshold)
    cbar_orient = determine_cbar_orient(xdim, ydim)
    figsize_value = determine_default_figsize(xdim, ydim, maxsize=maxsize)
    return aspect_value, cbar_orient, figsize_value


def determine_aspect(xdim, ydim, threshold=0.15):
    """Based on the dimensions of the 2D data, determine which aspect
    to use for plt.imshow. Arrays that are far from square will use "auto",
    in which pixels are allowed to be non-square, and the array is manipulated
    to fill the figure. Arrays that are close to square will use "equal", which
    enforces square pixels. Default colorbar orientation is vertical, to the right
    of the array. But if the array is wider than it is tall, then the colorbar
    moves to the horizontal orentation, below the array.

    Parameters
    ----------
    xdim : int
        Number of columns in the data array

    ydim : int
        Number of rows in the data array

    threshold : float
        Threshold value for the ratio of xdim/ydim. Ratio values below
        this value, or above the inverse of this value, will cause the
        aspect value to be 'auto'

    Returns
    -------
    aspect : str
        Aspect type for the array
    """
    dim_ratio = xdim / ydim
    if dim_ratio < threshold or dim_ratio > (1 / threshold):
        aspect = 'auto'
    else:
        aspect = 'equal'
    return aspect


def determine_cbar_orient(xdim, ydim):
    """Determine whether the colorbar should be beneath the array or to the right. Default
    value is to the right. Let's keep it there unless the array starts to become significantly
    non-square (although with a lower threshold than the aspect function uses.)

    Parameters
    ----------
    xdim : int
        Number of columns in the data array

    ydim : int
        Number of rows in the data array

    Returns
    -------
    orient : str
        Orientation
    """
    dim_ratio = xdim / ydim
    if dim_ratio <= 1.15:
        # X-dimension is smaller
        orient = "vertical"
    else:
        orient = "horizontal"
    return orient


def determine_default_figsize(xdim, ydim, maxsize=8):
    """Determine the default figure size, given the array dimensions.

    Parameters
    ----------
    xdim : int
        Number of columns in the data array

    ydim : int
        Number of rows in the data array

    maxsize : float
        Length of the largest dimension, in units accepted by plt.subplots

    Returns
    -------
    figsize : tup
        (xsize, ysize)
    """
    if xdim >= ydim:
        figsize = (maxsize, maxsize * ydim / xdim)
    else:
        figsize = (maxsize * xdim / ydim, maxsize)

    return figsize

def Fnu_to_Flam(wave_micron, flux_jansky):
    """Convert Jansky flux units to erg/s/cm^2/Angstrom with an input wavelength in microns

    Parameters
    ----------
    wave_micron : numpy.ndarray
        Array of wavelengths in microns

    flux_jansky : numpy.ndarray
        Array of fluxes in Jansky

    Returns
    -------
    f_lambda : numpy.ndarray
        Array of f_lambda values
    """
    f_lambda = 1E-21 * flux_jansky * (const.c.value) / (wave_micron**2) # erg/s/cm^2/Angstom
    return f_lambda


def formatted_tick_labels(tick_vals):
    """Given an array of tick values, create formatted tick labels based on the value of
    the numbers.

    Parameters
    ----------
    tick_vals : numpy.ndarray
        Array of tick values (floats)

    Returns
    -------
    tick_str : list
        List of string tick labels
    """
    delta = tick_vals[-1] - tick_vals[0]
    if delta >= 100:
        dig = 0
    elif ((delta < 100) & (delta >= 10)):
        dig = 1
    elif ((delta < 10) & (delta >= 1)):
        dig = 2
    elif delta < 1:
        dig = 3
    else:
        dig = 2
    format_string = "%.{}f".format(dig)
    tick_str = [format_string % number for number in tick_vals]
    return tick_str


def hide_axes(fig):
    """Hide all axes and labels for the given figure. This helps when making a thumbnail
    image.

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        Matplotlib figure instance

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Modified figure instance
    """
    for ax in fig.get_axes():
        ax.set_axis_off()              # hides everything (ticks, labels, spines)
        ax.set_title("")               # clears title if any
        #ax.set_xlabel("")              # clears x label
        #ax.set_ylabel("")              # clears y label
    return fig


def sigma_clip_ignore_nan(data, sigma=3):
    """
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning, append=True)
        # Filter the warning specifically for astropy.stats.sigma_clipping
        warnings.filterwarnings("ignore", message="Input data contains invalid values", module="astropy.stats.sigma_clipping")
        clp = sigma_clip(data, sigma=sigma)
    return clp

def shift_data_get_ticks(image, minval, maxval, num_ticks=5):
    """Given a multi-dimensional array along with the minimum and maximum values for the colorbar in imshow(),
    along with the desired number of ticks to add to the colorbar, shift the data in the image such that the
    minimum pixel value is 1. Then calculate values for the number of tick marks. Note that the tick mark values
    here will be in the shifted data space. Then calculate the corresponding tick values in the unshifted data space.

    Parameters
    ----------
    image : numpy.ndarray
        Multidimensional array

    minval : float
        Minimum value for the colormap to use in imshow

    maxval : float
        Maximum value for the colormap to use in imshow

    num_ticks : int
        Number of tick values to create

    Returns
    -------
    shifted_image : numpy.ndarray
        ``image``, shifted such that the minimum pixel value is 1.0. Primarily intended
        for log scaling in imshow

    shifted_min : float
        Minimum pixel value in the shifted data

    shifted_max : float
        Maximum pixel value in the shifted data

    shifted_tickvals : numpy.ndarray
        Array of tick values in the shifted data space

    unshifted_tickvals : numpy.ndarray
        Array of tick values in the original data (``image``) data space
    """
    shifted_image = image - minval + 1
    shifted_min = 1
    shifted_max = maxval - minval + 1

    # Generate tick labels
    shifted_tickvals = np.logspace(np.log10(shifted_min), np.log10(shifted_max), num_ticks)
    unshifted_tickvals =  shifted_tickvals + minval - 1
    return shifted_image, shifted_min, shifted_max, shifted_tickvals, unshifted_tickvals
