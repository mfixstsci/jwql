#! /usr/bin/env python

"""
General tools for working with multistripe data
"""

from pathlib import Path
import shutil

from astropy.io import fits
import numpy as np


def reconstruct(data, header, sub256_frame=None):
    """
    Reconstruct SOSS SUBSTRIP256 subarray from multistripe data

    Parameters
    ----------
    data : numpy.array
        4D array of data as it appears in the original uncal file

    header : astropy.io.fits.header
        Header object from the SCI extension of the original uncal file

    sub256_frame :

    Returns
    -------
    final : numpy.array
        Array containing re-arranged uncal data
    """
    nints, ngrps, y, x = data.shape
    nsuper = header['SSTR_NST']
    new_nints = int(nints / nsuper)
    final = np.zeros((new_nints, ngrps, y, 2048))
    superstep = header['SSTR_STP']
    reads1 = header['MSTR_RD1']
    skip1 = header['MSTR_SK1']

    # Starting indexes of each stripe integration
    starts = np.arange(nints)[::nsuper]

    # First read the RIGHTMOST reference pixels (skips the refpixel column at idx=2044 for SUB17STRIPE)
    final[:, :, :, -reads1:] = data[starts, :, :, -reads1:]

    # Iterate over start indexes
    for nint, start in enumerate(np.arange(0, nints, nsuper)):
        for stripe in range(nsuper):
            source = start + (nsuper - 1 - stripe)
            final[nint, :, :, (stripe * superstep) + reads1 + skip1:(stripe * superstep) + superstep + reads1 + skip1] = (data[source, :, :, :-reads1])

    return final

def save_new_soss_uncal_file(input_file, outdir='data/'):
    """
    Reconstruct the SOSS superstripe data into SUBSTRIP256 subarrays and save them in new uncal files

    Parameters
    ----------
    input_file : str
        Input FITS filename

    outdir : str
        Directory into which the output is saved
    """
    input_file = Path(input_file)
    outdir = Path(outdir)
    output_file = outdir / (f"{input_file.stem}_reconstructed{input_file.suffix}")

    # Copy original file
    shutil.copy2(input_file, output_file)

    # Modify copy in place
    with fits.open(output_file, mode="update") as hdul:

        # Reconstruct SUBSTRIP256
        hdr = hdul[0].header
        new_data = reconstruct(hdul["SCI"].data, hdr)
        hdul["SCI"].data = new_data

        # Exposure information
        hdr["EXP_TYPE"] = "NIS_SOSS"

        # Integration count
        hdr["NINTS"] = new_data.shape[0]

        # Subarray information
        hdr["SUBARRAY"] = "SUBSTRIP256"
        hdr["SUBSIZE1"] = 2048
        hdr["SUBSIZE2"] = new_data.shape[2]

        # SCI extension dimensions
        sci_hdr = hdul["SCI"].header
        sci_hdr["NAXIS1"] = 2048
        sci_hdr["NAXIS2"] = new_data.shape[2]

        hdul.flush()

    return str(output_file)


def soss_uncal_multistripe_check(hdulist):
    """Check if the given hdulist is from an uncal file of a SOSS multistripe exposure

    Parameters
    ----------
    hdulist : astropy.io.fits.HDUList
        hdulist from a JWST fits file

    Returns
    -------
    hdulist : astropy.io.fits.HDUList
        If hdulist is from a SOSS multistripe exposure, return hdulist of a
        rearranged file that will produce better preview images
    """
    # We only need to rearragne uncal files
    if (('uncal' in hdulist[0].header['FILENAME']) & ('STRIPE_SOSS' in hdulist[0].header['SUBARRAY'])):

        # Reconstruct SUBSTRIP256
        new_data = reconstruct(hdulist["SCI"].data, hdulist[0].header)
        hdulist["SCI"].data = new_data

        # Exposure information
        hdulist[0].header["EXP_TYPE"] = "NIS_SOSS"

        # Integration count
        hdulist[0].header["NINTS"] = new_data.shape[0]

        # Subarray information
        hdulist[0].header["SUBARRAY"] = "SUBSTRIP256"
        hdulist[0].header["SUBSIZE1"] = 2048
        hdulist[0].header["SUBSIZE2"] = new_data.shape[2]

        # SCI extension dimensions
        hdulist["SCI"].header["NAXIS1"] = 2048
        hdulist["SCI"].header["NAXIS2"] = new_data.shape[2]

    return hdulist

