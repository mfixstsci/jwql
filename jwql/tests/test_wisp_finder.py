#! /usr/bin/env python

"""Tests for the ``wisp_finder`` module.

Authors
-------

    - Bryan Hilbert

Use
---

    These tests can be run via the command line (omit the ``-s`` to
    suppress verbose output to stdout):
    ::

        pytest -s test_wisp_finder.py
"""

import datetime
import os
import pytest

from django import setup
import numpy as np
import torch
import torchvision

from jwql.instrument_monitors.nircam_monitors import wisp_finder, prepare_wisp_pngs
from jwql.utils.constants import ON_GITHUB_ACTIONS, ON_READTHEDOCS
from jwql.utils.utils import get_config
from jwql.website.apps.jwql.archive_database_update import files_in_filesystem

if not ON_GITHUB_ACTIONS and not ON_READTHEDOCS:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "jwql.website.jwql_proj.settings")
    setup()
    from jwql.website.apps.jwql.models import RootFileInfo


@pytest.mark.skipif(ON_GITHUB_ACTIONS, reason='Requires access to database.')
def test_add_wisp_flag():
    """Test that the wisp flag is successfully set on a given rootfileinfo
    """
    basename = 'jw01068001001_02102_00003_nrcb4'
    wisp_finder.add_wisp_flag(basename)

    root_file_info = RootFileInfo.objects.get(root_name=basename)
    assert root_file_info.anomalies.wisps is True

    if root_file_info.anomalies.wisps is True:
        # If the flag was checked and successfully set, return it back
        # to False for future tests
        root_file_info.anomalies.wisps = False
        root_file_info.anomalies.save(update_fields=['wisps'])


def test_create_transform():
    """Test that the pytorch transform is successfully created
    """
    transform = wisp_finder.create_transform()
    assert isinstance(transform, torchvision.transforms.transforms.Compose)


@pytest.mark.skipif(ON_GITHUB_ACTIONS, reason='Requires access to central store.')
def test_load_ml_model():
    """Test that a file containing a saved ML model can be successfully loaded
    """
    modelname = get_config()['wisp_finder_ML_model']
    model = wisp_finder.load_ml_model(modelname)
    assert isinstance(model.fc, torch.nn.modules.linear.Linear)


@pytest.mark.skipif(ON_GITHUB_ACTIONS, reason='Requires access to central store')
def test_predict_wisp():
    modelname = get_config()['wisp_finder_ML_model']
    model = wisp_finder.load_ml_model(modelname)
    transform = wisp_finder.create_transform()

    fits_file = ['jw01068004001_02101_00001_nrcb4_rate.fits']
    filepath_public = files_in_filesystem(fits_file, 'public')
    copied_file = wisp_finder.copy_files_to_working_dir(filepath_public)
    working_dir = get_config()["working"]
    png_filename = prepare_wisp_pngs.run(copied_file[0], out_dir=working_dir)
    prediction = wisp_finder.predict_wisp(model, png_filename, transform)
    assert isinstance(prediction, str)
    assert prediction in ['wisp', 'no wisp']
    os.remove(png_filename)
    os.remove(os.path.join(working_dir, copied_file[0]))


def test_query_mast():
    """Test that a MAST query returns the expected data
    """
    results = wisp_finder.query_mast(59714.625, 59714.6458)
    assert results == ['jw01068004001_02101_00001_nrcb4_rate.fits']


def test_remove_duplicate_files():
    """Test that duplicate instances of a given file are removed
    """
    files = ['/location/one/jw01068001001_02101_00001_nrcb4_rate.fits',
             '/location/one/jw01068001001_05101_00001_nrcb4_rate.fits',
             '/location/one/jw01068001001_03101_00001_nrcb4_rate.fits',
             '/location/two/jw01068001001_03101_00001_nrcb4_rate.fits',
             '/location/two/jw01068001001_02101_00001_nrcb4_rate.fits',
             '/location/one/jw01068001001_09101_00001_nrcb4_rate.fits'
             ]
    unique_files = sorted(wisp_finder.remove_duplicate_files(files))
    assert unique_files == ['/location/one/jw01068001001_02101_00001_nrcb4_rate.fits',
                            '/location/one/jw01068001001_03101_00001_nrcb4_rate.fits',
                            '/location/one/jw01068001001_05101_00001_nrcb4_rate.fits',
                            '/location/one/jw01068001001_09101_00001_nrcb4_rate.fits'
                            ]


def test_rescale_array():
    """Test that an input array is correctly rescaled
    """
    arr = np.random.normal(0, 1, size=(100, 100))
    arr[3, 3] = 10.
    rescaled = prepare_wisp_pngs.rescale_array(arr)
    assert rescaled[3, 3] == 255


def test_resize_image():
    """Test image resizing
    """
    img = np.zeros((500, 500))
    resized = prepare_wisp_pngs.resize_image(img)
    assert resized.size == (256, 256)
