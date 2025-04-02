#! /usr/bin/env python

"""Tests for the ``archive_database_update`` module.

Authors
-------

    - Bryan Hilbert

Use
---

    These tests can be run via the command line (omit the ``-s`` to
    suppress verbose output to stdout):
    ::

        pytest -s test_archive_database_update.py
"""


import pytest

from jwql.website.apps.jwql import archive_database_update


def test_filter_rootnames():
    """Test the filtering of source-based level 2 files
    """
    files = ['jw06434-c1021_s000001510_nircam_f444w-grismr.fits',
             'jw01068004001_02102_00001_nrcb4_rate.fits',
             'jw06434-c1021_t000_nircam_clear-f090w_segm.fits',
             'jw06434-o001_t000_nircam_clear-f090w_segm.fits',
             'jw02183117001_03103_00001-seg001_nrca1_rate.fits']

    filtered = archive_database_update.filter_rootnames(files)
    expected = ['jw01068004001_02102_00001_nrcb4_rate.fits',
                'jw02183117001_03103_00001-seg001_nrca1_rate.fits']
    assert filtered == expected
