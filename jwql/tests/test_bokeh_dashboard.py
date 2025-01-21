#!/usr/bin/env python

"""Tests for the ``bokeh_dashboard`` module in the ``jwql`` web
application.

Authors
-------

    - Bryan Hilbert

Use
---

    These tests can be run via the command line (omit the -s to
    suppress verbose output to stdout):

    ::

        pytest -s test_bokeh_dashboard.py
"""

import os

from django import setup
import pandas as pd
import pytest

from jwql.utils.constants import DEFAULT_MODEL_CHARFIELD, ON_GITHUB_ACTIONS, ON_READTHEDOCS

# Skip testing this module if on Github Actions
if not ON_GITHUB_ACTIONS and not ON_READTHEDOCS:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "jwql.website.jwql_proj.settings")
    setup()
    from jwql.website.apps.jwql import bokeh_dashboard  # noqa: E402 (module level import not at top of file)


@pytest.mark.skipif(ON_GITHUB_ACTIONS, reason='Requires access to django models.')
def test_build_table_latest_entry():
    tab = bokeh_dashboard.build_table('FilesystemCharacteristics')
    assert isinstance(tab, pd.DataFrame)
    assert len(tab['date']) > 0
