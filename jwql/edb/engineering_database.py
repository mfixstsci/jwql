#! /usr/bin/env python
"""Module for dealing with JWST DMS Engineering Database mnemonics.

This module provides ``jwql`` with convenience classes and functions
to retrieve and manipulate mnemonics from the JWST DMS EDB. It uses
the `edb_interface` module to interface the EDB directly.

Authors
-------

    - Johannes Sahlmann

Use
---

    This module can be imported and used with

    ::

        from jwql.edb.engineering_database import get_mnemonic
        get_mnemonic(mnemonic_identifier, start_time, end_time)

    Required arguments:

    ``mnemonic_identifier`` - String representation of a mnemonic name.
    ``start_time`` - astropy.time.Time instance
    ``end_time`` - astropy.time.Time instance

Notes
-----
    A valid MAST authentication token has to be present in the local
    ``jwql`` configuration file (config.json).

"""

from collections import OrderedDict

from astropy.time import Time
from bokeh.embed import components
from bokeh.plotting import figure
import numpy as np

from jwql.utils.utils import get_config
from .edb_interface import query_single_mnemonic, query_mnemonic_info

# should use oauth.register_oauth()?
settings = get_config()
MAST_TOKEN = settings['mast_token']


class EdbMnemonic:
    """Class to hold and manipulate results of DMS EngDB queries."""

    def __init__(self, mnemonic_identifier, start_time, end_time, data, meta, info):
        """Populate attributes.

        Parameters
        ----------
        mnemonic_identifier : str
            Telemetry mnemonic identifier
        start_time : astropy.time.Time instance
            Start time
        end_time : astropy.time.Time instance
            End time
        data : astropy.table.Table
            Table representation of the returned data.
        meta : dict
            Additional information returned by the query
        info : dict
            Auxiliary information on the mnemonic (description,
            category, unit)

        """
        self.mnemonic_identifier = mnemonic_identifier
        self.start_time = start_time
        self.end_time = end_time
        self.data = data
        self.meta = meta
        self.info = info

    def __str__(self):
        """Return string describing the instance."""
        return 'EdbMnemonic {} with {} records between {} and {}'.format(
            self.mnemonic_identifier, len(self.data), self.start_time.isot,
            self.end_time.isot)

    def interpolate(self, times, **kwargs):
        """Interpolate value at specified times."""
        raise NotImplementedError

    def bokeh_plot(self):
        """Make basic bokeh plot showing value as a function of time.

        Returns
        -------
        [div, script] : list
            List containing the div and js representations of figure.

        """
        abscissa = Time(self.data['MJD'], format='mjd').datetime
        ordinate = self.data['euvalue']

        p1 = figure(tools='pan,box_zoom,reset,wheel_zoom,save', x_axis_type='datetime',
                    title=self.mnemonic_identifier, x_axis_label='Time',
                    y_axis_label='Value ({})'.format(self.info['unit']))
        p1.line(abscissa, ordinate, line_width=1, line_color='blue', line_dash='dashed')
        p1.circle(abscissa, ordinate, color='blue')

        script, div = components(p1)

        return [div, script]


def get_mnemonic(mnemonic_identifier, start_time, end_time):
    """Execute query and return a EdbMnemonic instance."""
    data, meta, info = query_single_mnemonic(mnemonic_identifier, start_time, end_time,
                                             token=MAST_TOKEN)

    # create and return instance
    mnemonic = EdbMnemonic(mnemonic_identifier, start_time, end_time, data, meta, info)
    return mnemonic


def get_mnemonics(mnemonics, start_time, end_time):
    """Query DMS EDB with a list of mnemonics and a time interval.

    Parameters
    ----------
    mnemonics : list or numpy.ndarray
        Telemetry mnemonic identifiers, e.g. ['SA_ZFGOUTFOV',
        'IMIR_HK_ICE_SEC_VOLT4']
    start_time : astropy.time.Time instance
        Start time
    end_time : astropy.time.Time instance
        End time

    Returns
    -------
    mnemonic_dict : dict
        Dictionary. keys are the queried mnemonics, values are
        instances of EdbMnemonic

    """
    if not isinstance(mnemonics, (list, np.ndarray)):
        raise RuntimeError('Please provide a list/array of mnemonic_identifiers')

    mnemonic_dict = OrderedDict()
    for mnemonic_identifier in mnemonics:
        # fill in dictionary
        mnemonic_dict[mnemonic_identifier] = get_mnemonic(mnemonic_identifier, start_time, end_time)

    return mnemonic_dict


def get_mnemonic_info(mnemonic_identifier):
    """Return the mnemonic description.

    Parameters
    ----------
    mnemonic_identifier : str
        Telemetry mnemonic identifier, e.g. ``SA_ZFGOUTFOV``

    Returns
    -------
    info : dict
        Object that contains the returned data

    """
    return query_mnemonic_info(mnemonic_identifier, token=MAST_TOKEN)
