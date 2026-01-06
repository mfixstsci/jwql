#! /usr/bin/env python

"""Generally useful funtions related to the engineering database (EDB)

Authors
-------

    - Jonathan Aguilar
    - Marshall Perrin
    - Bryan Hilbert

Use
---

    Functions in this module can be imported and used in this way:

    ::

        from jwql.edb.utils import get_ta_centroids
        get_ta_centroids(filename)

Notes
-----
    Code for querying MAST for ICTM log entries was copied from the
    misc_jwst repo, authored by Marshall Perrin. We could alternatively
    add misc_jwst to the list of JWQL dependencies.
"""
import os
import re
import warnings

from datetime import datetime, timedelta, timezone
from requests import Session

import astropy
from astropy import time
from astropy.io import fits
from astroquery.mast import Mast, Observations
from jwst.lib.engdb_mast import EngdbMast

from jwql.utils.constants import INSTRUMENT_SERVICE_MATCH, JWST_INSTRUMENT_NAMES_MIXEDCASE, JWST_INSTRUMENT_NAMES_SHORTHAND
from jwql.utils.utils import get_config



def get_oss_log_messages(visitid=None, start_time=None, end_time=None):
    """ Retrieve OSS event log messages during a given visit or time interval

    See also get_ictm_event_log. This function instead uses the EngDB interface included in the JWST pipeline.
    Returns an astropy Table with the EngDB message, message ID, and message source.

    Parameters
    ----------
    visitid : str
        Visit ID string, like 'V01234001001'
    start_time : str
        Query start time, in ISO format, e.g. "2025-04-04T12:00:00"
    end_time : str
        Query end time, in ISO format, e.g. "2025-04-05T12:00:00"

    Returns
    -------
    msg_table : astropy.table.Table
        Timestamps and message values from OSS log
    """
    if start_time is None and end_time is None:
        visitid = get_visitid(visitid)  # Handle either allowed format of visit ID

        #----- When was that visit? -----
        start_time, end_time = query_visit_time(visitid)

        if start_time is None:
            raise RuntimeError(f"Cannot find start time for visit {visitid}. That visit may not have happened yet.")
    else:
        start_time = astropy.time.Time(start_time)
        end_time = astropy.time.Time(end_time)

    #----- Retrieve relevant messages from the ICTM event log stream -----
    MAST_TOKEN = get_config()['mast_token']
    service = EngdbMast(token=MAST_TOKEN)

    # There are multiple mnemonics we care about,
    # in particular the EVENT_MSG has the text, and the MSG_ID and MSG_SRC give metadata on the source
    # Retrieve all of these and organize into a table for convenience.

    msg_times, messages = service.get_values("ICTM_EVENT_MSG", start_time.isot, end_time.isot, include_obstime=True, zip_results=False)
    msg_times_2, msg_ids = service.get_values("ICTM_EVENT_MSG_ID", start_time.isot, end_time.isot, include_obstime=True, zip_results=False)
    msg_times_3, msg_srcs = service.get_values("ICTM_EVENT_MSG_SRC", start_time.isot, end_time.isot, include_obstime=True, zip_results=False)

    #----- Arrange those 3 sets of results into a single Table -----
    # Ideally we should have gotten the same number of rows in all 3 queries above\
    # These -should- all have matching counts and time stamps... but for some reason this is not always the case. Hmm.
    # So check here and if necessary handle the case of an inconsistency.

    if len(messages) == len(msg_ids) and len(messages) == len(msg_srcs):
        msg_table = astropy.table.Table([msg_times, messages, msg_ids, msg_srcs],
                                       names = ['TIME', "EVENT_MSG", "EVENT_MSG_ID", "EVENT_MSG_SRC"])
    else:
        print("Inconsistent number of EVENT_MSG and EVENT_MSG_ID records returned; matching based on telemetry time stamps ")
        # This occurs for instance in visit V07344017001, a NIRCam WFSC visit.

        msg_table = astropy.table.Table([msg_times[0:1], messages[0:1], msg_ids[0:1], msg_srcs[0:1]],
                           names = ['TIME', "EVENT_MSG", "EVENT_MSG_ID", "EVENT_MSG_SRC"])
        # match up the rows that do have consistent timestamps
        # When there's not a match, look 1 row before or after to see if we can find a match
        n = min(len(msg_times), len(msg_times_2), len(msg_times_3))
        index_offset = 0  # We will use this to track offsets between mnemonic time series
        for i in range(1, n):
            # Compare time stamps between EVENT_MSG and EVENT_MSG_ID mnemonic time series
            if msg_times[i] - msg_times_2[i+index_offset] == 0*u.second:
                # times match, no need to adjust
                pass
            else:
                if msg_times[i] == msg_times_2[i+index_offset-1]:
                    #print('found extra EVENT_MSG relative to EVENT_MSG_ID')
                    index_offset -= 1
                elif msg_times[i] == msg_times_2[i+index_offset+1]:
                    #print('found skipped EVENT_MSG relative to EVENT_MSG_ID')
                    index_offset += 1
                else:
                    raise RuntimeError("Inconsistent number of telemetry records returned, with bigger gaps than this function can currently sort out.")
            msg_table.add_row([msg_times[i], messages[i], msg_ids[i+index_offset], msg_srcs[i+index_offset]])

    return msg_table


def get_ta_centroids(filename):
    """Get the centroid location of the TA target, as reported in the EDB

    Parameters
    ----------
    filename : str
        Name of fits file containing TA data
    """
    header = fits.getheader(filename)
    visit_id = header['VISIT_ID']
    instrument = header['INSTRUME'].lower()

    # Get OSS messages for the visit associated with the file
    log_entries = get_oss_log_messages(visitid=visit_id)

    # Extract the TA-related lines
    inst_shorthand = [k.upper() for k, v in JWST_INSTRUMENT_NAMES_SHORTHAND.items() if v == instrument][0]
    ta_idxes = [i for i,row in enumerate(log_entries['EVENT_MSG']) if f'{inst_shorthand}TAMAIN' in row]
    ta_entries = log_entries[ta_idxes[0]: ta_idxes[1] + 1]

    centroid_idxes = [i for i,row in enumerate(ta_entries['EVENT_MSG']) if 'postage-stamp coord (colCen' in row]
    centroids = []
    for centroid_idx in centroid_idxes:
        centroid_str = ta_entries[centroid_idx]['EVENT_MSG'].split('(')[-1][:-1].split(', ')
        centroid = [float(val) for val in centroid_str]
        centroids.append(centroid)

    return centroids


def get_visitid(visitstr):
    """ Common util function to normalize visit specification

    Parameters
    ----------
    visitid : str
        Visit ID. e.g. "01068001001", "V01068001001" or "1068:1:1

    Returns
    -------
    visit_id : str
        Normalized visit value in the format "V01068001001"
    """
    if visitstr.startswith("V"):
        # Full visit ID like V04503031001
        return visitstr
    elif ':' in visitstr:
        # This is PPS format visit ID, like 4503:31:1
        parts = [int(p) for p in visitstr.split(':')]
        if len(parts) == 2:
            # if given only like 4503:31, assume the visit number is 1
            parts.append(1)
        return f"V{parts[0]:05d}{parts[1]:03d}{parts[2]:03d}"
    elif len(visitstr) == 11:
        # Full visit ID but without the leading V, like 04503031001
        return 'V'+visitstr


def query_program_visit_times(program,  verbose=False):
    """ Get the start and end times of all completed visits in a program.

    See also visit_start_end_times for a specifc named visit..

    Parameters
    ----------
    program : int or str
        Program ID
    verbose : bool
        be more verbose in output?

    Returns
    -------
    vis_table : astropy.table.Table
        Columns for visit ID and start and end times.
    """

    # use Observations query interface to find all observations
    obs = Observations.query_criteria(obs_collection=["JWST"], proposal_id=[program])

    # Annoyingly, that query interface doesn't return start/end times,
    # therefore we have to separately call a different query interface for those, per instrument
    instruments = [val.split('/')[0] for val in set(obs['instrument_name'])]
    visit_times = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # Because we expect at least some of these may have a warning about no results found
        for inst in instruments:
            if verbose:
                print(f"querying for visits using {inst}")
            visit_times += _query_program_visit_times_by_inst(program, inst)

    # Format outputs
    vids = [v[0] for v in visit_times]
    starts =astropy.time.Time([float(v[1]) for v in visit_times], format='mjd')
    ends = astropy.time.Time([float(v[2]) for v in visit_times], format='mjd')
    insts = [v[3] for v in visit_times]
    vis_table =  astropy.table.Table([vids, starts, ends, insts],
                                     names=('visit_id', 'start_mjd', 'end_mjd', 'instrument'))
    vis_table.sort(keys='start_mjd')
    return vis_table


def query_visit_time(visitid, verbose=False):
    """ Find start and end time of a visit

    Parameters
    ----------
    visitid : str
        visit id, like 'V01234005001' or '1234:5:1'

    verbose : bool
        Whether to print extra output to the screen

    Returns
    -------
    starting_time : astropy.time.Time
        Start time of the visitid

    ending_time : astropy.time.Time
        Ending time of the visitid
    """
    visitid = get_visitid(visitid)  # Handle either allowed format of visit ID

    # Get table of times for all visits in that program
    program = visitid[1:6]

    visit_times = query_program_visit_times(program, verbose=verbose)

    #  Scan through the table for a row matching that visit id
    for vid, vstart, vend, inst in visit_times:
        if verbose:
            print(vid, visitid, vid==visitid, inst)
        if vid==visitid:
            # Return times as astropy Time objects
            starting_time = astropy.time.Time(vstart, format='mjd')
            ending_time = astropy.time.Time(vend, format='mjd')
            starting_time.format = 'iso'
            ending_time.format = 'iso'
            return starting_time, ending_time
    else:
        return None, None


def _query_program_visit_times_by_inst(program, instrument, verbose=False):
    """ Get the start and end times of all completed visits in a program, per instrument.
    Not intended for general use; this is mostly a helper to query_program_visit_times.

    Getting the vststart_mjd and visitend_mjd fields requires using the instrument keywords
    interface, so one has to specify which instrument ahead of time.

    Parameters
    ----------
    program : int or str
        Program ID
    instrument : str
        instrument name
    verbose : bool
        be more verbose in output?

    Returns
    -------
    visit_times : list
        (visitid, start, end) tuples.

    """
    service = INSTRUMENT_SERVICE_MATCH[JWST_INSTRUMENT_NAMES_MIXEDCASE[instrument.lower()]]

    collist = 'filename, program, observtn, visit_id, vststart_mjd, visitend_mjd, bstrtime'
    all_columns = False

    def set_params(parameters):
        return [{"paramName" : p, "values" : v} for p, v in parameters.items()]

    keywords = {'program': [str(program),]}
    parameters = {'columns': '*' if all_columns else collist,
                  'filters': set_params(keywords)}

    if verbose:
        print("MAST query parameters:")
        print(parameters)

    responsetable = Mast.service_request(service, parameters)
    if 'bstrtime' in collist:
        responsetable.sort(keys='bstrtime')

    visit_times = []
    for row in responsetable:
        visit_times.append( ('V'+row['visit_id'], row['vststart_mjd'], row['visitend_mjd'], instrument))

    visit_times= set(visit_times)
    return list(visit_times)
