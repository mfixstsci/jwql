#! /usr/bin/env python
"""Tests for the ``edb/utils.py`` module.

Authors
-------

    - Bryan Hilbert


Use
---

    These tests can be run via the command line (omit the ``-s`` to
    suppress verbose output to ``stdout``):

    ::

        pytest -s test_edb_utils.py
"""

from jwql.edb import utils

def test_get_oss_log_messages():
    visitid = '01068001001'
    logs = utils.get_oss_log_messages(visitid=visitid)

    assert len(logs) == 76
    assert logs['EVENT_MSG'][-1] == 'VISIT V01068001001 ENDED'


def test_get_visitid():
    visitid = '01068001001'
    vid = utils.get_visitid(visitid)
    assert vid == 'V01068001001'

    visitid = '1068:1:1'
    vid = utils.get_visitid(visitid)
    assert vid == 'V01068001001'


def test_query_program_visit_times():
    visits = utils.query_program_visit_times('1068')
    assert len(visits) == 21
    assert visits[0]['start_mjd'].value == 59714.551588773145
    assert visits[-1]['start_mjd'].value == 59714.67325975694


def test_query_visit_time():
    visitid = '01068001001'
    starting, ending = utils.query_visit_time(visitid)
    assert starting.value == '2022-05-15 13:14:17.270'
    assert ending.value == '2022-05-15 14:15:56.980'


def test___query_program_visit_times_by_inst():
    info = utils._query_program_visit_times_by_inst(1068, 'NIRCam')
    assert info[0] == ('V01068004001', 59714.62667017361, 59714.63190871528, 'NIRCam')
    assert info[-1] == ('V01068003001', 59714.618445023145, 59714.6266346412, 'NIRCam')


