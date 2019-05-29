"""Various functions to collect data to be used by the ``views`` of the
``jwql`` app.

This module contains several functions that assist in collecting and
producing various data to be rendered in ``views.py`` for use by the
``jwql`` app.

Authors
-------

    - Lauren Chambers
    - Matthew Bourque

Use
---

    The functions within this module are intended to be imported and
    used by ``views.py``, e.g.:

    ::
        from .data_containers import get_proposal_info
"""

import copy
import glob
import os
import re
import tempfile

from astropy.io import fits
from astropy.time import Time
from django.conf import settings
import numpy as np

# astroquery.mast import that depends on value of auth_mast
# this import has to be made before any other import of astroquery.mast
from jwql.utils.utils import get_config, filename_parser, check_config
check_config('auth_mast')
auth_mast = get_config()['auth_mast']
mast_flavour = '.'.join(auth_mast.split('.')[1:])
from astropy import config
conf = config.get_config('astroquery')
conf['mast'] = {'server': 'https://{}'.format(mast_flavour)}
from astroquery.mast import Mast
from jwedb.edb_interface import mnemonic_inventory

from jwql.edb.engineering_database import get_mnemonic, get_mnemonic_info
from jwql.instrument_monitors.miri_monitors.data_trending import dashboard as miri_dash
from jwql.instrument_monitors.nirspec_monitors.data_trending import dashboard as nirspec_dash
from jwql.jwql_monitors import monitor_cron_jobs
from jwql.utils.utils import ensure_dir_exists
from jwql.utils.constants import MONITORS, JWST_INSTRUMENT_NAMES_MIXEDCASE
from jwql.utils.preview_image import PreviewImage
from jwql.utils.credentials import get_mast_token
from .forms import MnemonicSearchForm, MnemonicQueryForm, MnemonicExplorationForm


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
FILESYSTEM_DIR = os.path.join(get_config()['jwql_dir'], 'filesystem')
PREVIEW_IMAGE_FILESYSTEM = os.path.join(get_config()['jwql_dir'], 'preview_images')
THUMBNAIL_FILESYSTEM = os.path.join(get_config()['jwql_dir'], 'thumbnails')
PACKAGE_DIR = os.path.dirname(__location__.split('website')[0])
REPO_DIR = os.path.split(PACKAGE_DIR)[0]


def data_trending():
    """Container for Miri datatrending dashboard and components

    Returns
    -------
    variables : int
        nonsense
    dashboard : list
        A list containing the JavaScript and HTML content for the
        dashboard
    """
    dashboard, variables = miri_dash.data_trending_dashboard()

    return variables, dashboard


def nirspec_trending():
    """Container for Miri datatrending dashboard and components

    Returns
    -------
    variables : int
        nonsense
    dashboard : list
        A list containing the JavaScript and HTML content for the
        dashboard
    """
    dashboard, variables = nirspec_dash.data_trending_dashboard()

    return variables, dashboard


def get_acknowledgements():
    """Returns a list of individuals who are acknowledged on the
    ``about`` page.

    The list is generated by reading in the contents of the ``jwql``
    ``README`` file.  In this way, the website will automatically
    update with updates to the ``README`` file.

    Returns
    -------
    acknowledgements : list
        A list of individuals to be acknowledged.
    """

    # Locate README file
    readme_file = os.path.join(REPO_DIR, 'README.md')

    # Get contents of the README file
    with open(readme_file, 'r') as f:
        data = f.readlines()

    # Find where the acknowledgements start
    for i, line in enumerate(data):
        if 'Acknowledgments' in line:
            index = i

    # Parse out the list of individuals
    acknowledgements = data[index + 1:]
    acknowledgements = [item.strip().replace('- ', '').split(' [@')[0].strip()
                        for item in acknowledgements]

    return acknowledgements


def get_all_proposals():
    """Return a list of all proposals that exist in the filesystem.

    Returns
    -------
    proposals : list
        A list of proposal numbers for all proposals that exist in the
        filesystem
    """

    proposals = glob.glob(os.path.join(FILESYSTEM_DIR, '*'))
    proposals = [proposal.split('jw')[-1] for proposal in proposals]
    proposals = [proposal for proposal in proposals if len(proposal) == 5]

    return proposals


def get_dashboard_components():
    """Build and return dictionaries containing components and html
    needed for the dashboard.

    Returns
    -------
    dashboard_components : dict
        A dictionary containing components needed for the dashboard.
    dashboard_html : dict
        A dictionary containing full HTML needed for the dashboard.
    """

    output_dir = get_config()['outputs']
    name_dict = {'': '',
                 'monitor_mast': 'Database Monitor',
                 'monitor_filesystem': 'Filesystem Monitor'}

    # Run the cron job monitor to produce an updated table
    monitor_cron_jobs.status(production_mode=True)

    # Build dictionary of Bokeh components from files in the output directory
    dashboard_components = {}
    for dir_name, _, file_list in os.walk(output_dir):
        monitor_name = os.path.basename(dir_name)

        # Only continue if the dashboard knows how to build that monitor
        if monitor_name in name_dict.keys():
            formatted_monitor_name = name_dict[monitor_name]
            dashboard_components[formatted_monitor_name] = {}
            for fname in file_list:
                if 'component' in fname:
                    full_fname = '{}/{}'.format(monitor_name, fname)
                    plot_name = fname.split('_component')[0]

                    # Generate formatted plot name
                    formatted_plot_name = plot_name.title().replace('_', ' ')
                    for lowercase, mixed_case in JWST_INSTRUMENT_NAMES_MIXEDCASE.items():
                        formatted_plot_name = formatted_plot_name.replace(lowercase.capitalize(),
                                                                          mixed_case)
                    formatted_plot_name = formatted_plot_name.replace('Jwst', 'JWST')
                    formatted_plot_name = formatted_plot_name.replace('Caom', 'CAOM')

                    # Get the div
                    html_file = full_fname.split('.')[0] + '.html'
                    with open(os.path.join(output_dir, html_file), 'r') as f:
                        div = f.read()

                    # Get the script
                    js_file = full_fname.split('.')[0] + '.js'
                    with open(os.path.join(output_dir, js_file), 'r') as f:
                        script = f.read()

                    # Save to dictionary
                    dashboard_components[formatted_monitor_name][formatted_plot_name] = [div, script]

    # Add HTML that cannot be saved as components to the dictionary
    with open(os.path.join(output_dir, 'monitor_cron_jobs', 'cron_status_table.html'), 'r') as f:
        cron_status_table_html = f.read()
    dashboard_html = {}
    dashboard_html['Cron Job Monitor'] = cron_status_table_html

    return dashboard_components, dashboard_html


def get_edb_components(request):
    """Return dictionary with content needed for the EDB page.

    Parameters
    ----------
    request : HttpRequest object
        Incoming request from the webpage

    Returns
    -------
    edb_components : dict
        Dictionary with the required components

    """
    mnemonic_name_search_result = {}
    mnemonic_query_result = {}
    mnemonic_query_result_plot = None
    mnemonic_exploration_result = None

    # If this is a POST request, we need to process the form data
    if request.method == 'POST':

        if 'mnemonic_name_search' in request.POST.keys():
            # authenticate with astroquery.mast if necessary
            logged_in = log_into_mast(request)

            mnemonic_name_search_form = MnemonicSearchForm(request.POST, logged_in=logged_in,
                                                           prefix='mnemonic_name_search')

            if mnemonic_name_search_form.is_valid():
                mnemonic_identifier = mnemonic_name_search_form['search'].value()
                if mnemonic_identifier is not None:
                    mnemonic_name_search_result = get_mnemonic_info(mnemonic_identifier)

            # create forms for search fields not clicked
            mnemonic_query_form = MnemonicQueryForm(prefix='mnemonic_query')
            mnemonic_exploration_form = MnemonicExplorationForm(prefix='mnemonic_exploration')

        elif 'mnemonic_query' in request.POST.keys():
            # authenticate with astroquery.mast if necessary
            logged_in = log_into_mast(request)

            mnemonic_query_form = MnemonicQueryForm(request.POST, logged_in=logged_in,
                                                    prefix='mnemonic_query')

            # proceed only if entries make sense
            if mnemonic_query_form.is_valid():
                mnemonic_identifier = mnemonic_query_form['search'].value()
                start_time = Time(mnemonic_query_form['start_time'].value(), format='iso')
                end_time = Time(mnemonic_query_form['end_time'].value(), format='iso')

                if mnemonic_identifier is not None:
                    mnemonic_query_result = get_mnemonic(mnemonic_identifier, start_time, end_time)
                    mnemonic_query_result_plot = mnemonic_query_result.bokeh_plot()

                    # generate table download in web app
                    result_table = mnemonic_query_result.data

                    # save file locally to be available for download
                    static_dir = os.path.join(settings.BASE_DIR, 'static')
                    ensure_dir_exists(static_dir)
                    file_name_root = 'mnemonic_query_result_table'
                    file_for_download = '{}.csv'.format(file_name_root)
                    path_for_download = os.path.join(static_dir, file_for_download)

                    # add meta data to saved table
                    comments = []
                    comments.append('DMS EDB query of {}:'.format(mnemonic_identifier))
                    for key, value in mnemonic_query_result.info.items():
                        comments.append('{} = {}'.format(key, str(value)))
                    result_table.meta['comments'] = comments
                    comments.append(' ')
                    comments.append('Start time {}'.format(start_time.isot))
                    comments.append('End time   {}'.format(end_time.isot))
                    comments.append('Number of rows {}'.format(len(result_table)))
                    comments.append(' ')
                    result_table.write(path_for_download, format='ascii.fixed_width',
                                       overwrite=True, delimiter=',', bookend=False)
                    mnemonic_query_result.file_for_download = file_for_download

            # create forms for search fields not clicked
            mnemonic_name_search_form = MnemonicSearchForm(prefix='mnemonic_name_search')
            mnemonic_exploration_form = MnemonicExplorationForm(prefix='mnemonic_exploration')

        elif 'mnemonic_exploration' in request.POST.keys():
            mnemonic_exploration_form = MnemonicExplorationForm(request.POST,
                                                                prefix='mnemonic_exploration')
            if mnemonic_exploration_form.is_valid():
                mnemonic_exploration_result, meta = mnemonic_inventory()

                # loop over filled fields and implement simple AND logic
                for field in mnemonic_exploration_form.fields:
                    field_value = mnemonic_exploration_form[field].value()
                    if field_value != '':
                        column_name = mnemonic_exploration_form[field].label

                        # matching indices in table (case-insensitive)
                        index = [
                            i for i, item in enumerate(mnemonic_exploration_result[column_name]) if
                            re.search(field_value, item, re.IGNORECASE)
                        ]
                        mnemonic_exploration_result = mnemonic_exploration_result[index]

                mnemonic_exploration_result.n_rows = len(mnemonic_exploration_result)

                # generate tables for display and download in web app
                display_table = copy.deepcopy(mnemonic_exploration_result)

                # temporary html file,
                # see http://docs.astropy.org/en/stable/_modules/astropy/table/
                tmpdir = tempfile.mkdtemp()
                file_name_root = 'mnemonic_exploration_result_table'
                path_for_html = os.path.join(tmpdir, '{}.html'.format(file_name_root))
                with open(path_for_html, 'w') as tmp:
                    display_table.write(tmp, format='jsviewer')
                mnemonic_exploration_result.html_file_content = open(path_for_html, 'r').read()

                # pass on meta data to have access to total number of mnemonics
                mnemonic_exploration_result.meta = meta

                # save file locally to be available for download
                static_dir = os.path.join(settings.BASE_DIR, 'static')
                ensure_dir_exists(static_dir)
                file_for_download = '{}.csv'.format(file_name_root)
                path_for_download = os.path.join(static_dir, file_for_download)
                display_table.write(path_for_download, format='ascii.fixed_width',
                                    overwrite=True, delimiter=',', bookend=False)
                mnemonic_exploration_result.file_for_download = file_for_download

                if mnemonic_exploration_result.n_rows == 0:
                    mnemonic_exploration_result = 'empty'

            # create forms for search fields not clicked
            mnemonic_name_search_form = MnemonicSearchForm(prefix='mnemonic_name_search')
            mnemonic_query_form = MnemonicQueryForm(prefix='mnemonic_query')

    else:
        mnemonic_name_search_form = MnemonicSearchForm(prefix='mnemonic_name_search')
        mnemonic_query_form = MnemonicQueryForm(prefix='mnemonic_query')
        mnemonic_exploration_form = MnemonicExplorationForm(prefix='mnemonic_exploration')

    edb_components = {'mnemonic_query_form': mnemonic_query_form,
                      'mnemonic_query_result': mnemonic_query_result,
                      'mnemonic_query_result_plot': mnemonic_query_result_plot,
                      'mnemonic_name_search_form': mnemonic_name_search_form,
                      'mnemonic_name_search_result': mnemonic_name_search_result,
                      'mnemonic_exploration_form': mnemonic_exploration_form,
                      'mnemonic_exploration_result': mnemonic_exploration_result}

    return edb_components


def get_expstart(rootname):
    """Return the exposure start time (``expstart``) for the given
    group of files.

    The ``expstart`` is gathered from a query to the
    ``astroquery.mast`` service.

    Parameters
    ----------
    rootname : str
        The rootname of the observation of interest (e.g.
        ``jw86700006001_02101_00006_guider1``).

    Returns
    -------
    expstart : float
        The exposure start time of the observation (in MJD).
    """

    return 5000.00


def get_filenames_by_instrument(instrument):
    """Returns a list of paths to files that match the given
    ``instrument``.

    Parameters
    ----------
    instrument : str
        The instrument of interest (e.g. `FGS`).

    Returns
    -------
    filepaths : list
        A list of full paths to the files that match the given
        instrument.
    """

    # Query files from MAST database
    # filepaths, filenames = DatabaseConnection('MAST', instrument=instrument).\
    #     get_files_for_instrument(instrument)

    # Find all of the matching files in filesytem
    # (TEMPORARY WHILE THE MAST STUFF IS BEING WORKED OUT)
    instrument_match = {'FGS': 'guider',
                        'MIRI': 'mir',
                        'NIRCam': 'nrc',
                        'NIRISS': 'nis',
                        'NIRSpec': 'nrs'}
    search_filepath = os.path.join(FILESYSTEM_DIR, '*', '*.fits')
    filepaths = [f for f in glob.glob(search_filepath) if instrument_match[instrument] in f]

    return filepaths


def get_filenames_by_proposal(proposal):
    """Return a list of filenames that are available in the filesystem
    for the given ``proposal``.

    Parameters
    ----------
    proposal : str
        The one- to five-digit proposal number (e.g. ``88600``).

    Returns
    -------
    filenames : list
        A list of filenames associated with the given ``proposal``.
    """

    proposal_string = '{:05d}'.format(int(proposal))
    filenames = sorted(glob.glob(os.path.join(
        FILESYSTEM_DIR, 'jw{}'.format(proposal_string), '*')))
    filenames = [os.path.basename(filename) for filename in filenames]

    return filenames


def get_filenames_by_rootname(rootname):
    """Return a list of filenames available in the filesystem that
    are part of the given ``rootname``.

    Parameters
    ----------
    rootname : str
        The rootname of interest (e.g. ``jw86600008001_02101_00007_guider2``).

    Returns
    -------
    filenames : list
        A list of filenames associated with the given ``rootname``.
    """

    proposal = rootname.split('_')[0].split('jw')[-1][0:5]
    filenames = sorted(glob.glob(os.path.join(
        FILESYSTEM_DIR,
        'jw{}'.format(proposal),
        '{}*'.format(rootname))))
    filenames = [os.path.basename(filename) for filename in filenames]

    return filenames


def get_header_info(file):
    """Return the header information for a given ``file``.

    Parameters
    ----------
    file : str
        The name of the file of interest.

    Returns
    -------
    header : str
        The primary FITS header for the given ``file``.
    """

    dirname = file[:7]
    fits_filepath = os.path.join(FILESYSTEM_DIR, dirname, file)
    header = fits.getheader(fits_filepath, ext=0).tostring(sep='\n')

    return header


def get_image_info(file_root, rewrite):
    """Build and return a dictionary containing information for a given
    ``file_root``.

    Parameters
    ----------
    file_root : str
        The rootname of the file of interest.
    rewrite : bool
        ``True`` if the corresponding JPEG needs to be rewritten,
        ``False`` if not.

    Returns
    -------
    image_info : dict
        A dictionary containing various information for the given
        ``file_root``.
    """

    # Initialize dictionary to store information
    image_info = {}
    image_info['all_jpegs'] = []
    image_info['suffixes'] = []
    image_info['num_ints'] = {}

    preview_dir = os.path.join(get_config()['jwql_dir'], 'preview_images')

    # Find all of the matching files
    dirname = file_root[:7]
    search_filepath = os.path.join(FILESYSTEM_DIR, dirname, file_root + '*.fits')
    image_info['all_files'] = glob.glob(search_filepath)

    for file in image_info['all_files']:

        # Get suffix information
        suffix = os.path.basename(file).split('_')[4].split('.')[0]
        image_info['suffixes'].append(suffix)

        # Determine JPEG file location
        jpg_dir = os.path.join(preview_dir, dirname)
        jpg_filename = os.path.basename(os.path.splitext(file)[0] + '_integ0.jpg')
        jpg_filepath = os.path.join(jpg_dir, jpg_filename)

        # Check that a jpg does not already exist. If it does (and rewrite=False),
        # just call the existing jpg file
        if os.path.exists(jpg_filepath) and not rewrite:
            pass

        # If it doesn't, make it using the preview_image module
        else:
            if not os.path.exists(jpg_dir):
                os.makedirs(jpg_dir)
            im = PreviewImage(file, 'SCI')
            im.output_directory = jpg_dir
            im.make_image()

        # Record how many integrations there are per filetype
        search_jpgs = os.path.join(preview_dir, dirname,
                                   file_root + '_{}_integ*.jpg'.format(suffix))
        num_jpgs = len(glob.glob(search_jpgs))
        image_info['num_ints'][suffix] = num_jpgs

        image_info['all_jpegs'].append(jpg_filepath)

    return image_info


def get_instrument_proposals(instrument):
    """Return a list of proposals for the given instrument

    Parameters
    ----------
    instrument : str
        Name of the JWST instrument

    Returns
    -------
    proposals : list
        List of proposals for the given instrument
    """

    service = "Mast.Jwst.Filtered.{}".format(instrument)
    params = {"columns": "program",
              "filters": []}
    response = Mast.service_request_async(service, params)
    results = response[0].json()['data']
    proposals = list(set(result['program'] for result in results))

    return proposals


def get_preview_images_by_instrument(inst):
    """Return a list of preview images available in the filesystem for
    the given instrument.

    Parameters
    ----------
    inst : str
        The instrument of interest (e.g. ``NIRCam``).

    Returns
    -------
    preview_images : list
        A list of preview images available in the filesystem for the
        given instrument.
    """

    # Make sure the instrument is of the proper format (e.g. "Nircam")
    instrument = inst[0].upper() + inst[1:].lower()

    # Query MAST for all rootnames for the instrument
    service = "Mast.Jwst.Filtered.{}".format(instrument)
    params = {"columns": "filename",
              "filters": []}
    response = Mast.service_request_async(service, params)
    results = response[0].json()['data']

    # Parse the results to get the rootnames
    filenames = [result['filename'].split('.')[0] for result in results]

    # Get list of all preview_images
    preview_images = glob.glob(os.path.join(PREVIEW_IMAGE_FILESYSTEM, '*', '*.jpg'))

    # Get subset of preview images that match the filenames
    preview_images = [os.path.basename(item) for item in preview_images if
                      os.path.basename(item).split('_integ')[0] in filenames]

    # Return only

    return preview_images


def get_preview_images_by_proposal(proposal):
    """Return a list of preview images available in the filesystem for
    the given ``proposal``.

    Parameters
    ----------
    proposal : str
        The one- to five-digit proposal number (e.g. ``88600``).

    Returns
    -------
    preview_images : list
        A list of preview images available in the filesystem for the
        given ``proposal``.
    """

    proposal_string = '{:05d}'.format(int(proposal))
    preview_images = glob.glob(os.path.join(PREVIEW_IMAGE_FILESYSTEM, 'jw{}'.format(proposal_string), '*'))
    preview_images = [os.path.basename(preview_image) for preview_image in preview_images]

    return preview_images


def get_preview_images_by_rootname(rootname):
    """Return a list of preview images available in the filesystem for
    the given ``rootname``.

    Parameters
    ----------
    rootname : str
        The rootname of interest (e.g. ``jw86600008001_02101_00007_guider2``).

    Returns
    -------
    preview_images : list
        A list of preview images available in the filesystem for the
        given ``rootname``.
    """

    proposal = rootname.split('_')[0].split('jw')[-1][0:5]
    preview_images = sorted(glob.glob(os.path.join(
        PREVIEW_IMAGE_FILESYSTEM,
        'jw{}'.format(proposal),
        '{}*'.format(rootname))))
    preview_images = [os.path.basename(preview_image) for preview_image in preview_images]

    return preview_images


def get_proposal_info(filepaths):
    """Builds and returns a dictionary containing various information
    about the proposal(s) that correspond to the given ``filepaths``.

    The information returned contains such things as the number of
    proposals, the paths to the corresponding thumbnails, and the total
    number of files.

    Parameters
    ----------
    filepaths : list
        A list of full paths to files of interest.

    Returns
    -------
    proposal_info : dict
        A dictionary containing various information about the
        proposal(s) and files corresponding to the given ``filepaths``.
    """

    proposals = list(set([f.split('/')[-1][2:7] for f in filepaths]))
    thumbnail_dir = os.path.join(get_config()['jwql_dir'], 'thumbnails')
    thumbnail_paths = []
    num_files = []
    for proposal in proposals:
        thumbnail_search_filepath = os.path.join(
            thumbnail_dir, 'jw{}'.format(proposal), 'jw{}*rate*.thumb'.format(proposal)
        )
        thumbnail = glob.glob(thumbnail_search_filepath)
        if len(thumbnail) > 0:
            thumbnail = thumbnail[0]
            thumbnail = '/'.join(thumbnail.split('/')[-2:])
        thumbnail_paths.append(thumbnail)

        fits_search_filepath = os.path.join(
            FILESYSTEM_DIR, 'jw{}'.format(proposal), 'jw{}*.fits'.format(proposal)
        )
        num_files.append(len(glob.glob(fits_search_filepath)))

    # Put the various information into a dictionary of results
    proposal_info = {}
    proposal_info['num_proposals'] = len(proposals)
    proposal_info['proposals'] = proposals
    proposal_info['thumbnail_paths'] = thumbnail_paths
    proposal_info['num_files'] = num_files

    return proposal_info


def get_thumbnails_by_instrument(inst):
    """Return a list of thumbnails available in the filesystem for the
    given instrument.

    Parameters
    ----------
    inst : str
        The instrument of interest (e.g. ``NIRCam``).

    Returns
    -------
    preview_images : list
        A list of thumbnails available in the filesystem for the
        given instrument.
    """

    # Make sure the instrument is of the proper format (e.g. "Nircam")
    instrument = inst[0].upper() + inst[1:].lower()

    # Query MAST for all rootnames for the instrument
    service = "Mast.Jwst.Filtered.{}".format(instrument)
    params = {"columns": "filename",
              "filters": []}
    response = Mast.service_request_async(service, params)
    results = response[0].json()['data']

    # Parse the results to get the rootnames
    filenames = [result['filename'].split('.')[0] for result in results]

    # Get list of all thumbnails
    thumbnails = glob.glob(os.path.join(THUMBNAIL_FILESYSTEM, '*', '*.thumb'))

    # Get subset of preview images that match the filenames
    thumbnails = [os.path.basename(item) for item in thumbnails if
                  os.path.basename(item).split('_integ')[0] in filenames]

    return thumbnails


def get_thumbnails_by_proposal(proposal):
    """Return a list of thumbnails available in the filesystem for the
    given ``proposal``.

    Parameters
    ----------
    proposal : str
        The one- to five-digit proposal number (e.g. ``88600``).

    Returns
    -------
    thumbnails : list
        A list of thumbnails available in the filesystem for the given
        ``proposal``.
    """

    proposal_string = '{:05d}'.format(int(proposal))
    thumbnails = glob.glob(os.path.join(THUMBNAIL_FILESYSTEM, 'jw{}'.format(proposal_string), '*'))
    thumbnails = [os.path.basename(thumbnail) for thumbnail in thumbnails]

    return thumbnails


def get_thumbnails_by_rootname(rootname):
    """Return a list of preview images available in the filesystem for
    the given ``rootname``.

    Parameters
    ----------
    rootname : str
        The rootname of interest (e.g. ``jw86600008001_02101_00007_guider2``).

    Returns
    -------
    thumbnails : list
        A list of preview images available in the filesystem for the
        given ``rootname``.
    """

    proposal = rootname.split('_')[0].split('jw')[-1][0:5]
    thumbnails = sorted(glob.glob(os.path.join(
        THUMBNAIL_FILESYSTEM,
        'jw{}'.format(proposal),
        '{}*'.format(rootname))))

    thumbnails = [os.path.basename(thumbnail) for thumbnail in thumbnails]

    return thumbnails


def log_into_mast(request):
    """Login via astroquery.mast if user authenticated in web app.

    Parameters
    ----------
    request : HttpRequest object
        Incoming request from the webpage

    """
    if Mast.authenticated():
        return True

    # get the MAST access token if present
    access_token = str(get_mast_token(request))

    # authenticate with astroquery.mast if necessary
    if access_token != 'None':
        Mast.login(token=access_token)
        return Mast.authenticated()
    else:
        return False


def random_404_page():
    """Randomly select one of the various 404 templates for JWQL

    Returns
    -------
    random_template : str
        Filename of the selected template
    """
    templates = ['404_space.html', '404_spacecat.html']
    choose_page = np.random.choice(len(templates))
    random_template = templates[choose_page]

    return random_template


def thumbnails(inst, proposal=None):
    """Generate a page showing thumbnail images corresponding to
    activities, from a given ``proposal``

    Parameters
    ----------
    inst : str
        Name of JWST instrument
    proposal : str (optional)
        Number of APT proposal to filter

    Returns
    -------
    dict_to_render : dict
        Dictionary of parameters for the thumbnails
    """

    filepaths = get_filenames_by_instrument(inst)

    # JUST FOR DEVELOPMENT
    # Split files into "archived" and "unlooked"
    if proposal is not None:
        page_type = 'archive'
    else:
        page_type = 'unlooked'
    filepaths = split_files(filepaths, page_type)

    # Determine file ID (everything except suffix)
    # e.g. jw00327001001_02101_00002_nrca1
    full_ids = set(['_'.join(f.split('/')[-1].split('_')[:-1]) for f in filepaths])

    # If the proposal is specified (i.e. if the page being loaded is
    # an archive page), only collect data for given proposal
    if proposal is not None:
        proposal_string = '{:05d}'.format(int(proposal))
        full_ids = [f for f in full_ids if f[2:7] == proposal_string]

    detectors = []
    proposals = []
    for i, file_id in enumerate(full_ids):
        for file in filepaths:
            if '_'.join(file.split('/')[-1].split('_')[:-1]) == file_id:

                # Parse filename to get program_id
                try:
                    program_id = filename_parser(file)['program_id']
                    detector = filename_parser(file)['detector']
                except ValueError:
                    # Temporary workaround for noncompliant files in filesystem
                    program_id = nfile_id[2:7]
                    detector = file_id[26:]

        # Add parameters to sort by
        if detector not in detectors and not detector.startswith('f'):
            detectors.append(detector)
        if program_id not in proposals:
            proposals.append(program_id)

    # Extract information for sorting with dropdown menus
    # (Don't include the proposal as a sorting parameter if the
    # proposal has already been specified)
    if proposal is not None:
        dropdown_menus = {'detector': detectors}
    else:
        dropdown_menus = {'detector': detectors,
                          'proposal': proposals}

    dict_to_render = {'inst': inst,
                      'tools': MONITORS,
                      'dropdown_menus': dropdown_menus,
                      'prop': proposal}

    return dict_to_render


def thumbnails_ajax(inst, proposal=None):
    """Generate a page that provides data necessary to render the
    ``thumbnails`` template.

    Parameters
    ----------
    inst : str
        Name of JWST instrument
    proposal : str (optional)
        Number of APT proposal to filter

    Returns
    -------
    data_dict : dict
        Dictionary of data needed for the ``thumbnails`` template
    """

    # Get the available files for the instrument
    filepaths = get_filenames_by_instrument(inst)

    # Get set of unique rootnames
    rootnames = set(['_'.join(f.split('/')[-1].split('_')[:-1]) for f in filepaths])

    # If the proposal is specified (i.e. if the page being loaded is
    # an archive page), only collect data for given proposal
    if proposal is not None:
        proposal_string = '{:05d}'.format(int(proposal))
        rootnames = [rootname for rootname in rootnames if rootname[2:7] == proposal_string]

    # Initialize dictionary that will contain all needed data
    data_dict = {}
    data_dict['inst'] = inst
    data_dict['file_data'] = {}

    # Gather data for each rootname
    for rootname in rootnames:

        # Parse filename
        try:
            filename_dict = filename_parser(rootname)
        except ValueError:
            # Temporary workaround for noncompliant files in filesystem
            filename_dict = {'activity': rootname[17:19],
                             'detector': rootname[26:],
                             'exposure_id': rootname[20:25],
                             'observation': rootname[7:10],
                             'parallel_seq_id': rootname[16],
                             'program_id': rootname[2:7],
                             'visit': rootname[10:13],
                             'visit_group': rootname[14:16]}

        # Get list of available filenames
        available_files = get_filenames_by_rootname(rootname)

        # Add data to dictionary
        data_dict['file_data'][rootname] = {}
        data_dict['file_data'][rootname]['filename_dict'] = filename_dict
        data_dict['file_data'][rootname]['available_files'] = available_files
        data_dict['file_data'][rootname]['expstart'] = get_expstart(rootname)
        data_dict['file_data'][rootname]['suffixes'] = [filename_parser(filename)['suffix'] for
                                                        filename in available_files]

    # Extract information for sorting with dropdown menus
    # (Don't include the proposal as a sorting parameter if the
    # proposal has already been specified)
    detectors = [data_dict['file_data'][rootname]['filename_dict']['detector'] for
                 rootname in list(data_dict['file_data'].keys())]
    proposals = [data_dict['file_data'][rootname]['filename_dict']['program_id'] for
                 rootname in list(data_dict['file_data'].keys())]
    if proposal is not None:
        dropdown_menus = {'detector': detectors}
    else:
        dropdown_menus = {'detector': detectors,
                          'proposal': proposals}

    data_dict['tools'] = MONITORS
    data_dict['dropdown_menus'] = dropdown_menus
    data_dict['prop'] = proposal

    return data_dict
