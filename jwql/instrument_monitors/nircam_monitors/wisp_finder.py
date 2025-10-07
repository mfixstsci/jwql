#! /usr/bin/env python

"""This module contains code for the wisp finder monitor.

Author
------
    - Bryan Hilbert

Use
---
    This module can be used from the command line as such:

    ::

        python wisp_monitor.py


Overall flow:

1. Look in database table for last successful run on the monitor.
2. Get the datetime of that run.
3. Query MAST for all NIRCam B4 full frame files (exclude coron?) since that datetime
4. Copy over rate files to working dir
5. Re-scale, and create png files using the same method that was used for the ML training
6. Load the trained model
7. Use the model to predict whether each png contains a wisp
8. For those files where the prediction is that a wisp is present, set the wisp flag in the anomalies database
9. Delete pngs
10. Update the database with the datetime of the current run
"""

import argparse
import datetime
import logging
import os
import shutil
import warnings

from astropy.time import Time
from astroquery.mast import Observations
from django import setup
from django.utils import timezone
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

from jwql.utils import monitor_utils
from jwql.utils.constants import ON_GITHUB_ACTIONS, ON_READTHEDOCS, WISP_PROBABILITY_THRESHOLD
from jwql.utils.logging_functions import log_info, log_fail
from jwql.utils.utils import get_config
from jwql.website.apps.jwql.archive_database_update import files_in_filesystem
from jwql.instrument_monitors.nircam_monitors import prepare_wisp_pngs
from jwql.utils.utils import filesystem_path

if not ON_GITHUB_ACTIONS and not ON_READTHEDOCS:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "jwql.website.jwql_proj.settings")
    setup()
    from jwql.website.apps.jwql.models import Anomalies, RootFileInfo
    from jwql.website.apps.jwql.monitor_models.wisp_finder import WispFinderB4QueryHistory


MAX_QUERY_DURATION = 7.  # days

def add_wisp_flag(basename):
    """Add the wisps flag to the RootFileInfo entry for the given filename

    Parameters
    ----------
    basename : str
        Filename minus the suffix and ".fits". e.g. "jw01068004001_02101_00001_nrcb1"
    """
    # Get the RootFileInfo instance
    root_file_info = RootFileInfo.objects.get(root_name=basename)

    # Set user name and date
    user_name = 'ML_wisp_finder'
    entry_date = timezone.now()

    # Set the wisps flag, and add the current time, and say that the flag is coming from the wisp finder
    anomalies_exist = hasattr(root_file_info, 'anomalies')
    if anomalies_exist:
        # If an Anomalies instance is already associated with the RootFileInfo instance, then
        # set the wisps flag
        root_file_info.anomalies.wisps = True
        root_file_info.anomalies.flag_date = timezone.now()
        root_file_info.anomalies.user = 'ML_wisp_finder'
        root_file_info.anomalies.save(update_fields=['wisps', 'flag_date', 'user'])
    else:
        # If an Anomaly object is not associated with the RootFileInfo instance, create one
        default_dict = {'flag_date': entry_date,
                        'user': user_name}
        for anomaly in Anomalies.get_all_anomalies():
            default_dict[anomaly] = (anomaly in ['wisps'])
        Anomalies.objects.update_or_create(root_file_info=root_file_info, defaults=default_dict)


def copy_files_to_working_dir(filepaths):
    """Copy files from MAST into a working directory

    Parameters
    ----------
    filepaths : list
        List of full paths of files to be copied

    Returns
    -------
    copied_filepaths : list
        List of new locations for the files
    """
    working_dir = get_config()["working"]

    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
    copied_filepaths = []

    for filepath in filepaths:
        shutil.copy2(filepath, working_dir)
        copied_filepaths.append(os.path.join(working_dir, os.path.basename(filepath)))
        logging.info(f'Copying {filepath} to {working_dir}')

    return copied_filepaths


def create_transform():
    """Create a transform function that will be used to modify images
    and place them in the format expected by the ML model

    Returns
    -------
    transform : torchvision.transforms.transforms.Compose
        Image transform model
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),          # Resize images to a fixed size
        transforms.ToTensor(),                  # Convert images to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])
    return transform


def define_model_architecture():
    """Define the basic architecture of the ML model. This will be the framework into which
    the model parameters will be loaded, in order to fully define the function.

    Returns
    -------
    model : torchvision.models.resnet.ResNet
        ResNet model to use for wisp prediction
    """
    # Load pre-trained ResNet-18 model
    model = models.resnet18(weights='IMAGENET1K_V1')

    # Modify the final fully connected layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 1)

    # Add a sigmoid activation after the final layer
    model.add_module('sigmoid', nn.Sigmoid())
    return model


def define_options(parser=None, usage=None, conflict_handler='resolve'):
    """Add command line options

    Parameters
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

    parser.add_argument('-m', '--model_filename', type=str, default=None,
                        help='Filename of saved ML model. (default=%(default)s)')
    parser.add_argument('-s', '--starting_date', type=float, default=None,
                        help='Earliest MJD to search for data. If None, date is retrieved from database.')
    parser.add_argument('-e', '--ending_date', type=float, default=None,
                        help='Latest MJD to search for data. If None, the current date is used.')
    parser.add_argument('-f', '--file_list', type=str, nargs='+', default=None,
                        help='List of full paths to files to run the monitor on.')
    return parser


def get_latest_run():
    """Retrieve the ending time of the latest successful run from the database

    Returns
    -------
    latset_date : float
        MJD of the ending time of the latest successful run of the monitor
    """
    filters = {"run_monitor": True}
    record = WispFinderB4QueryHistory.objects.filter(**filters).order_by("-end_time_mjd").first()

    if record is None:
        query_result = 59607.0  # a.k.a. Jan 28, 2022 == First JWST images
        logging.info(f'\tNo successful previous runs found. Beginning search date will be set to {query_result}.')
    else:
        query_result = record.end_time_mjd

    return query_result


def load_ml_model(model_filename):
    """Load the ML model for wisp prediction

    Parameters
    ----------
    model_filename : str
        Location of file containing the model. e.g. /path/to/my_best_model.pth

    Returns
    -------
    model : torchvision.models.resnet.ResNet
        ResNet model to use for wisp prediction
    """
    model = define_model_architecture()
    model.load_state_dict(torch.load(model_filename))
    model.eval()  # Set model to evaluation mode
    return model


def predict_wisp(model, image_path, transform):
    """Use the model to predict whether there is a wisp in the image. The model returns
    a probability. So we use a threshold to separate those predictions into 'wisp' and
    'no wisp' bins.

    Parameters
    ----------
    model : torchvision.models.resnet.ResNet
        ResNet model to use for wisp prediction

    image_path : str
        Full path to the png file

    transform : torchvision.transforms.transforms.Compose
        Image transform function used to modify the input images into the format
        expected by the ML model.

    Returns
    -------
    prediction_label : str
        "wisp" or "no wisp"
    """
    image_tensor = preprocess_image(image_path, transform)  # Preprocess the image

    with torch.no_grad():  # Make prediction without gradients
        output = model(image_tensor)

    # The model outputs a single probability (e.g., for "wisp"). So, use a threshold
    # to determine whether the prediction is wisp or no_wisp.
    probability = torch.sigmoid(output).item()
    threshold = WISP_PROBABILITY_THRESHOLD
    prediction_label = "wisp" if probability >= threshold else "no wisp"
    return prediction_label, probability, threshold


def preprocess_image(image_path, transform):
    """Load the png file and prepare it for input to the model

    Parameters
    ----------
    image_path : str
        Path and filename of the png file

    transform : torchvision.transforms.transforms.Compose
        Image transform function used to modify the input images into the format
        expected by the ML model.

    Returns
    -------
    image : torch.Tensor
        Tensor on which the model will run
    """
    image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension
    return image


def query_mast(starttime, endtime):
    """Query MAST between the given dates. Generate a list of NRCB4 files on which
    the wisp model will be run

    Parameters
    ----------
    starttime : float or str
        MJD of the beginning of the search interval

    endtime : float or str
        MJD of the end of the search interval

    Returns
    -------
    rate_files : list
        List of filenames
    """
    logging.info("Running sci_obs_id query")
    sci_obs_id_table = Observations.query_criteria(instrument_name=["NIRCAM/IMAGE"],
                                                   provenance_name=["CALJWST"],  # Executed observations
                                                   t_min=[starttime, endtime]
                                                   )

    sci_files_to_download = []

    # Loop over visits identifying uncalibrated files that are associated
    # with them
    for i, exposure in enumerate(sci_obs_id_table):
        products = Observations.get_product_list(exposure)
        filtered_products = Observations.filter_products(products,
                                                         productType='SCIENCE',
                                                         productSubGroupDescription='RATE',
                                                         calib_level=[2])
        logging.info(f"\tExposure {i+1} of {len(sci_obs_id_table)}: {len(products)} products filters to {len(filtered_products)} rate files")
        sci_files_to_download.extend(filtered_products['dataURI'])

    # The current ML wisp finder model is only trained for the wisps on the B4 detector,
    # so keep only those files. Also, keep only the filenames themselves.
    logging.info(f"Sorting {len(sci_files_to_download)} rate files")
    rate_files = sorted([fname.replace('mast:JWST/product/', '') for fname in sci_files_to_download if 'nrcb4' in fname])
    return rate_files


def remove_duplicate_files(file_list):
    """When running locally, it's possible to end up with duplicates of some filenames in
    the list of files, because the files are present in both the public and proprietary
    lists. This function will remove the duplicates.

    Parameters
    ----------
    file_list : list
        List of full paths to input files

    Returns
    -------
    unique_files : list
        List of files with unique basenames
    """
    file_list = np.array(file_list)
    unique_files = []
    basenames_only = sorted(list(set([os.path.basename(e) for e in file_list])))

    for basename in basenames_only:
        matches = np.array([basename in e for e in file_list])

        # We've already checked for file existence, so no need to do that here
        unique_files.append(file_list[matches][0])
    return unique_files

@log_fail
@log_info
def run(model_filename=None, starting_date=None, ending_date=None, file_list=None):
    """Run the wisp finder monitor. From user-input dates or dates retrieved from
    the database, query MAST for all NIRCam NRCB4 full-frame imaging mode data. For
    each file, create a png file continaing an image of the rate file, scaled to a
    consistent brightness/range as well as size. Use a trained neural network model
    to predict whether the image contains a wisp. If so, set the wisps anomaly flag
    for that file.

    Parameters
    ----------
    model_filename : str
        Name of a pytorch-generated model to load and use for prediction

    starting_date : float
        Earliest MJD to use when querying MAST for data

    ending_date : float
        Latest MJD to use when querying MAST for data

    file_list : list
        List of filenames (e.g. ["jw01068004001_02101_00001_nrcb4_rate.fits"])
        to run the wisp prediction for. If this list is provided, the MAST query
        is skipped.
    """
    # If no model_filename is given, the retrieve the default model_filename
    # from the config file
    if model_filename is None:
        model_filename = get_config()['wisp_finder_ML_model']

    if os.path.isfile(model_filename):
        logging.info(f'Using ML model saved in: {model_filename}')
    else:
        raise FileNotFoundError(f"WARNING: {model_filename} does not exist. Unable to load ML model.")

    if file_list is None:

        # If ending_date is not provided, set it equal to the current time
        if ending_date is None:
            ending_date = timezone.now()
            t_end = Time(ending_date, scale='utc')
            ending_date = t_end.mjd

        # If starting date is not provided, then query the database for the last
        # successful run of this monitor. Use the ending date of that run for the
        # starting_date of this run
        if starting_date is None:
            latest_run_end = get_latest_run()
            starting_date = latest_run_end

        logging.info(f"Using MJD {starting_date} to {ending_date} to search for files")

        # If the starting and ending dates span a long time, break up the time into
        # smaller chunks in order to get reasonable MAST query lists and so as not to
        # copy really large numbers of files to the working directory
        if ending_date - starting_date > MAX_QUERY_DURATION:
            logging.info(f"Time range is greater than the maximum allowed duration of {MAX_QUERY_DURATION} days")
            starting_dates = np.arange(starting_date, ending_date, MAX_QUERY_DURATION)
            # Make the ending_dates 0.1 second shy of MAX_QUERY_DURATION so that they are not
            # exactly the same as the subsequent starting_date
            ending_dates = starting_dates + MAX_QUERY_DURATION - (0.1 / 3600. / 24.)
            # Set the final ending_date equal to the originally requested ending_date
            if ending_dates[-1] > ending_date:
                ending_dates[-1] = ending_date
            logging.info(f"Breaking up the query into {len(starting_dates)} smaller queries.")
        else:
            starting_dates = np.array([starting_date])
            ending_dates = np.array([ending_date])

        for subq_starting_date, subq_ending_date in zip(starting_dates, ending_dates):
            # Query MAST between starting_date and ending_date, and get a list of files
            # to run the wisp prediction on.
            rate_files = query_mast(subq_starting_date, subq_ending_date)
            logging.info(f"MAST query betwen MJD {subq_starting_date} and {subq_ending_date} returned {len(rate_files)} rate files")
            run_predictor(rate_files, model_filename, subq_starting_date, subq_ending_date)

    else:
        rate_files = file_list
        starting_date = 0.0
        ending_date = 0.0
        logging.info(f"Running predictor on list of {len(rate_files)} files.")
        run_predictor(rate_files, model_filename, subq_starting_date, subq_ending_date)

    logging.info('Wisp Finder Monitor completed successfully.')


def run_predictor(ratefiles, model_file, start_date, end_date):
    """Given a list of files, a ML model file, and dates, check all of the files for wisps.

    Parameters
    ----------
    ratefiles : list
        List of fits files to check for wisps.

    model_file : str
        Name of a file containing the ML model to be used

    start_date : float
        MJD of the starting date of the range encompassing ``ratefiles``.
        Used for populating the history database table.

    end_date : float
        MJD of the ending date of the range encompassing ``ratefiles``.
        Used for populating the history database table.
    """
    if len(ratefiles) > 0:
        monitor_run = True

        # Find the location in the filesystem for all files
        logging.info("Locating files in the filesystem")
        filepaths = []
        for rate_file in ratefiles:
            try:
                filepaths.append(filesystem_path(rate_file, check_existence=True))
            except Exception as e:
                logging.warning(f"Failed to find {rate_file} with error {e}")

        # Remove any duplicates coming from files that are present in both the
        # public and proprietary filesystems
        n_filepaths_before = len(filepaths)
        filepaths = remove_duplicate_files(filepaths)
        n_filepaths_after = len(filepaths)

        # Copy files to working directory
        logging.info(f"Copying {n_filepaths_after} files from the filesystem to the working directory (removed {n_filepaths_before - n_filepaths_after} duplicates).")
        working_filepaths = copy_files_to_working_dir(filepaths)

        # Load the trained ML model
        model = load_ml_model(model_file)

        # Create transform to use when creating image tensor
        transform = create_transform()

        # For each fits file, create a png file, and have the ML model predict if there is a wisp
        for working_filepath in working_filepaths:
            # Create png
            working_dir = os.path.dirname(working_filepath)
            logging.info(f'Creating png for {os.path.basename(working_filepath)}. Saving to {working_dir}')
            png_filename = prepare_wisp_pngs.run(working_filepath, out_dir=working_dir)

            # Predict
            prediction, probability, threshold = predict_wisp(model, png_filename, transform)

            # If a wisp is predicted, set the wisp flag in the anomalies database
            if prediction == "wisp":
                # Create the rootname. Strip off the path info, and remove '.fits' and the suffix
                # (i.e. 'rate'')
                rootfile = '_'.join(os.path.basename(working_filepath).split('.')[0].split('_')[0:-1])
                logging.info(f"\tFound wisp in {rootfile} (probability {probability} > threshold {threshold})\n\n")

                # Add the wisp flag to the RootFileInfo object for the rootfile
                add_wisp_flag(rootfile)
            else:
                rootfile = '_'.join(os.path.basename(working_filepath).split('.')[0].split('_')[0:-1])
                logging.info(f'\tNo wisp in {rootfile} (probability {probability} < threshold {threshold})\n')

            # Delete the png and fits files
            os.remove(png_filename)
            os.remove(working_filepath)
    else:
        # If no ratefiles are found
        logging.info(f"No rate files found. Ending monitor run.")
        monitor_run = False

    # Update the database with info about this run of the monitor. We keep the
    # starting and ending dates of the search. No need to keep the names of the files
    # that are found to contain a wisp, because that info will be in the  RootFileInfo
    # instances.
    new_entry = {'start_time_mjd': start_date,
                 'end_time_mjd': end_date,
                 'run_monitor': monitor_run,
                 'entry_date': datetime.datetime.now(datetime.timezone.utc)}
    entry = WispFinderB4QueryHistory(**new_entry)
    entry.save()


if __name__ == '__main__':
    module = os.path.basename(__file__).strip('.py')
    start_time, log_file = monitor_utils.initialize_instrument_monitor(module)
    parser = define_options()
    args = parser.parse_args()

    run(args.model_filename,
        file_list=args.file_list,
        starting_date=args.starting_date,
        ending_date=args.ending_date)

    monitor_utils.update_monitor_table(module, start_time, log_file)
