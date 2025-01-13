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

import datetime
import logging
import os
import shutil
import warnings

from astroquery.mast import Observations
from django import setup
from django.utils import timezone
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

from jwql.utils.utils import get_config
from jwql.website.apps.jwql.archive_database_update import files_in_filesystem
from jwql.website.apps.jwql.models import Anomalies, RootFileInfo
from . import prepare_wisp_pngs

if 1>0:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "jwql.website.jwql_proj.settings")
    setup()


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
    """
    working_dir = get_config()["working"]
    copied_filepaths = []
    for filepath in filepaths:
        shutil.copy2(filepath, working_dir)
        copied_filepaths.append(os.path.join(working_dir, os.path.basename(filepath)))


        print(f'copying {filepath} to {working_dir}')


    return copied_filepaths


def create_transform():
    """
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),          # Resize images to a fixed size
        transforms.ToTensor(),                  # Convert images to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])
    return transform


def define_model_architecture():
    """
    """
    # Load pre-trained ResNet-18 model
    model = models.resnet18(weights='IMAGENET1K_V1')

    # Modify the final fully connected layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 1)

    # Add a sigmoid activation after the final layer
    model.add_module('sigmoid', nn.Sigmoid())
    return model


def load_ml_model(model_filename):
    """Load the ML model for wisp prediction

    Parameters
    ----------
    model_filename : str
        Location of file containing the model. e.g. /path/to/my_best_model.pth
    """

    #model = torch.load(model_filename)

    model = define_model_architecture()
    model.load_state_dict(torch.load(model_filename))


    model.eval()  # Set model to evaluation mode
    return model


def predict_wisp(model, image_path, transform):
    image_tensor = preprocess_image(image_path, transform)  # Preprocess the image

    with torch.no_grad():  # Make prediction without gradients
        output = model(image_tensor)

    # Interpret the result
    #_, predicted_class = torch.max(output, 1)
    #class_labels = ["no wisp", "wisp"]
    #prediction_label = class_labels[predicted_class.item()]

    # If your model instead outputs a single probability (e.g., for "wisp"), use a threshold
    # instead of the lines above
    probability = torch.sigmoid(output).item()
    threshold = 0.5
    prediction_label = "wisp" if probability >= threshold else "no wisp"

    #print(f"The model predicts: {prediction_label} with probability {probability:.2f}")
    return prediction_label


def preprocess_image(image_path, transform):
    """Load the png file and prepare it for input to the model
    """
    image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension
    return image


def query_mast(starttime, endtime):
    """Query MAST between the given dates. Generate a list of NRCB4 files on which
    the wisp model will be applied

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
    sci_obs_id_table = Observations.query_criteria(instrument_name=["NIRCAM/IMAGE"],
                                                   provenance_name=["CALJWST"],  # Executed observations
                                                   t_min=[starttime, endtime]
                                                   )

    sci_files_to_download = []

    # Loop over visits identifying uncalibrated files that are associated
    # with them
    for exposure in (sci_obs_id_table):
        products = Observations.get_product_list(exposure)
        filtered_products = Observations.filter_products(products,
                                                         productType='SCIENCE',
                                                         productSubGroupDescription='RATE',
                                                         calib_level=[2])
        sci_files_to_download.extend(filtered_products['dataURI'])

    # The current ML wisp finder model is only trained for the wisps on the B4 detector,
    # so keep only those files. Also, keep only the filenames themselves.
    rate_files = sorted([fname.replace('mast:JWST/product/', '') for fname in sci_files_to_download if 'nrcb4' in fname])
    return rate_files


def remove_duplicate_files(file_list):
    """When running locally, it's possible to end up with duplicates of some filenames in
    the list of files, because the files are present in both the public and proprietary
    lists. This function will remove the duplicates.
    """
    file_list = np.array(file_list)
    unique_files = []
    basenames_only = set([os.path.basename(e) for e in file_list])
    for basename in basenames_only:
        matches = np.array([basename in e for e in file_list])
        unique_files.append(file_list[matches][0])
    return unique_files


def run(model_filename, starting_date=None, ending_date=None, file_list=None):
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
    if file_list is None:

        # If ending_date is not provided, set it equal to the current time
        if ending_date is None:
            ending_date = timezone.now()

        # If starting date is not provided, then query the database for the last
        # successful run of this monitor. Use the ending date of that run for the
        # starting_date of this run
        if starting_date is None:
            latest_run_end = get_latest_run()
            starting_date = latest_run_end

        # Query MAST between starting_date and ending_date, and get a list of files
        # to run the wisp prediction on.
        rate_files = query_mast(starting_date, ending_date)

    else:
        rate_files = file_list

    # Find the location in the filesystem for all files
    filepaths_public = files_in_filesystem(rate_files, 'public')
    filepaths_proprietary = files_in_filesystem(rate_files, 'proprietary')
    filepaths = filepaths_public + filepaths_proprietary
    filepaths = remove_duplicate_files(filepaths)

    working_filepaths = copy_files_to_working_dir(filepaths)

    # Load the trained ML model
    model = load_ml_model(model_filename)

    # Create transform to use when creating image tensor
    transform = create_transform()

    # For each fits file, create a png file, and have the ML model predict if there is a wisp
    for working_filepath in working_filepaths:


        # we can probably find a way to simply create an Image instance and predict, rather than
        # saving and then reading in a png...

        # Create png
        working_dir = os.path.dirname(working_filepath)
        png_filename = prepare_wisp_pngs.run(working_filepath, out_dir=working_dir)
        print(png_filename)

        # Predict
        prediction = predict_wisp(model, png_filename, transform)

        print(png_filename, prediction)  # FOR DEVELOPMENT ONLY. REMOVE BEFORE MERGING

        # If a wisp is predicted, set the wisp flag in the anomalies database
        if prediction == 'wisp':
            print('Found wisp!!')
            # Create the rootname. Strip off the path info, and remove '.fits' and the suffix
            # (i.e. 'rate'')
            rootfile = '_'.join(os.path.basename(working_filepath).split('.')[0].split('_')[0:-1])

            # Add the wisp flag to the RootFileInfo object for the rootfile
            add_wisp_flag(rootfile)
            print('Added wisp flag')
        else:
            print('No wisp')

        # Delete the png and fits files
        print(f'Removing {png_filename} and {working_filepath}')
        os.remove(png_filename)
        os.remove(working_filepath)

    # Update the database with info about this run of the monitor
    if file_list is None:
        do_it()
    else:
        print('What dates do we add to the database in this case?')

