# get nibs data
# get beta img and design matrix

# find ways to make variants
## different smoothing on original epi
## smoothing on betas
## average betas from several scans
## create distribution from several scans and generate betas from random selection
### can we do something similar with MVPA ?
### - asking for a friend..

# boost/attenuate regional beta
# boost specific fixed effect

# with predicted image
# * specific region
## with time noise
## with spatial noise
## with idiosyncratic noise
##
## play, Where's Waldo (...?)
##

# what is the simplest thing I can submit??

# acquisition of data

# full model fit
# show contrasts

# make predicted from betas
# full model fit on predicted
# show contrasts

# compare contrasts

#################
## IMPORTANT: essential mod to nistats has yet to be integrated.
## please use https://github.com/josephsmann/nistats until further notice
##
import sys
import nilearn
import numpy as np
from nilearn.input_data import NiftiMasker

### IMPORTANT: change the following path to correct location

local_nistats_dir = "/Users/josephmann/Documents/GitHub/nistats_j"
sys.path.insert(0, local_nistats_dir)

## Following code is almost perfect copy from nistats example plot_bids_features.py
## with changes to parameters in first_level_models_from_bids:
##  smoothig_fwhm has been removed
## signal_scaling is now False
## noise_model = 'ols' (probably not essential at this point) TODO

##############################################################################
# Fetch openneuro BIDS dataset
# -----------------------------
# We download one subject from the stopsignal task in the ds000030 V4 BIDS
# dataset available in openneuro.
# This dataset contains the necessary information to run a statistical analysis
# using Nistats. The dataset also contains statistical results from a previous
# FSL analysis that we can employ for comparison with the Nistats estimation.
# from nistats.datasets import (fetch_openneuro_dataset_index,
#                               fetch_openneuro_dataset, select_from_index)
#
# _, urls = fetch_openneuro_dataset_index()
#
# exclusion_patterns = ['*group*', '*phenotype*', '*mriqc*',
#                       '*parameter_plots*', '*physio_plots*',
#                       '*space-fsaverage*', '*space-T1w*',
#                       '*dwi*', '*beh*', '*task-bart*',
#                       '*task-rest*', '*task-scap*', '*task-task*']
# urls = select_from_index(
#     urls, exclusion_filters=exclusion_patterns, n_subjects=1)
#
# data_dir, _ = fetch_openneuro_dataset(urls=urls)

# my attempt at using different data
# IMPORTANT: Bids compliant data needs a 'derivatives' directory that has pre-processed images in it
# try
data_dir = '/Users/josephmann/nilearn_data/bids_langloc_example/bids_langloc_dataset/'
# data_dir = 'Users/josephmann/datalad/openfmri/ds000007'

##############################################################################
# Obtain automatically FirstLevelModel objects and fit arguments
# ---------------------------------------------------------------
# From the dataset directory we obtain automatically FirstLevelModel objects
# with their subject_id filled from the BIDS dataset. Moreover we obtain
# for each model the list of run imgs and their respective events and
# confounder regressors. Confounders are inferred from the confounds.tsv files
# available in the BIDS dataset.
# To get the first level models we have to specify the dataset directory,
# the task_label and the space_label as specified in the file names.
# We also have to provide the folder with the desired derivatives, that in this
# case were produced by the fmriprep BIDS app.

# TODO space label is necessary because...

# from nistats.first_level_model import first_level_models_from_bids
# # task_label = 'stopsignal'
# # task_label = 'stopmanual'
# task_label = 'languagelocalizer'
# # space_label = 'MNI152NLin2009cAsym'
# space_label = 'MNI152nonlin2009aAsym'
# derivatives_folder = 'derivatives'
# models, models_run_imgs, models_events, models_confounds = \
#     first_level_models_from_bids(
#         data_dir, task_label, space_label, #smoothing_fwhm=5.0,   ### TODO if I can get my own datadir then I
#         derivatives_folder=derivatives_folder,
#         noise_model='ols', signal_scaling=False)

# TODO load multiple runs

# use datalad to fetch data
# use bids to get info
# seems simple...

#############################################################################
# Take model and model arguments of the subject and process events
# model, imgs, events, confounds = (
#     models[0], models_run_imgs[0], models_events[0], models_confounds[0])

#############################################################################
# End of data acquisition, and  plot_bids_features.py code
#############################################################################

def beta_img_from_model_events_confounds(model, imgs, events, confounds):
    # model.fit will create a design matrix from events and confounds
    # and then fit the GLM to the imgs,
    model.fit(imgs, events, confounds)

    # note: design_matrices_ will not exist until fit has been run
    design_matrix_a = model.design_matrices_[0]

    # we use a contrast matrix to obtain the effect sizes from model
    # this contrast matrix is the identity matrix and does NOT contrast effect sizes (despite its name)
    contrast_matrix_a = np.eye(design_matrix_a.shape[1])
    contrasts = dict([(column, contrast_matrix_a[i])
                      for i, column in enumerate(design_matrix_a.columns)])

    # theta_imgs_l is a list of Ni1Images that contain the effect sizes for each column in the design_matrix
    theta_imgs_l = list()


    for i in range(design_matrix_a.shape[1]):
        theta_imgs_l.append(
            model.compute_contrast( contrast_matrix_a[i], output_type = 'effect_size')
        )

    # con_img is a Ni1image that contains all of the theta values in one image.
    con_img = nilearn.image.concat_imgs(theta_imgs_l)
    return con_img, design_matrix_a

# masked_con_img is a 2-dim matrix ( parameter x voxels)
def predictedImg_from_betaImg_designMatrix(beta_img, design_matrix, con_params_l = []):

    # con_params is a list indicating which contrasts we will be taking from the model
    # initially we take all of them
    if len(con_params_l) == 0:
        con_params_l = np.ones(design_matrix.shape[1], dtype=np.bool)

    con_mask = NiftiMasker()
    masked_con_img = con_mask.fit_transform(beta_img)
    # this is just: design_matrix[:, contrast_params] @ masked_con_img
    pred_a = np.einsum('tp,pv->tv',
                       design_matrix.values[:, con_params_l],
                       masked_con_img[con_params_l, :])
    new_img = con_mask.inverse_transform(pred_a)
    return new_img

from nistats import reporting

### How about if I want to compare just the fixed-effect contrasts

# TODO make function that takes 4-d scan (1 run) and design matrix or effects/confounds, contrast and returns 4-d with betas

import pytest
# @pytest.fixture
# def contrasts_l():
#     return np.ones(13)

def get_contrasts():
    l = list()
    l.append( np.ones(13))
    return l


from nistats.first_level_model import first_level_models_from_bids

def get_data():
    data_dir = '/Users/josephmann/nilearn_data/bids_langloc_example/bids_langloc_dataset/'
    task_label = 'languagelocalizer'
    space_label = 'MNI152nonlin2009aAsym'
    derivatives_folder = 'derivatives'
    models, models_run_imgs, models_events, models_confounds = \
        first_level_models_from_bids(
            data_dir, task_label, space_label, #smoothing_fwhm=5.0,
            derivatives_folder=derivatives_folder,
            noise_model='ols', signal_scaling=False)
    return list(zip(models, models_run_imgs, models_events, models_confounds))

@pytest.fixture(params= get_data())
def  beta_img0(tuple_arg):
    model, imgs, events, confounds = tuple_arg
    return beta_img_from_model_events_confounds(model, imgs,events, confounds)

@pytest.mark.parametrize("arg_tuple", get_data())
@pytest.mark.parametrize("contrasts_l", get_contrasts())
def test_contrasts(arg_tuple, contrasts_l):
    model, imgs,events, confounds = arg_tuple
    beta_img0, design_matrix = beta_img_from_model_events_confounds(model, imgs,events, confounds)
    contrasts_l = np.array(contrasts_l).astype(np.bool)
    new_img = predictedImg_from_betaImg_designMatrix(beta_img0, design_matrix, contrasts_l)
    img_j = nilearn.image.load_img(imgs[0])
    # masker = nilearn.input_data.NiftiMasker()
    # masker.fit(img_j)
    corr = reporting.compare_niimgs([new_img], [img_j], model.masker_, plot_hist=False)
    print(corr)
    assert( corr[0] > 0.9)
