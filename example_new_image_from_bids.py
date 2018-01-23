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


# IMPORTANT: Bids compliant data needs a 'derivatives' directory that has pre-processed images in it
data_dir = '/Users/josephmann/nilearn_data/bids_langloc_example/bids_langloc_dataset/'

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

# TODO add doc strings to functions
# TODO use datalad to fetch data
# TODO use bids to get info

def beta_img_from_model_events_confounds(model, imgs, events, confounds):
    # model.fit will create a design matrix from events and confounds
    # and then fit the GLM to the imgs,
    model.fit(imgs, events, confounds)

    # note: design_matrices_ will not exist until fit has been run
    design_matrix_a = model.design_matrices_[0]

    # we use a contrast matrix to obtain the effect sizes from model
    # this contrast matrix is the identity matrix and does NOT contrast effect sizes (despite its name)

    n_features = design_matrix_a.shape[1]

    contrast_matrix_a = np.eye(n_features)

    # theta_imgs_l is a list of Ni1Images that contain the effect sizes for each column in the design_matrix
    theta_imgs_l = [  model.compute_contrast( contrast_matrix_a[i], output_type = 'effect_size')
                      for i in range(n_features)]

    # con_img is a Ni1image that contains all of the theta values in one image.
    con_img = nilearn.image.concat_imgs(theta_imgs_l)
    return con_img, design_matrix_a

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

### TODO How about if I want to compare just the fixed-effect contrasts

import pytest

def get_contrasts():
    l = list()
    l.append( np.ones(13))
    l.append( [1,1,1,0,0,0,0,0,0,0,0,0,1])
    return l

from nistats.first_level_model import first_level_models_from_bids

def get_data():
    n_runs = 3
    data_dir = '/Users/josephmann/nilearn_data/bids_langloc_example/bids_langloc_dataset/'
    task_label = 'languagelocalizer'
    space_label = 'MNI152nonlin2009aAsym'
    derivatives_folder = 'derivatives'
    models, models_run_imgs, models_events, models_confounds = \
        first_level_models_from_bids(
            data_dir, task_label, space_label, #smoothing_fwhm=5.0,
            derivatives_folder=derivatives_folder,
            noise_model='ols', signal_scaling=False)
    return list(zip(models, models_run_imgs, models_events, models_confounds))[:n_runs]

# i don't want to have to regenerate these images for every test so trying this
@pytest.fixture(params= get_data())
def  beta_img_design_matrix(request):
    model, imgs, events, confounds = request.param
    return beta_img_from_model_events_confounds(model, imgs, events, confounds) + (imgs[0],)

# @pytest.mark.parametrize("arg_tuple", get_data())
@pytest.mark.parametrize("contrasts_l", get_contrasts())
def test_contrasts(beta_img_design_matrix, contrasts_l):
    beta_img0, design_matrix, img0 = beta_img_design_matrix
    contrasts_l = np.array(contrasts_l).astype(np.bool)
    new_img = predictedImg_from_betaImg_designMatrix(beta_img0, design_matrix, contrasts_l)
    img_j = nilearn.image.load_img(img0)
    masker = nilearn.input_data.NiftiMasker()
    masker.fit(new_img) # trying background masking_strategy on new_img (should be same as 'epi' on previous
    corr = reporting.compare_niimgs([new_img], [img_j], masker, plot_hist=False)
    print(corr)
    assert( corr[0] > 0.9)
