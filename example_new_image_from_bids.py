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
# TODO How about if I want to compare just the fixed-effect contrasts (between what?)
# TODO region to voxel function (see atlas stuff in  nisim.py)

def beta_img_from_model_events_confounds(model, imgs, events, confounds):
    """
    :param model: nistats.FirstLevelModel
    :param imgs: a list of 4D Nifti1Images
    :param events: a dataframe containing columns 'onsets' and events
    :param confounds: a dataframe contain confounds
    :return: a tuple containing
            - a 4d NiftiImage have the same dimensions as the input images
            except for the 4th dim which has the effect sizes of the columns in the design matrix
            - the design matrix

    This function finds for each voxel in each of the imgs the effect size for each column of the design matrix
    created from the events and confounds.
    """
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

from nilearn import image
from nilearn import datasets

def standardize_atlas(atlas_ni):
    """
    return an atlas where each feature is normalized to values between
    0 and 1.

    """
    atlas_data = atlas_ni.get_data()
    max_features = atlas_ni.get_data().reshape(-1, atlas_ni.shape[-1] ).max(axis=0)
    std_data = (np.abs(atlas_data) / max_features).reshape(atlas_ni.shape)
    return image.new_img_like(atlas_ni,std_data )

def test_standardize_atlas():
    allen = datasets.fetch_atlas_allen_2011()
    atlas_ni = image.load_img(allen.rsn28)
    std_atlas_ni = standardize_atlas(atlas_ni)
    assert((0 <= std_atlas_ni.get_data()).all() & (std_atlas_ni.get_data() <= 1.0).all())
    assert(std_atlas_ni.shape == atlas_ni.shape)


def contrast_weights_from_regions(atlas_ni=None, region_weights=[]):
    if atlas_ni == None:
        allen = datasets.fetch_atlas_allen_2011()
        atlas_ni = image.load_img(allen.rsn28)

    std_atlas_ni = standardize_atlas(atlas_ni)
    n_components = atlas_ni.shape[-1]
    if len(region_weights) != n_components:
        region_weights = np.zeros(n_components)
        region_weights[0] = 1
    atlas_a = std_atlas_ni.get_data()
    atlas_a *= region_weights
    return image.new_img_like(std_atlas_ni, atlas_a)

def test_contrast_weights_from_regions():
    """
    sanity check for contrast_weights_from_regions
    :return: nothing, just a test
    """
    img = contrast_weights_from_regions()
    # ensure all regions besides the first one are set to zero
    assert( img.get_data()[:,:,:,1:].sum() == 0.0)

    # verify the allen atlas is similar (modulo summation) to our weighted version
    allen = datasets.fetch_atlas_allen_2011()
    atlas_ni = image.load_img(allen.rsn28)
    std_atlas_ni = standardize_atlas(atlas_ni)
    assert( img.get_data().sum() == std_atlas_ni .get_data()[:,:,:,0].sum())

def predictedImg_from_betaImg_designMatrix(beta_img, design_matrix, con_params_l = []):
    """

    :param a 4d Nifti1Image with design matrix coefficients:
    :param design_matrix:
    :param con_params_l: a (currently) binary list to select which parameters to use
            in our projection. Could be a matrix? so that the weights were voxel dependent.

    :return: the projected image.
    """


    # con_params is a list indicating which contrasts we will be taking from the model
    # initially we take all of them
    if len(con_params_l) == 0:
        con_params_l = np.ones(design_matrix.shape[1], dtype=np.bool)

    con_mask = NiftiMasker()
    masked_con_img = con_mask.fit_transform(beta_img)

    # ok, could put in something to change p to a (p,v)-matrix  with
    # 'p,v->pv' and then apply
    # 'pv,pv -> pv'
    # but first we want,
    # resample to get atlas in right size !
    # image.resample_to_img
    select_masked_con_img = np.einsum('p,pv->pv', con_params_l, masked_con_img)
    pred_a = np.einsum('tp,pv->tv',
                       design_matrix.values,
                       select_masked_con_img)
    new_img = con_mask.inverse_transform(pred_a)
    return new_img

from nistats import reporting


import pytest

def get_contrasts():
    l = list()
    l.append( np.ones(13))
    l.append( [1,1,1,0,0,0,0,0,0,0,0,0,1])
    return l

from nistats.first_level_model import first_level_models_from_bids

# thought that I could standardize here, and then use model.masker_ to
def get_data():
    n_runs = 3
    data_dir = '/Users/josephmann/nilearn_data/bids_langloc_example/bids_langloc_dataset/'
    task_label = 'languagelocalizer'
    space_label = 'MNI152nonlin2009aAsym'
    derivatives_folder = 'derivatives'
    res_l = list()
    for noise_model in ['ols','ar1']:
        for standardize_b in [ False]: # never standardize
            models, models_run_imgs, models_events, models_confounds = \
                first_level_models_from_bids(
                    data_dir, task_label, space_label, #smoothing_fwhm=5.0,
                    derivatives_folder=derivatives_folder,
                    noise_model=noise_model, signal_scaling=False, standardize=standardize_b)
            res_l.extend(list(zip(models, models_run_imgs, models_events, models_confounds))[:n_runs])
    return res_l

@pytest.fixture(params= get_data())
def  beta_img_design_matrix(request):
    model, imgs, events, confounds = request.param
    return beta_img_from_model_events_confounds(model, imgs, events, confounds) + (imgs[0], model)

@pytest.mark.parametrize("contrasts_l", get_contrasts())
def test_contrasts(beta_img_design_matrix, contrasts_l):
    """
    :param beta_img_design_matrix: a tuple having a
        (4d Nifti1Image with design matrix parameter amplitudes in each voxel,
        Dataframe having the design matrix
        4D NiftiImage which is the image that is modeled with our beta values and design matrix
        a nistats.FirstLevelModel - currently not used...
    :param contrasts_l: when we project our new image we can select with parameters we want to use using this list
        TODO should be fixed length relative to design matrix, also these could just be weights... or even better
        a matrix that is (param x (region or voxel)
    :return: nothing, we assert that correlation between the two images is greater than 0.9
    """
    beta_img0, design_matrix, img0, model = beta_img_design_matrix
    contrasts_l = np.array(contrasts_l).astype(np.bool)
    new_img = predictedImg_from_betaImg_designMatrix(beta_img0, design_matrix, contrasts_l)
    img_j = nilearn.image.load_img(img0)
    masker = nilearn.input_data.NiftiMasker()
    masker.fit(new_img) # trying background masking_strategy on new_img (should be same as 'epi' on previous
    corr = reporting.compare_niimgs([new_img], [img_j], masker, plot_hist=False)
    print(corr)
    assert( corr[0] > 0.9)


