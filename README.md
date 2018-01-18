# Imaging Simulations

Primary resource will be: [Nilearn tutorial using simulations](http://nilearn.github.io/auto_examples/02_decoding/plot_simulated_data.html)

In the above nilearn tutorial, code is given to create simulated brain images which are then used to evaluate different classifiers. 
More specifically, a visually simple voxel weighting is created (variable $w$ in the code) and then used with a random signal, $y$ to created a matrix $X$ that represents a time series of voxels. Additionally, a noise component $XX$ is added to the matrix $X$. The tutorial then applies different algorithms that attempt to recreate the inital voxel weighting from the noisy voxel time series and the initial signal, $y$. Because $w$ is easily recognizable, the disparity of the recreated $w'$ is visual apparent.

In order for our simulation to be more realistic, we would replace the recognizable voxel weighting, $w$, with a weighting that is a a "realistic" function of a certain kind of stimulus (eg. occipital lobes to images) as well genetic influences eg. SNP's.

To do:
- map specific brain regions to collections of voxels 
- map stimuli to brain regions
- map stimuli x SNP to brain regions
  - presum ably certain genes will affect cognition.
  

#### Update: January 17, 2018

Big digression into nistats in order to use nistats model building code.
Currently have file `example_new_image_from_nibs.py`, acquires data in nibs format,
generate contrast maps for every variable in the design matrix, and then 
creates a new image using theses contrast maps and the design matrix.

