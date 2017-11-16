# Imaging_Simulations

Primary resource will be: [Nilearn tutorial using simulations](http://nilearn.github.io/auto_examples/02_decoding/plot_simulated_data.html)

In the above tutorial, code is given to create simulations on which different classifiers are then tested. 
More specifically, a visually simple voxel weighting is created and then used with a random signal to created a matrix that  represents a time series of voxels. The tutorial then applies different algorithms that attempt to recreate the inital voxel weighting from the voxel time series and the initial random signal. 

In order for our simulation to be more realistic, we would replace the recognizable voxel weighting with a weighting that is a a "realistic" function of a certain kind of stimulus (eg. occipital lobes to images) as well genetic influences eg. SNP's.

To do:
- map specific brain regions to voxels 
- map stimuli to brain regions
- map stimuli x SNP to brain regions
