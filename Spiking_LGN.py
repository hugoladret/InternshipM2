#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:32:50 2018

@author: Hugo Ladret, INT-CNRS

This file contains all the methods built from 10/10/18 to 7/11/18 ;
    Spatiotemporal filters based on Cai and Marr,
    Signal convolution methods,
    Natural image pre-processing,
    V1-like gabor generations,
    Optimal filter distribution in a natural image
    
Running the main input_to_currents() function transforms an input image or movie
into currents produced by simulated LGN cells (actually ST filters), which can
be plugged into a spiking neural net simulator to generate orientation-selective
V1 neuron.

The driving idea behind this code is to treat the retina+LGN as an input 
pre-processing layer, an idea that was already Python-ed in NeuralEnsemble GSoC blog :
http://neuralensemble.blogspot.com/2014/06/gsoc-open-source-brain-retinal-filter-i.html
http://neuralensemble.blogspot.com/2014/06/gsoc-open-source-brain-retinal-filter-ii.html
We use the same code for spatial filters and an alike temporal filter but expand
upon it.

Supports multithreading ! 

TODO : Lower sampling rate in option to fasten calculations
"""

import numpy as np
from joblib import Parallel, delayed
from LogGabor import LogGabor


def input_to_currents(video, FPS, total_time,
                      distrib_size, safeguard_offset, random_shift, grid_res,
                      N_theta, B_theta, sf_0, B_sf,
                      on_thresh, off_thresh,
                      filter_size, filter_res, sampling_rate,
                      n_jobs, backend, mt_verbose,
                      off_gain = 1,
                      gabors_params=None,
                      verbose=True,
                     get_coordinates = False):
    '''
    Main method, transforms a numpy array input into currents that can be used
    by a spiking neuron simulator (optimized for Nest with PyNN)

    Args :
        > Input arguments <
        video : the numpy array input, a video of shape H x W x frames
                the format is for plt.imshow convenience, careful about the H x W order 
        FPS : video's frame per second
        total_time : simulation total time, should you want it shorter than video's length

        > Gabor parameters <
        gabors_params : if set to None, the default parameter dictionnary is used

        > Generating gabors centers and filters coordinates <
        distrib_size : the size of a group of RF distribution (ie size of the gabor)
        safeguard_offset : an offset from the borders to prevent filters outside the image
        random_shift : shift each center by a random factor to make our retina distribution
                        slightly noisy
        N_theta = number of theta to do
        grid_res : resolution of the filter grid, passed in a np.mgrid
        B_theta, sf_0, B_sf : Parameters for the LogGabor shape
                            B_theta is the opening of the gabor, 
                            sf_0 is the spatial frequency 
                            b_sf is the bandwidth frequency
        on_thresh, off_thresh : On center and off center threshold for generating coordinates

        > Current generation <
        filter_size, filter_res = spatiotemporal filters arguments
                            other parameters have been optimized 
                            and are left by default for now
        sampling_rate = lower the size of the linspace to fasten computations
                        expressed as a fraction of the original sampling rate
                        (1000Hz), i.e. sampling_rate = 1 samples at 1000hz
                        and sampling_rate = 2 samples at 500Hz
                            so it's more a sampling rate divider rather than a 
                            real sampling rate in Hz
        off_gain = multiplies the off currents by a certain factor, otherwise there is usually
                    a massive difference due to the gaussian shape.
                    " auto " multiplies the off by the radio off np.max differences, with a .7 multiplication afterwards
                    otherwise an int does the multiplication

        > Multithreading <
        n_jobs = Number of workers to be used by joblib
        backend = Joblib's backend, loky by default
        mt_verbose = joblib's level of verbosity (10 by default)

        > Extras <
        verbose : Various print statement for sanity checks
        get_coordinates : Boolean, wether the function will return the coordinates or not
    '''

    if gabors_params == None:
        gabors_params = {'B_sf': 0.4, 'B_theta': 0.17453277777777776, 'N_X': 256, 'N_Y': 256,
                         'N_image': 100, 'base_levels': 1.618, 'datapath': 'database', 'do_mask': True,
                         'do_whitening': True, 'dpi': 450, 'edgefigpath': 'results/edges', 'edgematpath': 'cache_dir/edges',
                         'ext': '.pdf', 'figpath': 'results', 'figsize': 14.0, 'formats': ['pdf', 'png', 'jpg'],
                         'mask_exponent': 3.0, 'matpath': 'cache_dir', 'n_theta': 24, 'noise': 0.1, 'seed': None,
                         'use_cache': True, 'verbose': 0, 'white_N': 0.07, 'white_N_0': 0.0, 'white_alpha': 1.4,
                         'white_f_0': 0.4, 'white_n_learning': 0, 'white_name_database': 'kodakdb',
                         'white_recompute': False, 'white_steepness': 4.0}
    else:
        gabors_params = gabors_params

    if verbose:
        print('Video shape', video.shape)
        print('Frames per second:', FPS)
        print('Frame duration at %s FPS: %.2f ms' % (FPS, total_time/FPS))
        print('Video length inferred from fps: %s s' %
              video[:, :, ::int(FPS)].shape[-1])

    frame_duration = int(total_time/FPS)
    stimuli = []
    for ms in range(int(total_time/frame_duration)):
        for same_frame in range(frame_duration):
            stimuli.append(video[:, :, ms])
    while len(stimuli) < total_time:
        stimuli.append(stimuli[-1])
    if stimuli[0][0, 30] != stimuli[frame_duration+1][0, 30] and verbose:
        print('FPS conversion sanity check passed !\n')

    #print(np.swapaxes(np.asarray(stimuli).T, 0,1).shape)
    stimuli = np.swapaxes(np.asarray(stimuli).T, 0, 1)
    print('Stimuli shape',stimuli.shape)
    centers_coordinates = generate_centers_coordinates(
        distrib_size, safeguard_offset, random_shift, stimuli)

    if verbose:
        print('Generating filters coordinates with gabors ..')
    gabor_coordinates = Parallel(n_jobs=n_jobs, backend=backend, verbose=mt_verbose)(delayed(generate_gabors_coordinates)
                                     (theta,
                                      params=gabors_params,
                                      N_X=stimuli.shape[1], N_Y=stimuli.shape[0],
                                      centers_coordinates=centers_coordinates,
                                      B_theta=B_theta, sf_0=sf_0, B_sf=B_sf,
                                      on_thresh=on_thresh, off_thresh=off_thresh,
                                      distrib_size=distrib_size, grid_res=grid_res)
                                     for theta in np.linspace(0, np.pi/2, N_theta, endpoint = True))

    if verbose:
        print('Done ! Generating currents from filters ..')

    currents_generated = Parallel(n_jobs=n_jobs, backend=backend, verbose=mt_verbose)(delayed(coordinates_to_currents_multithread)
                                      (filters,
                                       stimuli=stimuli, total_time=total_time,
                                       filter_size=filter_size, filter_res=filter_res,
                                       sampling_rate = sampling_rate,
                                       off_gain = off_gain)
                                      for filters in gabor_coordinates)
    if get_coordinates :
        return currents_generated, gabor_coordinates
    else :
        return currents_generated


def generate_centers_coordinates(distrib_size, safeguard_offset, random_shift, video):
    '''
    Generate gabor center coordinates in a wide-spread way but constrained from
    the image edge, in order to avoid filter elimination during current generation

    Args : 
        distrib_size : the size of a group of RF distribution (ie size of the gabor)
        safeguard_offset : an offset from the borders to prevent filters outside the image
        random_shift : shift each center by a random factor to make our distribution a bit noisier

        video : input video, used to get the shape of the spreading

    '''

    if safeguard_offset < distrib_size:
        print(
            'Warning : offset lower than distribution, risk of filters being eliminated\n')

    # the number of gabors distributed horizontally
    Nx_gabors = int((video.shape[1]-safeguard_offset)/distrib_size)
    # X coordinates of the gabors
    Xs = np.linspace(safeguard_offset,
                     video.shape[1]-safeguard_offset, Nx_gabors)

    # number of gabors distributed vertically
    Ny_gabors = int((video.shape[0]-safeguard_offset)/distrib_size)
    # Y coordinates of the gabors
    Ys = np.linspace(safeguard_offset,
                     video.shape[0]-safeguard_offset, Ny_gabors)

    # now we mash them together and add some noise
    Xs = np.repeat(Xs, Ny_gabors)
    Ys = np.tile(Ys, Nx_gabors)
    if random_shift != 0:
        Xs = Xs + np.random.randint(-random_shift, random_shift, Xs.shape)
        Ys = Ys + np.random.randint(-random_shift, random_shift, Ys.shape)

    centers_coordinates = np.array([Xs, Ys])

    return centers_coordinates

#############
# ST Filters
#############


def spatial_filter(wx=5., wy=5., xres=.1, yres=.1, sigma_center=1., sigma_surround=1.2,
                   x_trans=0, y_trans=0, theta=0, gain=-1.):
    '''
    A spatial filter mimicking ON/OFF Marr-like receptor field, using difference of 
    gaussians (DOG) approach, from NeuralEnsemble amazing GSoC 

    Args :
        wx, wy : x/y filter width
        xres, yres : x/y filter resolution

        sigma_center : U.A related to the center of the circle
        sigma_surround : U.A related to the surround of the circle
        Both are best left at a ratio of .85, as this produce a nice DOG

        x_trans, y_trans : spatial translation of the filter
        gain : filter multiplier, makes it an ON center (negative value) or OFF center (positive)
    '''
    x = np.arange((-wx+x_trans), (wx+x_trans), xres)
    y = np.arange((-wy+y_trans), (wy+y_trans), yres)
    X, Y = np.meshgrid(x, y)

    radius = np.sqrt((X-x_trans)**2 + (Y-y_trans)**2)
    center = (1.0 / sigma_center**2) * np.exp(-.5*(radius / sigma_center)**2)
    surround = (1.0 / sigma_surround**2) * \
        np.exp(-.5*(radius / sigma_surround)**2)

    Z = surround - center
    Z *= gain
    return X, Y, Z


def temporal_filter(t=1,
                    K1=.92, c1=0.2, n1=7, t1=-6,
                    K2=.2, c2=.12, n2=8, t2=-6,
                    baseline=0., gain=90):
    '''
    A temporal filter, equations parameters based on Cai 98 (J.Neurophy) and code based
    on NeuralEnsemble GSoC

    Args :
        t = time at which the temporal response is calculated

        K, c, n, t = parameters of both gaussians, see code or wikipedia for more details

        baseline = level at which the filter is initialized
            if != 0, can cause a drift in the long run
        gain = filter multiplier, used to match Cai's recordings values
    '''

    p = baseline
    p += K1 * ((c1 * (t - t1))**n1 * np.exp(-c1 * (t-t1))) / \
        (n1**2)*np.exp(-n1)
    p -= K2 * ((c2 * (t - t2))**n2 * np.exp(-c2 * (t-t2))) / \
        (n2**2)*np.exp(-n2)
    return p*gain


##########
# Gabors
##########
def gabor_connectivity(filters, phi, theta, threshold, on=True):
    '''
    From an array of filters and a gabor phi-space, returns the ON/OFF filters
    coordinates

    Args :
        Filters (ndarray) : A 2D array of filters to localize on the gabor
        Phi : A phi space from a LogGabor
        Threshold : the threshold above which (or below, given 'on' param) we select the filters
        on : True if we're passing on center filters, False if we're doing off center filters
    '''
    if on:
        gab_above_threshold = np.array(
            [*np.where(phi[theta, 0] > threshold)], dtype=float).T
    else:
        gab_above_threshold = np.array(
            [*np.where(phi[theta, 0] < threshold)], dtype=float).T

    filters_in_gabor = []
    for it, filt in enumerate(filters):
        for gabs in gab_above_threshold:
            if np.all(filt.astype(int) == gabs):
                filters_in_gabor.append(filt)

    return filters_in_gabor


def generate_gabors_coordinates(theta, params, N_X, N_Y, centers_coordinates,
                                B_theta=15, sf_0=.05, B_sf=.5,
                                distrib_size=8, grid_res=3,
                                on_thresh=.1, off_thresh=-.1,
                                verbose=True):
    '''
    Given some gabor parameters, a set of coordinates for centering gabors, returns a set of 
    coordinates for filters belonging into the gabors


    Params :
        theta : gabor theta angle
        params : the default parameters dictionnary for the gabor generation
        N_X, N_Y : Gabor size, usually the same as the video

        centers_coordinates : a 2D array giving the centers of each gabor

        B_theta, sf_0, B_sf : Parameters for the LogGabor shape
                            B_theta is the opening of the gabor, 
                            sf_0 is the spatial frequency 
                            b_sf is the bandwidth frequency

        distrib_size : the size of each group of filters, in image coordinates
        grid_res : resolution of the group of filters, passed in a np.mgrid

        on_thresh, off_thresh : threshold at which a filter is selected to 
                                be on/off, by scanning the Gabor phi-space

        verbose : display the filter size as a sanity check  
    '''

    xs = centers_coordinates[0]
    ys = centers_coordinates[1]
    nbr_gabors = len(xs)

    N_X = int(N_X)
    N_Y = int(N_Y)
    N_phase = 2

    lg = LogGabor(params)
    lg.set_size((N_X, N_Y))

    B_theta = B_theta / 180 * np.pi
    params = {'sf_0': sf_0, 'B_sf': B_sf, 'B_theta': B_theta}
    params.update(theta=theta)

    phi = np.zeros((1, N_phase, N_X, N_Y))

    filters_per_gab = []
    for gab in range(nbr_gabors):
        x = xs[gab]
        y = ys[gab]

        for i_phase in range(N_phase):
            phase = i_phase * np.pi/2
            kernel = lg.invert(lg.loggabor(
                x, y, **params)*np.exp(-1j*phase))
            phi[0, i_phase, :] = lg.normalize(kernel)

        fx_min = x - distrib_size
        fx_max = x + distrib_size
        fy_min = y - distrib_size
        fy_max = y + distrib_size
        filters_coordinates = np.mgrid[fx_min:fx_max:grid_res,
                                       fy_min:fy_max:grid_res].reshape(2, -1).T
        if verbose and gab == 0:
            print('Thread started !\nFilter grid shape',
                  filters_coordinates.shape, '\n')

        filters_in_gabor = gabor_connectivity(filters=filters_coordinates,
                                              phi=phi, theta=0, threshold=on_thresh)
        off_filters_in_gabor = gabor_connectivity(filters=filters_coordinates,
                                                  phi=phi, theta=0, threshold=off_thresh, on=False)

        filters_per_gab.append((filters_in_gabor, off_filters_in_gabor))

    return filters_per_gab


#####################
# Current generation
#####################
def coordinates_to_currents_multithread(filters,
                                        stimuli, total_time, filter_size, filter_res,
                                        sampling_rate,
                                        off_gain):
    '''
    Creates currents from spatiotemporal filters in given coordinates set, by convolving
    the apex of the ST filter with the receptive field in a fed video

    Args :
        filters_ : an array containing [0]on centers and [1]off centers coordinates

        stimuli = a video transformed into a numpy array (see the associated .py file)
                    must be of (W x H x len) shape

        total_time = total time of the current simulation, usually equal to the stimuli length

        filter_size, filter_res = spatiotemporal filters arguments
                            other parameters have been optimized 
                            and are left by default for now
                            
        sampling_rate = see main function documentation
    '''

    levels_per_gabor = []


    for gabors in filters:
        filters_in_gabor, off_filters_in_gabor = gabors[0], gabors[1]



        # On filters iteration

        on_st_levels = []

        for filt in range(len(filters_in_gabor)):
            X, Y, Z = spatial_filter(wx=filter_size, wy=filter_size,
                                     xres=filter_res, yres=filter_res,
                                     x_trans=filters_in_gabor[filt][0],
                                     y_trans=filters_in_gabor[filt][1])
            t = [temporal_filter(i)
                 for i in np.linspace(0, total_time, int(total_time/sampling_rate))]

            # Convolution coordinates
            X_convo_minus = int(filters_in_gabor[filt][0]-filter_size)
            X_convo_plus = int(filters_in_gabor[filt][0]+filter_size)
            Y_convo_minus = int(filters_in_gabor[filt][1]-filter_size)
            Y_convo_plus = int(filters_in_gabor[filt][1]+filter_size)
            n_downscale = int(Z.shape[0]/(X_convo_plus-X_convo_minus))
            
            activations = []
            for i in np.linspace(0, total_time-1, int(total_time/sampling_rate)):
                #Kron upscaling and filter max-ed convolution
                arr = stimuli[:, :, int(i)][X_convo_minus:X_convo_plus, Y_convo_minus:Y_convo_plus]
                activations.append(np.max(homemade_kronecker(arr, n_downscale) * Z))

            ys = np.convolve(t, activations)
            st = Z[:, :, None] * ys[None, None, :]

            on_st_level = []
            for i in range(int(total_time/sampling_rate)):
                on_st_level.append(np.max(st[:, :, i]) * sampling_rate)
            on_st_levels.append(on_st_level)
            

        # Off filters iterations

        off_st_levels = []

        for filt in range(len(off_filters_in_gabor)):
            X, Y, Z = spatial_filter(wx=filter_size, wy=filter_size,
                                     xres=filter_res, yres=filter_res,
                                     x_trans=off_filters_in_gabor[filt][0],
                                     y_trans=off_filters_in_gabor[filt][1],
                                     gain=1.)
            t = [temporal_filter(i)
                 for i in np.linspace(0, total_time, int(total_time/sampling_rate))]

            X_convo_minus = int(off_filters_in_gabor[filt][0]-filter_size)
            X_convo_plus = int(off_filters_in_gabor[filt][0]+filter_size)
            Y_convo_minus = int(off_filters_in_gabor[filt][1]-filter_size)
            Y_convo_plus = int(off_filters_in_gabor[filt][1]+filter_size)
            n_downscale = int(Z.shape[0]/(X_convo_plus-X_convo_minus))
            
            activations = []
            for i in np.linspace(0, total_time-1, int(total_time/sampling_rate)):
                #Kron upscaling and filter max-ed convolution
                arr = stimuli[:, :, int(i)][X_convo_minus:X_convo_plus, Y_convo_minus:Y_convo_plus]
                activations.append(np.min(homemade_kronecker(arr, n_downscale) * Z))

            ys = np.convolve(t, activations)
            st = Z[:, :, None] * ys[None, None, :]


            off_st_level = []
            for i in range(int(total_time/sampling_rate)):
                off_st_level.append(np.max(st[:, :, i]) * sampling_rate)
            off_st_levels.append(off_st_level)

        if off_gain == 'auto' :
            off_filters_gain = .7*(np.max(on_st_levels)/np.max(off_st_levels))
        elif type(off_gain) == int or type(off_gain)==float:
            off_filters_gain = off_gain
            
        levels_per_gabor.append((np.asarray(on_st_levels),
                                 np.asarray(off_st_levels)*off_filters_gain))

    return levels_per_gabor


def homemade_kronecker(arr, n):
    '''
    Homemade kronecker product of two arrays, used here to upscale the simulus
    It's 12 µs faster than numpy's, which is significant given how many kron product
    we'll have to do
    '''
    
    kr = np.repeat(arr, n* np.ones(arr.shape[0], np.int), axis = 0)
    kr = np.repeat(kr, n* np.ones(arr.shape[0], np.int), axis = 1)
    
    return kr