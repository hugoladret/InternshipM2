#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 15:54:38 2018

@author: hugo
"""
import numpy as np
import matplotlib.pyplot as plt   
from joblib import Parallel, delayed
import pyNN.nest as sim
from pyNN.random import RandomDistribution as rnd
from pyNN.utility.plotting import Figure, Panel


class Net_Parameters :
    def __init__(self, ring=True, recurrent=False, seed=42, source='poisson', DEBUG=1):
        self.seed = seed
        self.source = source
        self.recurrent = recurrent
        self.ring = ring
        self.DEBUG = DEBUG

        #c : connectivity probability
        #w : global weight, value for every connections
        #i_rate : input rate, mean firing rate of source population neurons
        #w_in : weight of connections between source population and excitatory neurons population
        #s : synaptic delay
        #g : inhibition-excitation coupling
        #p : excitatory neurons percentage in network
        #n_model : neuron dynamic model
        #b_input : input orientation distribution bandwidth
        #angle_input : the most represented orientation angle in input distribution
        #b_xx : orientation selectivity for a projection xx
        '''
        Parameters : Network parameters, Neuron parameters, Simulation parameters,
        '''
        #a.General network params
        #Ring network
        if self.ring:
            self.c = 1.
            if self.recurrent :
                self.w, self.w_inh, w_input_exc = .2, .3, .3
                self.g = 3.
            else:
                self.w, self.w_inh, w_input_exc = .1, .0, .3
                self.g = 0.
        #Non ring network
        else:
            self.c = 0.15
            if self.recurrent :
                self.w, self.w_inh, w_input_exc = .25, .2, .3
                self.g = 1.
            else:
                self.w, self.w_inh, w_input_exc = .0, .0, .3
                self.g = 0.
                
        #b.Point neuron params        
        self.cell_params = {
        'tau_m'      : 20.0,   # (ms)
        'tau_syn_E'  : 2.0,    # (ms)
        'tau_syn_I'  : 4.0,    # (ms)
        'e_rev_E'    : 0.0,    # (mV)
        'e_rev_I'    : -70.0,  # (mV)
        'tau_refrac' : 2.0,    # (ms)
        'v_rest'     : -60.0,  # (mV)
        'v_reset'    : -70.0,  # (mV)
        'v_thresh'   : -50.0,  # (mV)
        'cm'         : 0.5,    # (nF)
        }
        
        #c.Simulation params
        self.sim_params = {
        'simtime'     : 1000,   #(ms)
        'input_rate'  : 10.,   #(Hz)
        'b_input'     : np.inf,#infinite
        'angle_input' : 90,    #degrees

        'nb_neurons'  : 1000,  #neurons number
        'p'           : .5,        #inhibitors rate in the population
        'neuron_model': 'IF_cond_exp',    #point neuron model
        'v_init_min'   : -53.5,  # (mV)
        'v_init_max'   : -49.75,  # (mV)

        #'c_input_exc' : 1., #self.c*10,
        'c_input_inh' : 0, #self.c*0.,
        'w_input_exc' : w_input_exc,
        
        #delays ?? in ms
        's_input_exc' : 1,
        's_input_inh' : 1,
        's_exc_inh'   : 1,
        's_inh_exc'   : 1,
        's_exc_exc'   : 1,
        's_inh_inh'   : 1,
            
        #B_thetas for the ring ?
        'b_exc_inh'   : np.inf,
        'b_exc_exc'   : np.inf,
        'b_inh_exc'   : np.inf,
        'b_inh_inh'   : np.inf,
        
        #connectivity patterns
        'c_exc_inh'   : self.c,
        'c_inh_exc'   : self.c,
        'c_exc_exc'   : self.c,
        'c_inh_inh'   : self.c,

        #synaptic weight (µS)
        'w_exc_inh'   : self.w*self.g,
        'w_inh_exc'   : self.w_inh,
        'w_exc_exc'   : self.w,
        'w_inh_inh'   : self.w_inh*self.g,
        }
        
        #optimized params for ring
        if self.ring :
            self.sim_params['b_input'] = 10.
            self.sim_params['b_exc_inh'] = 5.
            self.sim_params['b_exc_exc'] = 10.
            self.sim_params['b_inh_exc'] = 40.
            self.sim_params['b_inh_inh'] = 10.

class RNN_Simulation:
    def __init__(self, sim_params=None, cell_params=None, verbose = True):
        '''
        Parameters : Stimulus, Population, Synapses, Recording, Running 
        '''
        
        self.verbose = verbose
        self.sim_params = sim_params
        self.cell_params = cell_params
        
        sim.setup()#spike_precision='on_grid')#timestep = .1)
        
        N_inh = int(sim_params['nb_neurons']*sim_params['p']) #total pop * proportion of inhib
        self.spike_source = sim.Population(N_inh, sim.SpikeSourcePoisson(rate=sim_params['input_rate'],
                                                                        duration=sim_params['simtime']/2))
        
        #orientation stimulus, see bottom section of notebook
        angle = 1. * np.arange(N_inh)
        rates = self.tuning_function(angle, sim_params['angle_input']/180.*N_inh, sim_params['b_input'], N_inh)
        rates /= rates.mean()
        rates *= sim_params['input_rate']
        for i, cell in enumerate(self.spike_source):
            cell.set_parameters(rate=rates[i])
        
        #neuron model selection
        if sim_params['neuron_model'] == 'IF_cond_alpha':
            model = sim.IF_cond_alpha #LIF with nice dynamics
        else: 
            model = sim.IF_cond_exp #LIF with exp dynamics
        
        #populations
        E_neurons = sim.Population(N_inh,
                                   model(**cell_params),
                                   initial_values={'v': rnd('uniform', (sim_params['v_init_min'], sim_params['v_init_max']))},
                                   label="Excitateurs")
        I_neurons = sim.Population(int(sim_params['nb_neurons'] - N_inh),
                                   model(**cell_params),
                                   initial_values={'v': rnd('uniform', (sim_params['v_init_min'], sim_params['v_init_max']))},
                                   label="Inhibiteurs")
        
        #input to excitatories
        input_exc = sim.Projection(self.spike_source, E_neurons,
                                sim.OneToOneConnector(),
                                sim.StaticSynapse(weight=sim_params['w_input_exc'], delay=sim_params['s_input_exc'])
                                )
        
        #loop through connections type and use associated params, can be a bit slow
        conn_types = ['exc_inh', 'inh_exc', 'exc_exc', 'inh_inh']   #connection types
        '''
        self.proj = self.set_synapses(conn_types = conn_types, sim_params =sim_params, 
                                      E_neurons = E_neurons, I_neurons = I_neurons, 
                                      N_inh = N_inh)
        '''
        #Multi threading support NE MARCHE PAS LAISSER LE NJOBS EN 1
        self.proj = Parallel(n_jobs =1, backend = 'multiprocessing')(delayed(self.set_synapses)(conn_type,sim_params =sim_params, 
                                      E_neurons = E_neurons, I_neurons = I_neurons, 
                                      N_inh = N_inh, conn_types = conn_types,
                                        verbose = verbose) for conn_type in range(len(conn_types)))
        if verbose :print('Done building synapses !')
            
        #record
        self.spike_source.record('spikes')
        E_neurons.record('spikes')
        I_neurons.record('spikes')
        
        #run
        if verbose : print('Running simulation..')
        sim.run(sim_params['simtime'])
        if verbose : print('Done running !')
        
        #get the spikes
        self.E_spikes = E_neurons#.get_data().segments[0]
        self.I_spikes = I_neurons#.get_data().segments[0]
        self.P_spikes = self.spike_source#.get_data().segments[0]
        
    def set_synapses(self, conn_type, sim_params, E_neurons, I_neurons, N_inh, conn_types, verbose):
        syn = {}
        proj = {}
        verbose = True

        if verbose : print('Building %s synapses..' % conn_types[conn_type])
        weight = sim_params['w_{}'.format(conn_types[conn_type])]
        delay=sim_params['s_{}'.format(conn_types[conn_type])]
        syn[conn_types[conn_type]] = sim.StaticSynapse(delay=delay)

        if conn_types[conn_type][:3]=='exc': #string slicing, this co is TO exc
            pre_neurons = E_neurons
            receptor_type='excitatory'
        else:
            pre_neurons = I_neurons #TO inh
            receptor_type='inhibitory'
        if conn_types[conn_type][-3:]=='exc': #FROM exc
            post_neurons = E_neurons
        else:
            post_neurons = I_neurons #FROM inh

        sparseness = sim_params['c_{}'.format(conn_types[conn_type])]
        proj[conn_types[conn_type]]  = sim.Projection(pre_neurons, post_neurons,
                                        connector=sim.FixedProbabilityConnector(sparseness, rng=sim.NumpyRNG(seed=42)),
                                        synapse_type=syn[conn_types[conn_type]],
                                        receptor_type=receptor_type)
        bw = sim_params['b_{}'.format(conn_types[conn_type])]
        angle_pre = 1. * np.arange(proj[conn_types[conn_type]].pre.size)
        angle_post = 1. * np.arange(proj[conn_types[conn_type]].post.size)
        w_ij = self.tuning_function(angle_pre[:, np.newaxis], angle_post[np.newaxis, :], bw, N_inh)*weight
        proj[conn_types[conn_type]].set(weight=w_ij)
        
        return proj
        
    def tuning_function(self, i, j, B, N):
        if B==np.inf:
            VM = np.ones_like(i*j)
        else:
            VM = np.exp((np.cos(2.*((i-j)/N*np.pi))-1)/(B*np.pi/180)**2)
        VM /= VM.sum(axis=0)
        return VM
    


class PlotTwist:
    def __init__(self, sim_params = None, cell_params = None, verbose = True):
        self.verbose = verbose
        self.sim_params = sim_params
        self.cell_params = cell_params
        
    #RasterPlot using  pyNN    
    def RasterPlot(self, SpikesP, SpikesE, SpikesI, title = 'Title', markersize = .5):
        fig = Figure(
            Panel(SpikesP.spiketrains, xticks = False, ylabel = 'Input', color = 'k', markersize = markersize),
            Panel(SpikesE.spiketrains, xticks = False, ylabel = 'Excitatory', color = 'r', markersize = markersize),
            Panel(SpikesI.spiketrains, xticks = True, xlabel = 'Time(ms)', ylabel = 'Inhibitory',
                  color = 'b', markersize = markersize),
            title = title, settings = {'figure.figsize': [9., 6.]})
        
        '''for ax in fig.fig.axes:
            ax.set_xticks(np.linspace(0, self.sim_params['simtime'], 6, endpoint = True))'''
        fig.fig.subplots_adjust(hspace = 0)
        return fig
    
    def VanillaRasterPlot(self, SpikesP, SpikesE, SpikesI, title = 'Title', markersize = .5):
        def plot_spiketrains(ax, spikes, color, ylabel, do_ticks): #oh la jolie double fonction
            for spiketrain in spikes.spiketrains:
                y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
                ax.scatter(spiketrain, y, color = color, s = markersize)
                ax.set_ylabel(ylabel)
                if not do_ticks : ax.set_xticklabels([])
            
        fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize = (9,6))
        plot_spiketrains(axs[0], SpikesP, color = 'k', ylabel = 'Neuron Index', do_ticks = False)
        axs[0].set_title('Poisson input')
        plot_spiketrains(axs[1], SpikesI, color = 'b', ylabel = 'Neuron Index', do_ticks = False)
        axs[1].set_title('Inhibitory population')
        plot_spiketrains(axs[2], SpikesE, color = 'r', ylabel = 'Neuron Index', do_ticks = True)
        axs[2].set_title('Excitatory population')
        plt.xlabel('Time(ms)')
        
    #TODO finish this method
    def OneParamVar_RasterPlot(self, spikesP, spikesE, spikesI, title = 'Title', markersize = .5,
                              var_name = None, var_values = None, force_int = False):
        '''
        var_name is the variable name as it is in sim_param (otherwise it won't work)
        
        if force_int : values = [int(i) for i in values]
        
        for i, value in enumerate(var_values):
            sim_params[var_name] = value
            self.RasterPlot(SpikesP, SpikesE, SpikesI, title = 'Variable %s : %s' %(var_name, value))
        '''