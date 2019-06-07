from __future__ import print_function # Always do this >:( 
from __future__ import division
from __future__ import unicode_literals

#%load_ext line_profiler

import matplotlib
matplotlib.use("Agg")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.backends.backend_pdf import PdfPages
import itertools
from tqdm import trange
from tqdm import tqdm
import time
import math
from numba import jit
import pandas as pd
import seaborn as sns
from itertools import chain

# joypy
import joypy
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
# Import code for reading in the data from Dark Sage

import pylab
import os
import routines as r
import random

# Warnings are annoying
import warnings
warnings.filterwarnings("ignore")

##### SET PLOTTING DEFAULTS #####
fsize = 26
matplotlib.rcParams.update({'font.size': fsize, 'xtick.major.size': 10, 'ytick.major.size': 10, 'xtick.major.width': 1, 'ytick.major.width': 1, 'ytick.minor.size': 5, 'xtick.minor.size': 5, 'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.linewidth': 1, 'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Times New Roman', 'legend.numpoints': 1, 'legend.columnspacing': 1, 'legend.fontsize': fsize-4, 'xtick.top': True, 'ytick.right': True})

######  =================== #####

def cut_colormap(colormap_name, start, end):
    """
    Remove part of the colormap and create new one as 'cmap'
    
    Parameters:
    ==========
    colormap_name: Name of the colormap. Example: `plt.cm.viridis`
    start: Float from 0 to 1, starting color value.
    end: Float from 0 to 1, has to be bigger than `start`, ending color value.

    Returns:
    =======
    cmap: Eddited colormap.
    """

    # Remove from 0 to 0.15 in the colorbar
    interval = np.hstack([np.linspace(start, end)])
    colors = colormap_name(interval)
    cmap = LinearSegmentedColormap.from_list('name', colors)
    cmap.set_under(color='white')

    return cmap

@jit
def galdtype_darksage(Nannuli=30):
    """
    General structure of the DarkSage, taken from the DarkSage github.
    """

    floattype = np.float32
    Galdesc_full = [
                    ('Type'                         , np.int32),
                    ('GalaxyIndex'                  , np.int64),
                    ('HaloIndex'                    , np.int32),
                    ('SimulationHaloIndex'          , np.int32),
                    ('TreeIndex'                    , np.int32),
                    ('SnapNum'                      , np.int32),
                    ('CentralGalaxyIndex'           , np.int64),
                    ('CentralMvir'                  , floattype),
                    ('mergeType'                    , np.int32),
                    ('mergeIntoID'                  , np.int32),
                    ('mergeIntoSnapNum'             , np.int32),
                    ('dT'                           , floattype),
                    ('Pos'                          , (floattype, 3)),
                    ('Vel'                          , (floattype, 3)),
                    ('Spin'                         , (floattype, 3)),
                    ('Len'                          , np.int32),
                    ('LenMax'                       , np.int32),
                    ('Mvir'                         , floattype),
                    ('Rvir'                         , floattype),
                    ('Vvir'                         , floattype),
                    ('Vmax'                         , floattype),
                    ('VelDisp'                      , floattype),
                    ('DiscRadii'                    , (floattype, Nannuli+1)), 
                    ('ColdGas'                      , floattype),
                    ('StellarMass'                  , floattype),
                    ('MergerBulgeMass'              , floattype),
                    ('InstabilityBulgeMass'          , floattype),
                    ('HotGas'                       , floattype),
                    ('EjectedMass'                  , floattype),
                    ('BlackHoleMass'                , floattype),
                    ('IntraClusterStars'            , floattype),
                    ('DiscGas'                      , (floattype, Nannuli)),
                    ('DiscStars'                    , (floattype, Nannuli)),
                    ('SpinStars'                    , (floattype, 3)),
                    ('SpinGas'                      , (floattype, 3)),
                    ('SpinClassicalBulge'           , (floattype, 3)),
                    ('StarsInSitu'                  , floattype),
                    ('StarsInstability'             , floattype),
                    ('StarsMergeBurst'              , floattype),
                    ('DiscHI'                       , (floattype, Nannuli)),
                    ('DiscH2'                       , (floattype, Nannuli)),
                    ('DiscSFR'                      , (floattype, Nannuli)), 
                    ('MetalsColdGas'                , floattype),
                    ('MetalsStellarMass'            , floattype),
                    ('ClassicalMetalsBulgeMass'     , floattype),
                    ('SecularMetalsBulgeMass'       , floattype),
                    ('MetalsHotGas'                 , floattype),
                    ('MetalsEjectedMass'            , floattype),
                    ('MetalsIntraClusterStars'      , floattype),
                    ('DiscGasMetals'                , (floattype, Nannuli)),
                    ('DiscStarsMetals'              , (floattype, Nannuli)),
                    ('SfrFromH2'                    , floattype),
                    ('SfrInstab'                    , floattype),
                    ('SfrMergeBurst'                , floattype),
                    ('SfrDiskZ'                     , floattype),
                    ('SfrBulgeZ'                    , floattype),
                    ('DiskScaleRadius'              , floattype),
                    ('CoolScaleRadius'              , floattype), 
                    ('StellarDiscScaleRadius'       , floattype),
                    ('Cooling'                      , floattype),
                    ('Heating'                      , floattype),
                    ('LastMajorMerger'              , floattype),
                    ('LastMinorMerger'              , floattype),
                    ('OutflowRate'                  , floattype),
                    ('infallMvir'                   , floattype),
                    ('infallVvir'                   , floattype),
                    ('infallVmax'                   , floattype)
                    ]
    names = [Galdesc_full[i][0] for i in range(len(Galdesc_full))]
    formats = [Galdesc_full[i][1] for i in range(len(Galdesc_full))]
    Galdesc = np.dtype({'names':names, 'formats':formats}, align=True)
    return Galdesc

def create_G_dataframe(indir, n_files):
	
	"""
	Create DarkSage dataframe 'G'.

	Parameters:
	==========
	indir - String. Path to a directory where the simulation output is storred.
	n_files - Integer. Number of file to be used. (Big millenium has max 512 files; mini millenium has max 8 files)
	
	Returns:
	=======
	G  - Data Frame.

	"""


	n_files = number_of_files
	###### USER NEEDS TO SET THESE THINGS ######
	sim = 1 # which simulation Dark Sage has been run on -- if it's new, you will need to set its defaults below.
	#   0 = Mini Millennium, 1 = Full Millennium, 2 = SMDPL

	fpre = 'model_z0.000' # what is the prefix name of the z=0 files
	files = range(n_files) # list of file numbers you want to read

	Nannuli = 30 # number of annuli used for discs in Dark Sage
	FirstBin = 1.0 # copy from parameter file -- sets the annuli's sizes
	ExponentBin = 1.4
	###### ============================== ######



	##### SIMULATION DEFAULTS #####
	if sim==0:
	    h = 0.73
	    Lbox = 62.5/h * (len(files)/8.)**(1./3)
	elif sim==1:
	    h = 0.73
	    Lbox = 500.0/h * (len(files)/512.)**(1./3)
	elif sim==2:
	    h = 0.6777
	    Lbox = 400.0/h * (len(files)/1000.)**(1./3)
	# add here 'elif sim==3:' etc for a new simulation
	else:
	    print('Please specify a valid simulation.  You may need to add its defaults to this code.')
	    quit()
	######  ================= #####


	##### READ DARK SAGE DATA #####
	DiscBinEdge = np.append(0, np.array([FirstBin*ExponentBin**i for i in range(Nannuli)])) / h
	G = r.darksage_snap(indir+fpre, files, Nannuli=Nannuli)
	######  ================= #####


	# Make a sub_section of dataframe G so that it is controlled by the LenMax
	print('LenMax:', len(G['LenMax']))
	G = G [G['LenMax']>=100 ]
	print('LenMax > 100:', len(G['LenMax']))

	return G

# Start extracting needed indices
def extract_central_indices(G):
	
	"""
	Extract indisces (integers) of central galaxies from DarkSage dataframe.	

	Parameters:
	==========
	G - DarkSage dataframe

	Returns:
	=======
	store_cen_indices - List of central indices, integers.

	"""

	store_cen_indices = []
	store_cen_indices = np.where(G["Type"] == 0)[0]
	return store_cen_indices

def extract_all_indices(G):
	"""
	Extract all indisces (integers) from DarkSage dataframe.	

	Parameters:
	==========
	G - DarkSage dataframe

	Returns:
	=======
	store_all_indices - List of all indices, integers.

	"""

	central_IDs, unique_counts = np.unique(G["CentralGalaxyIndex"], return_counts=True) #find unique ids
	group_offset = np.cumsum(unique_counts) #compute offset between ids

	count = 0
	store_all_indices = []

	argsort_central_gal_idx = np.argsort(G["CentralGalaxyIndex"]) #argsort: array[1,0,-1]; sorted[-1,0,1], argsort[2,1,0]

	for offset in group_offset:
	    inds = np.arange(count, offset) #arrange counter and offsets
	    my_list = argsort_central_gal_idx[inds] #make my list where argsort have their indices
	    
	    store_all_indices.append(my_list)
	    
	    count += len(my_list)

	#print(empty[0:100])

	#check if the sorting is good; it it isn't it sill print output    
	for group in store_all_indices:
	    if not np.all(G["CentralGalaxyIndex"][group] == G["CentralGalaxyIndex"][group][0]):
	        print(G["CentralGalaxyIndex"][group])
	        print(group)

	return store_all_indices

def create_groups_dictionary(store_all_indices):
	"""
	Create groups dictionary which takes all galaxies in a halo.

	Parameters:
	==========
	store_all_indices - List of all indices, integers.
	
	Returns:
	=======
	groups - Dictionary. Keyed by the size of the group, "groups[group_size]"
		Contains list of arrays of a given group_size length.
	
	"""


	groups = {} #initiate groups dictionary

	#i = range(0, len(store_all_indices)) #gives range of halos

	for item in trange(len(store_all_indices)):
	#for item in trange(100):
	    indices = store_all_indices[item]
	    halo_length = len(indices) #gives length of each halo (central+satellite)
	    try:
	        groups[halo_length].append(indices)
	    except KeyError:
	        groups[halo_length] = []
	        groups[halo_length].append(indices)
	#print(groups)    
	return groups



@jit
def create_cen_sat_from_groups_dict(groups, store_cen_indices, my_mass_cutoff=0.06424):
    """
    Created dictionary which is used to extract indices of central and satellite galaxies, taking into account
    the mass cutoff provided. Also creates "Groups" which store information on a group basis and their sat/cen galaxies.
    
    Parameters
    ==========
    
    groups: dictionary. Keyed by the size of the group, ``groups[group_size]``
            Contains list of arrays of a given group_size length. 
    
    Mass_cutoff: Constant.
                 Equal to 0.06424: defined as median stellar mass for halos composed of 100 particles.
    
    store_cen_indices: List of integers
                       Contains indices which correspond to central galaxies
        
    Returns
    =======
    new_groups_dict: Nested dictionary. Keyed by the `groups`.
                     Contains Centrals and Satellites which are in groups where galaxies are above the my_mass_cutoff
                     
    Usage
    =====
    #This is updated dictionary which one can parse to the plotting function:
    updated_dict = create_cen_sat_from_groups_dict(groups, store_cen_indices) 
    
    ---------
    #Select groups of 3 and only central galaxies:
    central_inds = updated_dict[3]["Centrals"]
    G["StellarMass"][central_inds] #gives array of central galaxy stellar masses 
    
    """
    new_groups_dict = {}
    
    for group_size in groups.keys(): #go through the keys in groups dictionary
        central_gals = []
        sat_gals = []
        halo_gals = []
        halo_groups = []
        
        new_groups_dict[group_size] = {} #new dictionary based on group size
        new_groups_dict[group_size]["Groups"] = {} #new dictionary for storring whole groups
        counter = 0
        for group in groups[group_size]:
            
            central_gals_group = [] #store central in "Groups"
            sat_gals_group = [] #store satellites in "Groups"
            
            if (G["StellarMass"][group] < my_mass_cutoff).any(): #creates array of booleans; if theese are true hit continue
                continue
            
            halo_groups.append(group)
            
            # Use np.where to find indices of satellites and central galaxies
            for galaxy in group:
                halo_gals.append(galaxy) #add to halo_gals to know galaxies in the halo
                local_idx = np.where(galaxy == store_cen_indices)[0] # Check if the idx is in the store_cen_indices
                
                if len(local_idx) == 0: # If the idx is not present in store_cen_indisces then it is satellite
                    sat_gals.append(galaxy)
                    sat_gals_group.append(galaxy)
                elif len(local_idx) == 1: # If the idx is present then it is central galaxy
                    central_gals.append(galaxy)
                    central_gals_group.append(galaxy)
                else: # Check for funky stuff and raise an error
                    print("Somehow got a duplicate value in store_cen_indices.  Galaxy is {0}, group is {1} and local_idx value is {2}"
                          .format(galaxy, group, local_idx))
                    raise ValueError
            group_key = "Group_{0}".format(counter) #create group_key based on the groups
            new_groups_dict[group_size]["Groups"][group_key] = {} #new dictionary with all groups
            new_groups_dict[group_size]["Groups"][group_key]["Centrals"] = central_gals_group
            new_groups_dict[group_size]["Groups"][group_key]["Satellites"] = sat_gals_group
            counter += 1
            
                              
        # Append to dictionary found centrals and satellites
        new_groups_dict[group_size]["Centrals"] = central_gals
        new_groups_dict[group_size]["Satellites"] = sat_gals
        new_groups_dict[group_size]["All_galaxies"] = halo_gals
        new_groups_dict[group_size]['All_groups'] = halo_groups
        
        
    return new_groups_dict




# Should be the same with and without LenMax cut if I'm using the updated G dataframe where galaxies are already removed which have smaller LenMax

def plot_len_max(G):
	"""
	Print LenMax size and plot the distribution of galaxies based on their LenMax size.
	
	Parameters:
	==========
	G - DarkSage dataframe

	Returns:
	=======
	Printed LenMax and plot.

	"""
	
	Mlen_all_central = G['LenMax'] [ (G['CentralGalaxyIndex']==G['GalaxyIndex']) & (G['StellarMass']>0.06424)]/h
	print('Number of central galaxies:', len(Mlen_all_central))

	Mlen_all_sat = G['LenMax'] [ (G['CentralGalaxyIndex']!=G['GalaxyIndex']) & (G['StellarMass']>0.06424) ]/h
	print('Number of satellite galaxies', len(Mlen_all_sat))


	# Make a plot of those

	fig = plt.figure(figsize=(10,10))                                                               
	ax = fig.add_subplot(1,1,1)

	plt.hist( np.log10(Mlen_all_central), bins=50, color='grey',  label=r'Central galaxies')
	plt.hist( np.log10(Mlen_all_sat), bins=50, color='#225ea8',  label=r'Satellite galaxies')

	plt.axvline(2, 0, label='Len=100', color='red')
	ax.set_xlabel(r'log Len', fontsize=25)
	#ax.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)
	plt.legend()
	plt.show()
	plt.savefig('./plots/LenMax_plot.png')



def single_central_galaxies(groups, G, Mass_cutoff = 0.06424,Group_of_one = 1):

	"""
	Create indices of a single galaxies. 

	Parameters:
	==========
	groups - Dictionary.
	G - Data frame.
	Mass_cutoff - Float. 
	Group_of_one - Integer = 1 since I want only single galaxies.

	Return:
	======
	single_gal_ind - List of indices with single galaxies in a halo.

	"""

	single_gal_ind = []

	#initiate condition for group
	for group in groups[Group_of_one]:
		if (G["StellarMass"][group] < Mass_cutoff).any(): #creates array of booleans; if theese are true hit continue
			continue
	    
	    #Store indices of single galaxies 
		single_gal_idx = group
		single_gal_ind.append(single_gal_idx)
	
	return single_gal_ind


def plot_single_galaxies(G, single_gal_ind):

	"""
	Plot MHI vs Mstellar for single galaxies. 
		
	Parameters:
	==========
	G - Data frame.
	single_gal_ind - List of integers. Indices assigned to the single galaxies.

	Return:
	======
	Save figure.	
	"""

	Mstellar_single_gal = G["StellarMass"][single_gal_ind]*1e10/h
	Mcoldgas_single_gal = np.sum(G['DiscHI'],axis=1)[single_gal_ind]*1e10/h
	Mvir_single_gal = G["Mvir"][single_gal_ind]*1e10/h
	BTT_single = (G['InstabilityBulgeMass'][single_gal_ind]*1e10/h + G['MergerBulgeMass'][single_gal_ind]*1e10/h) / ( G['StellarMass'][single_gal_ind]*1e10/h )


	fig = plt.figure(figsize=(10,10))                                                               
	ax = fig.add_subplot(1,1,1)


	plt.plot(np.log10(Mstellar_single_gal), np.log10(Mcoldgas_single_gal), 'o', 
		 color='lightgrey', markeredgecolor='k', markersize=8, markeredgewidth=0.2, label='Single galaxies')


	ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
	ax.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)
	plt.legend(loc=4)
	plt.show()
	plt.savefig('./plots/Single_galaxies_MHIvsMst.png')

def plot_group_numbers_and_sizes(updated_dict):

	"""
	Plot group statistics: Number of groups vs N_sized groups.
	
	Parameters:
	==========
	updated_dict: Nested dictionary. Keyed by the `groups`.
                     Contains Centrals and Satellites which are in groups where galaxies are above the my_mass_cutoff
   	
	Returns:
	=======
	Saves plot of the group statistics.

	"""

	#How many groups are present and what is their length?
	Group_statistics = []
	Group_len = np.arange(100)
	
	for i in trange(0,100):
		try:
		#print(i, len(updated_dict[i]["Groups"].keys()))
			Group_statistics.append(len(updated_dict[i]["Groups"].keys()))
		except KeyError:
			Group_statistics.append(0)
			continue
	       		#print(i, 'No such group length')

	df_groups = pd.DataFrame({'GroupLength'            : Group_len,
				   'NumberNlengthGroups'    : Group_statistics })

	#print(df_groups[1:])
	fig = plt.figure(figsize=(10,10))                                                               
	ax = fig.add_subplot(1,1,1)
	plt.plot(df_groups['GroupLength'][2:], df_groups['NumberNlengthGroups'][2:], marker='o', label='Groups in DarkSage');
	ax.set_yscale('log')
	plt.axhline(1, 0, label='1 Group', color='k')
	plt.xlabel('Group Size')
	plt.ylabel('Number of Groups')
	plt.legend(loc=1)
	plt.show()
	plt.savefig('./plots/Group_statistics.png')


# Initiate figure and how large it will be
@jit(cache=True)
def plot_mhi_vs_ms_3x3(groups_dict):
    
    """
    Make 3x3 plot (HI mass vs Stellar mass) with galaxy groups based on the number of galaxies in a group and separate in each group central
    and satellite galaxy.
    
    Parameters
    ==========
    groups_dict: Dictionary. Keyed by the group size. Use: ``groups_dict[group_size]["Centrals"]``
                 Extracts groups with size N and the satellite & central galaxies
                 Created dictionary from (def create_cen_sat_from_groups_dict)
  
    Returns
    =======
    3x3 Figure with M_HI versus M_*
    
    Usage
    =====
    To use type: ``mhi_vs_ms_3x3(updated_dict) ``
                 Where updated_dict is the dictionary that is parsed to plotting function
                 ``updated_dict = create_cen_sat_from_groups_dict(groups, store_cen_indices)``
    
    """
    
    
    fig, ax = plt.subplots(nrows=4, ncols=5, sharex='col', sharey='row', figsize=(18, 15))
    
    #ax[0][0].plot(np.log10(Mstellar_single_gal), np.log10(Mcoldgas_single_gal), 'o', color='lightgrey',markersize=8, label='Single galaxies')
    
    # Put plot in row and columns:3by3 are 00, 01, 02, 10, 11, 12, 20, 21, 22 
    row_count = 0
    for size in trange(1, 20, 1): #I'm planning to have 1-9 sized groups. Made to form trange(1, 9, 1) as 3x3 pannels 
            
        if size % 5 == 0:
            row_count += 1
        this_ax = ax[row_count, size%5] #Axis. Created for plotting 3x3 plots
        
        group_size = size+1 # Select number of galaxies per group --- adding +1 because it is counting from 0.
    
        central_gals = groups_dict[group_size]["Centrals"]  #List of integers. Contains indices of central galaxies.  List is obtained through the dictionary. 
        sat_gals = groups_dict[group_size]["Satellites"] #List of integers. Contains indices of satellite galaxies. List is obtained through the dictionary.
    
        # Plot single galaxies           
        #this_ax.plot(np.log10(Mstellar_single_gal), np.log10(Mcoldgas_single_gal), 'o', color='lightgrey',markersize=8)
        
            
        # Do plotting of groups of N-length; it will be placed on each subplot + +.
        # Centrals
        this_ax.plot(np.log10(G['StellarMass'][central_gals]*1e10/h), 
                         np.log10(np.sum(G['DiscHI'],axis=1)[central_gals]*1e10/h), 
                         'o', color='white', markeredgecolor='#1f78b4', markersize=6, markeredgewidth=2)
        # Satellites
        this_ax.plot(np.log10(G['StellarMass'][sat_gals]*1e10/h), 
                         np.log10(np.sum(G['DiscHI'],axis=1)[sat_gals]*1e10/h), 
                         'o', color='#1f78b4', markersize=6)
    
        # Add label for each sub-plot to know the group size and what is central/satellite
        this_ax.plot([], [], 'o', color='white', label='Groups of %.f' %group_size)
        this_ax.plot([], [], 'o', color='lightgrey', label='Single galaxies')
        this_ax.plot([], [], 'o', color='#1f78b4', markersize=8, label='Satellite')
        this_ax.plot([], [], 'o', color='white', markeredgecolor='#1f78b4', markersize=8, markeredgewidth=2, label='Central')
        #Add legend    
        leg = this_ax.legend(loc=4,frameon=True,fancybox=True, framealpha=0.6, fontsize=16)
        
    # Add x and y axis labels only on the plot edges since there will be no space between panels
    for row in range(4):
        this_ax = ax[row,0]
        this_ax.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)
    for col in range(5):
        this_ax = ax[3,col]
        this_ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
        this_ax.xaxis.set_ticks(np.arange(9,11.9, 1))
    # Tight layout and remove spacing between subplots
    plt.tight_layout()
    # My first plot is only single galaxies so I add legend separately
    ax[0][0].legend(loc=4,frameon=True,fancybox=True, framealpha=0.6, fontsize=16)

    plt.subplots_adjust(wspace = 0.0, hspace = 0.0)
    plt.show()
    #return fig
    plt.savefig('./plots/NxN_Mhi_vs_Mst.png')


def two_sided_histogram_group_and_single(updated_dict, group_size):
    """
    Make a scatter plot with 3 datasets and then place histogram on both x and y axis for them. 
    Here I use logMhi vesus logMstar -- other data will need readjustment of the min/max.
    
    Parameters
    ==========
    x_cen, y_cen: x and y values of the first dataset. Floats. Can be list/array. 
                  In this case, I am using logMHI and logMstar of central galaxies.
    
    x_sat, y_sat: x and y values of the second dataset. Floats. Can be list/array. 
                  In this case, I am using logMHI and logMstar of satellite galaxies.
                  
    x_single, y_single: x and y values of the third dataset. Floats. Can be list/array. 
                        In this case, I am using logMHI and logMstar of single (central) galaxies.
                        
        
    Returns
    =======
    A scatter plot with histogram on x and y axis. 
    
    Usage
    =====
    To use type: ``two_sided_histogram_group_and_single(x_cen, y_cen, x_sat, y_sat, x_single, y_single)``
               I extracted single, satellite and central gal from dictionary using ``updated_dict`` and 
               group_size=1 for single galaxies; group_size=2 for pairs -- can be any group_size number
               
                single = updated_dict[1]['Centrals']
                central_2 = updated_dict[2]["Centrals"]
                satellite_2 = updated_dict[2]["Satellites"]
                
                x_cen = np.log10(G["StellarMass"][central_2]*1e10/h ) #gives array of central galaxy stellar masses 
                y_cen = np.log10(G["ColdGas"][central_2]*1e10/h+1) 
                
                etc. just update central_2 with other keys to specific indices.
                
    """

    nullfmt = NullFormatter()         # no labels
    single = updated_dict[1]['Centrals']
    central_X = updated_dict[group_size]["Centrals"]
    satellite_X = updated_dict[group_size]["Satellites"]

    x_cen = np.log10(G["StellarMass"][central_X]*1e10/h ) #gives array of central galaxy stellar masses 
    y_cen = np.log10(np.sum(G['DiscHI'],axis=1)[central_X]*1e10/h+1) #BECAUSE log0 goes to -inf so I add 1! has to be checked/removed!
    x_sat = np.log10(G["StellarMass"][satellite_X]*1e10/h) #gives array of satellite galaxy stellar masses 
    y_sat = np.log10(np.sum(G['DiscHI'],axis=1)[satellite_X]*1e10/h+1)

    x_single = np.log10(G["StellarMass"][single]*1e10/h) #gives array of satellite galaxy stellar masses 
    y_single = np.log10(np.sum(G['DiscHI'],axis=1)[single]*1e10/h+1)
	  
    x = x_single
 
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # start with a rectangular Figure
    fig = plt.figure(figsize=(10,10)) 
    #plt.figure( figsize=(8, 8))
    
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    # the scatter plot:
    axScatter.scatter(x_single, y_single, color='#fcc5c0', label='Single galaxies', alpha=0.2)
    axScatter.scatter(x_cen, y_cen, color='white', edgecolor='#1f78b4', linewidth=1, label='Centrals')
    axScatter.scatter(x_sat, y_sat, color='#1f78b4', label='Satellites', alpha=0.8)
    
    axScatter.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
    axScatter.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)

    # now determine nice limits by hand:
    binwidth = 0.1
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(x))])
    lim = (int(xymax/binwidth) + 1) * binwidth
    
    axScatter.set_xlim((8.5, 11.7))
    axScatter.set_ylim((0, 11.7))
    
    bins = np.arange(5.8, lim + binwidth, binwidth)
    #density=True -- the result is the value of the probability density function at the bin,
    #                normalized such that the integral over the range is 1 (buggy with uneqyal bin widths)
    axHistx.hist(x_sat, bins=bins, density=True, color='#1f78b4', alpha=0.6, edgecolor='k', linewidth=1)
    axHisty.hist(y_sat, bins=bins, density=True, orientation='horizontal', alpha=0.6, color='#1f78b4', edgecolor='k', linewidth=1.2)
    
    axHistx.hist(x_cen, bins=bins, density=True, color='white',alpha=0.4, edgecolor='#1f78b4', linewidth=1)
    axHisty.hist(y_cen, bins=bins, density=True, orientation='horizontal', color='white', alpha=0.4, edgecolor='#1f78b4', linewidth=2)
    
    axHistx.hist(x_single, bins=bins, density=True, color='#fcc5c0',alpha=0.4, edgecolor='black', linewidth=1)
    axHisty.hist(y_single, bins=bins, density=True, orientation='horizontal', color='#fcc5c0', alpha=0.4, edgecolor='black', linewidth=1)
 

    axHistx.set_ylabel(r'N$_{\textrm{gal}}$', fontsize=22)
    axHisty.set_xlabel(r'N$_{\textrm{gal}}$', fontsize=22)
    
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    axScatter.legend(loc=3)

    #for i in group_size:
    fname = f'./plots/Two_sided_Groups_and_single{group_size}.png'
    plt.savefig(fname)
    print(f"Saved plot to {fname}")
    return #plt.show(), x_cen, x_sat, y_cen, y_sat

def two_sided_histogram_groups(updated_dict, group_size, color='#1f78b4'):
    """

    Make a scatter plot with 2 datasets and then place histogram on both x and y axis for them. 
    Here I use logMhi vesus logMstar -- other data will need readjustment of the min/max.
    
    Parameters
    ==========
    x_cen, y_cen: x and y values of the first dataset. Floats. Can be list/array. 
                  In this case, I am using logMHI and logMstar of central galaxies.
    
    x_sat, y_sat: x and y values of the second dataset. Floats. Can be list/array. 
                  In this case, I am using logMHI and logMstar of satellite galaxies.                        
        
    color : Boolean. Chose color in case you want differen one. If not specified, will be the default one for satellites.
            Use different color in case there is only one dataset, for exaple:
            two_sided_histogram_groups(x, y, [], [], 'grey')
    
    
    Returns
    =======
    A scatter plot with histogram on x and y axis for both datasets. 
    
    Usage
    =====
    To use type: ``two_sided_histogram_groups(x_cen, y_cen, x_sat, y_sat)``
               I extracted satellite and central gal from dictionary using ``updated_dict`` and 
               group_size=2 for pairs -- can be any group_size number
               
                central_2 = updated_dict[2]["Centrals"]
                satellite_2 = updated_dict[2]["Satellites"]
                
                x_cen = np.log10(G["StellarMass"][central_2]*1e10/h ) #gives array of central galaxy stellar masses 
                y_cen = np.log10(G["ColdGas"][central_2]*1e10/h+1) 
                
                etc. just update central_2 with other keys to specific indices.
                
    """

    
    nullfmt = NullFormatter()         # no labels
   
    #group_size = group_size_for_two_sided 
    single = updated_dict[1]['Centrals']
    central_X = updated_dict[group_size]["Centrals"]
    satellite_X = updated_dict[group_size]["Satellites"]

    x_cen = np.log10(G["StellarMass"][central_X]*1e10/h ) #gives array of central galaxy stellar masses 
    y_cen = np.log10(np.sum(G['DiscHI'],axis=1)[central_X]*1e10/h+1) #BECAUSE log0 goes to -inf so I add 1! has to be checked/removed!
    x_sat = np.log10(G["StellarMass"][satellite_X]*1e10/h) #gives array of satellite galaxy stellar masses 
    y_sat = np.log10(np.sum(G['DiscHI'],axis=1)[satellite_X]*1e10/h+1)


    x =  x_cen
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # start with a rectangular Figure
    fig = plt.figure(figsize=(10,10)) 
    #plt.figure(1, figsize=(8, 8))
    
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    # the scatter plot:
    axScatter.scatter(x_cen, y_cen, color='white', edgecolor='#1f78b4', linewidth=1, label='Centrals')
    axScatter.scatter(x_sat, y_sat, color=color, label='Satellites', alpha=0.8)
    
    
    axScatter.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
    axScatter.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)

    # now determine nice limits by hand:
    binwidth = 0.1
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(x))])
    lim = (int(xymax/binwidth) + 1) * binwidth
    
    axScatter.set_xlim((8.5, 11.7))
    axScatter.set_ylim((0, 11.7))
    
    bins = np.arange(5.8, lim + binwidth, binwidth)
    axHistx.hist(x_sat, bins=bins, color=color, alpha=0.6, edgecolor='k', linewidth=1)
    axHisty.hist(y_sat, bins=bins, orientation='horizontal', alpha=0.6, color=color, edgecolor='k', linewidth=1.2)
    
    axHistx.hist(x_cen, bins=bins, color='white',alpha=0.4, edgecolor='#1f78b4', linewidth=1)
    axHisty.hist(y_cen, bins=bins, orientation='horizontal', color='white', alpha=0.4, edgecolor='#1f78b4', linewidth=2) 

    axHistx.set_ylabel(r'N$_{\textrm{gal}}$', fontsize=22)
    axHisty.set_xlabel(r'N$_{\textrm{gal}}$', fontsize=22)
    
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    axScatter.legend(loc=3)


    # Save the output
    fname = f'./plots/Two_sided_Groups{group_size}.png'
    plt.savefig(fname)
    print(f"Saved plot to {fname}")
   
    return #plt.show()


def hist_Mhi_vs_Mstar_each_group(groups_dict):

    """
    N times the scatter plot with thee datasets and their histograms on x and y axis. One dataset is always single galaxies
    the other two datasets are groups of lenght N++
    Here I use logMhi vesus logMstar -- other data will need readjustment of the min/max.
	    
	    
    Parameters
    ==========
    groups_dict: Dictionary. Keyed by the group size. Use: ``groups_dict[group_size]["Centrals"]``
			 Extracts groups with size N and the satellite & central galaxies
			 Created dictionary from (def create_cen_sat_from_groups_dict)
	  
    Returns
    =======
    N times the scatter plot with two datasets and their histograms on x and y axis.
        
	    
    Usage
    =====
    To use type: ``hist_Mhi_vs_Mstar_with_singles(updated_dict) ``
    Where updated_dict is the dictionary that is parsed to plotting function
    ``updated_dict = create_cen_sat_from_groups_dict(groups, store_cen_indices)``    
    """
		
    row_count = 0
    for size in trange(1, 9, 1): #I'm planning to have 1-9 sized groups
            
        group_size = size+1 # Select number of galaxies per group --- adding +1 because it is counting from 0.
    
        central_gals = groups_dict[group_size]["Centrals"]
        sat_gals = groups_dict[group_size]["Satellites"]
        single_gals = groups_dict[1]['Centrals']
     
        # Do plotting of groups of N-length; it will be placed on each figure
        # Centrals and Satellites; x & y axis histogram
        # Using function two_sided_histogram to plot the histogram output so here I call the function and give (x, y), (x, y)
        
        two_sided_histogram_group_and_single(updated_dict, group_size)
        
        # Optional! Same as above, just without signle galaxies in the background:
        #two_sided_histogram_groups(updated_dict, group_size)


@jit
def find_richer_central_for_Nsized_group(grp_length):

    """
    Find groups where centrals are the HI richest (central has the most HI than either of its satellites) and HI poorest and plot them.

    Parameters:
    ==========
    grp_length: Integer. Choose the group size for which to analyse this property.

    Return:
    ======
    richer_central_ind, poorer_central_ind, richer_sat_ind, poorer_sat_ind -- indices of the centrals and satellites where central is HI richest/poorest

    """

    richer_central_ind = []
    poorer_central_ind = []

    richer_sat_ind = []
    poorer_sat_ind = []

    for group_key in tqdm(updated_dict[grp_length]["Groups"].keys()):
        central_idx = updated_dict[grp_length]["Groups"][group_key]["Centrals"] #give indices
        central_mass = np.sum(G['DiscHI'],axis=1)[central_idx] #give masses of these galaxies
        
        satellite_inds = updated_dict[grp_length]["Groups"][group_key]["Satellites"]
        sat_mass = np.sum(G['DiscHI'],axis=1)[satellite_inds]
        
        if (central_mass > sat_mass).all(): #check conditions
            print("Central is the most HI massive")
            richer_central_ind.append(central_idx)
            richer_sat_ind.append(satellite_inds)
        else:
            print("Satellite is more HI massive thatn central")
            poorer_central_ind.append(central_idx)
            poorer_sat_ind.append(satellite_inds)
        #print(sat_mass > central_mass)
        
        print("Central mass is {0} and sat masses are {1}".format(central_mass, sat_mass))
        print("")

    #Masses, obtained from the indices: 
    richer_central_hi_m = np.log10( (np.sum(G['DiscHI'],axis=1)[richer_central_ind] *1e10/h )+1)
    poorer_central_hi_m = np.log10( (np.sum(G['DiscHI'],axis=1)[poorer_central_ind] *1e10/h )+1)

    richer_sat_hi_m = np.log10( (np.sum(G['DiscHI'],axis=1)[richer_sat_ind] *1e10/h )+1)
    poorer_sat_hi_m = np.log10( (np.sum(G['DiscHI'],axis=1)[poorer_sat_ind] *1e10/h )+1)

    richer_central_s_m = np.log10( G['StellarMass'][richer_central_ind] *1e10/h +1)
    poorer_central_s_m = np.log10( G['StellarMass'][poorer_central_ind] *1e10/h +1)

    richer_sat_s_m = np.log10( G['StellarMass'][richer_sat_ind] *1e10/h +1)
    poorer_sat_s_m = np.log10( G['StellarMass'][poorer_sat_ind] *1e10/h +1)


    b = richer_sat_hi_m.ravel() #RESHAPE 2D array into 1D


    # Plot the figure
    fig = plt.figure(figsize=(10,10))                                                               
    ax = fig.add_subplot(1,1,1)

    binwidth=0.1
    bins = np.arange(7.5, 11 + binwidth, binwidth)
        
    plt.hist(richer_central_hi_m, bins=bins, color='#2c7fb8', edgecolor='k', linewidth=1, alpha=0.8, label='HI rich(er) central')
    plt.hist(poorer_central_hi_m, bins=bins, color='#c7e9b4', edgecolor='k', linewidth=1, alpha=0.5, label='HI poor(er) central')

    ax.set_ylabel(r'N$_{\mathrm{galaxies}}$', fontsize=25)
    ax.set_xlabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)

    plt.xlim(7,11)
    plt.legend(loc=2)
    plt.show()
    plt.savefig("./plots/HI_richer_and_poorer_centrals.png")

       
    return richer_central_ind, poorer_central_ind, richer_sat_ind, poorer_sat_ind, richer_central_s_m, richer_central_hi_m, poorer_central_s_m, poorer_central_hi_m, richer_sat_hi_m, poorer_sat_hi_m, richer_sat_s_m, poorer_sat_s_m



def two_sided_histogram_rich(x_cen_rich, y_cen_rich, x_cen_poor, y_cen_poor):
    """
    Make a scatter plot with 2 datasets and then place histogram on both x and y axis for them. 
    Here I use logMhi vesus logMstar -- other data will need readjustment of the min/max.
    
    Parameters
    ==========
    x_cen_rich, y_cen_rich: x and y values of the first dataset. Floats. Can be list/array. 
                  In this case, I am using logMHI and logMstar of central galaxies.
    
    x_cen_poor, y_cen_poor: x and y values of the second dataset. Floats. Can be list/array. 
                  In this case, I am using logMHI and logMstar of satellite galaxies.                        
        
    color : Boolean. Chose color in case you want differen one. If not specified, will be the default one for satellites.
            Use different color in case there is only one dataset, for exaple:
            two_sided_histogram_groups(x, y, [], [], 'grey')
    
    
    Returns
    =======
    A scatter plot with histogram on x and y axis for both datasets. 
    
    Usage
    =====
    To use type: ``two_sided_histogram_groups(x_cen, y_cen, x_sat, y_sat)``
               I extracted satellite and central gal from dictionary using ``updated_dict`` and 
               group_size=2 for pairs -- can be any group_size number
               
               I extracted satellite and central gal from dictionary using ``updated_dict`` and 
               group_size=2 for pairs -- can be any group_size number
                
               Indices:
               central_idx = updated_dict[3]["Groups"][group_key]["Centrals"]
               satellite_inds = updated_dict[3]["Groups"][group_key]["Satellites"]
                
               To this, condition is added which check the mass of the central vs. satellites.
    """

    
    nullfmt = NullFormatter()         # no labels
    
    x = x_cen_rich
        
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    ## start with a rectangular Figure
    
    fig = plt.figure(figsize=(10,10)) 
    #
    #plt.figure(1, figsize=(12, 12))
    
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    # the scatter plot:
    axScatter.scatter(x_cen_rich, y_cen_rich, color='#2c7fb8', label='HI rich(er) central')
    axScatter.scatter(x_cen_poor, y_cen_poor, color='#c7e9b4', label='HI poor(er) central', alpha=0.8)
    
    axScatter.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
    axScatter.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)

    # now determine nice limits by hand:
    binwidth = 0.1
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(x))])
    lim = (int(xymax/binwidth) + 1) * binwidth
    
    axScatter.set_xlim((8.5, 11.7))
    axScatter.set_ylim((5.8, 11.7))
    
    bins = np.arange(5.8, lim + binwidth, binwidth)
    axHistx.hist(x_cen_rich, bins=bins, color='#2c7fb8', alpha=0.6, edgecolor='k', linewidth=1)
    axHisty.hist(y_cen_rich, bins=bins, orientation='horizontal', alpha=0.6, color='#2c7fb8', edgecolor='k', linewidth=1)
    
    axHistx.hist(x_cen_poor, bins=bins, color='#c7e9b4',alpha=0.4, edgecolor='k', linewidth=1)
    axHisty.hist(y_cen_poor, bins=bins, orientation='horizontal', color='#c7e9b4', alpha=0.4, edgecolor='k', linewidth=1) 

    axHistx.set_ylabel(r'N$_{\textrm{gal}}$', fontsize=22)
    axHisty.set_xlabel(r'N$_{\textrm{gal}}$', fontsize=22)
    
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    axScatter.legend(loc=3)

    plt.savefig("./plots/Two_sided_HI_richer_and_poorer_centrals.png")


    return 
    
@jit
def two_sided_histogram_rich_groups(grp_length, x_cen_rich, y_cen_rich, x_cen_poor, y_cen_poor,
                                   x_sat_rich, y_sat_rich, x_sat_poor, y_sat_poor):
    """
    Make a scatter plot with 8 datasets and then place histogram on both x and y axis for them as rich/poor central groups. 
    Here I use logMhi vesus logMstar -- other data will need readjustment of the min/max.
    
    Parameters
    ==========
    grp_lenght: Integer. Number of galaxies within a group.

    x_cen_rich, y_cen_rich: x and y values of the first dataset - here are centrals which are HI richer than their
                            satellites. Floats. Should be list/array; use .ravel() if needed
                            to reduce the dimensionality.  
                            In this case, I am using logMHI and logMstar of central galaxies.
    
    x_cen_poor, y_cen_poor: x and y values of the second dataset - here are centrals which are poorer than their
                            satellites. Should be list/array; use .ravel() if needed to reduce the dimensionality.
                            In this case, I am using logMHI and logMstar of satellite galaxies.                        
        
    x_sat_rich, y_sat_rich, x_sat_poor, y_sat_poor:
                            Same as above; just uses satellites around their respective central galaxy.
    
    Returns
    =======
    A scatter plot with histogram on x and y axis for both datasets. 
    
    Usage
    =====
    To use type: ``two_sided_histogram_rich_groups(richer_central_s_m.ravel(), richer_central_hi_m.ravel(), poorer_central_s_m.ravel(), poorer_central_hi_m.ravel(),
                                richer_sat_s_m.ravel(), richer_sat_hi_m.ravel(), poorer_sat_s_m.ravel(), poorer_sat_hi_m.ravel()) ``
               I extracted satellite and central gal from dictionary using ``updated_dict`` and 
               group_size=2 for pairs -- can be any group_size number
                Indices:
                central_idx = updated_dict[3]["Groups"][group_key]["Centrals"]
                satellite_inds = updated_dict[3]["Groups"][group_key]["Satellites"]
                
                To this, condition is added which check the mass of the central vs. satellites.
                
                
    """

    
    nullfmt = NullFormatter()         # no labels

    x = x_cen_rich    

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # start with a rectangular Figure
    fig = plt.figure(figsize=(10,10)) 
    #plt.figure(1, figsize=(12, 12))
    
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    # the scatter plot:
    axScatter.scatter(x_cen_rich, y_cen_rich, color='white', edgecolor='#2c7fb8', s=80, linewidth=2, label='HI rich(er) central')
    axScatter.scatter(x_sat_rich, y_sat_rich, color='#2c7fb8', edgecolor='#2c7fb8', s=80, linewidth=2, label='HI rich(er) central\'s satellite')

    axScatter.scatter(x_cen_poor, y_cen_poor, color='white', edgecolor='lightgrey', s=80, linewidth=2, label='HI poor(er) central', alpha=0.8)
    axScatter.scatter(x_sat_poor, y_sat_poor, color='lightgrey', edgecolor='lightgrey', s=80, linewidth=2, label='HI poor(er) central\'s satellite', alpha=0.8)
    
    
    axScatter.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
    axScatter.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)

    # now determine nice limits by hand:
    binwidth = 0.1
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(x))])
    lim = (int(xymax/binwidth) + 1) * binwidth
    
    axScatter.set_xlim((8.5, 11.7))
    axScatter.set_ylim((5.8, 11.7))
    
    bins = np.arange(5.8, lim + binwidth, binwidth)
   
    x_rich = sorted(list(set(x_cen_rich)) + list(set(x_sat_rich)))
    y_rich = sorted(list(set(y_cen_rich)) + list(set(y_sat_rich)))
    
    x_poor = sorted(list(set(x_cen_poor)) + list(set(x_sat_poor)))
    y_poor = sorted(list(set(y_cen_poor)) + list(set(y_sat_poor)))
    
    axHistx.hist(x_rich, bins=bins, color='#2c7fb8', alpha=0.6, edgecolor='k', linewidth=1)
    axHisty.hist(y_rich, bins=bins, orientation='horizontal', alpha=0.6, color='#2c7fb8', edgecolor='k', linewidth=1)
    
    axHistx.hist(x_poor, bins=bins, color='lightgrey',alpha=0.4, edgecolor='k', linewidth=1)
    axHisty.hist(y_poor, bins=bins, orientation='horizontal', color='lightgrey', alpha=0.4, edgecolor='k', linewidth=1) 

    axHistx.set_ylabel(r'N$_{\textrm{gal}}$', fontsize=22)
    axHisty.set_xlabel(r'N$_{\textrm{gal}}$', fontsize=22)
    
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    leg = axScatter.legend(loc=3, fontsize=13)

    leg.set_title('Group size = {0}'.format(grp_length)) 
    plt.savefig("./plots/Two_sided_rich_groups.png")


    return    

def group_indices_for_percentage(updated_dict):
    """
    Extract group based indices of centrals, satellites and all (from the same halo/group).

    Parameters:
    ==========
    updated_dict: Nested dictionary. Keyed by the `groups`.
                  Contains Centrals and Satellites which are in groups where galaxies are above the my_mass_cutoff
   	
    Returns:
    ========
    c_ind: List of integers. Central galaxies.
    s_ind: List of integers. Satellite galaxies.
    g_ind: List of integers. Halo galaxies. Needed for computation of the group properties.

    """

    # STORE INDICES

    c_ind = []
    s_ind = []
    g_ind = []

    for i in trange(3,21,1): #starting grom galaxy pairs
        for group_key in updated_dict[i]["Groups"].keys():
            central_idx = updated_dict[i]["Groups"][group_key]["Centrals"] #give indices
            c_ind.append(central_idx)
                   
            all_idx = updated_dict[i]["Groups"][group_key]["Centrals"]+updated_dict[i]["Groups"][group_key]["Satellites"]
            g_ind.append(all_idx)
            
            satellite_inds = updated_dict[i]["Groups"][group_key]["Satellites"]
            s_ind.append(satellite_inds)

    return c_ind, s_ind, g_ind


@jit
def compute_group_properties(c_ind, s_ind, g_ind):
    """
    Compute properties of the groups. Used for calculation of the % of the HI in the central galaxy.

    Parameters:
    ==========
    c_ind: List of integers. Central galaxies.
    s_ind: List of integers. Satellite galaxies.
    g_ind: List of integers. Contain galaxies within the group.

    Returns:
    =======
    g_m: Floats. Group HI mass.
    g_st: Floats. Group stellar mass.
    percentage: Floats. % of HI in central galaxy.
    BTT_cen: Floats. Bulge to total ratio for central galaxy.
    Mvir_cen: Floats. Virial mass of central galaxy. 
    Rvir_cen: Floats. Virial raius of central galaxy.

    """

    # Compute group properties 
    g_m = []
    g_st = []
    group_length = []

    for i in tqdm(g_ind):
        g_mass = np.sum(G['DiscHI'],axis=1)[i]*1e10/h
        g_st_mass = G['StellarMass'][i]*1e10/h
        g_m.append(np.sum(g_mass))
        g_st.append(np.sum(g_st_mass))
        
        # Extract group length
        size = len(i)
        group_length.append(size)
    
    # Compute central galaxies
    central_mass = np.sum(G['DiscHI'],axis=1)[c_ind]*1e10/h
    central_st_mass = G['StellarMass'][c_ind]*1e10/h
    
    #Compute bulge to total ratio of the central galaxy
    BTT_cen = (G['InstabilityBulgeMass'][c_ind] + G['MergerBulgeMass'][c_ind]) / ( G['StellarMass'][c_ind] )
    
    # Virial mass and radius
    Mvir_cen = np.log10( (G['Mvir'][c_ind])*1e10/h)
    Rvir_cen = G['Rvir'][c_ind]

    # Compute percentage
    percentage = (central_mass.ravel()/g_m)*100
    print('Computed percentage of HI in central')

    return g_m, g_st, percentage, BTT_cen, Mvir_cen, Rvir_cen, group_length


def plot_per_cent_of_HI_in_central(g_m, g_st, percentage):

    """
    Make a plot with group paramteres (stellar mass vs HI mass) and a colormap is % of the HI in central with respect to group HI mass.
       
    Parameters:
    ===========
    g_m: List of floats. Group HI mass.
    g_st: List of floats. Group stellar mass.
    percentage: List of floats. Percent of HI in central galaxy with respect to entire group HI content.

    Returns:
    =======
    Saves plot.

    """


    fig = plt.figure(figsize=(13,10))                                                               
    ax = fig.add_subplot(1,1,1)

    cm = plt.cm.get_cmap('YlGnBu')


    im = plt.scatter(np.log10(g_st), np.log10(g_m), s=100,
                            c=percentage, edgecolor='k', cmap=cm, label=r'Groups ($N\geq3$) properties')

    fig.colorbar(im, ax=ax, orientation='vertical', label=r'\% of the M$_{\mathrm{HI}}$ in central',
                pad=0.01)

    ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
    ax.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)
    ax.legend(loc=3)
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('k')

    plt.savefig('./plots/Percent_HIinCentral.png')
    return 

def make_dataframe_for_pairplot(g_st, g_m, percentage, group_length, BTT_cen, Mvir_cen, Rvir_cen):
    """
    Create a pandas dataframe.

    Parameters:
    ==========
    g_m: List of floats. Group HI mass.
    g_st: List of floats. Group stellar mass.
    percentage: List of floats. Percent of HI in central galaxy with respect to entire group HI content.
    BTT_cen: Floats. Bulge to total ratio for central galaxy.
    Mvir_cen: Floats. Virial mass of central galaxy. 
    Rvir_cen: Floats. Virial raius of central galaxy.
    group_length: Integers. Number of galaxies in a group.

    Returns:
    =======
    df_pair: Pandas dataframe.    
    """

    # Define what goes into dataframe
    df_pair = pd.DataFrame({'GroupStellarMass'   : np.log10(g_st),
                            'GroupHIMass'        : np.log10(g_m),
                            'Percent'            : percentage,
                            'GroupSize'          : group_length,
                            'BTTcentral'         : BTT_cen.ravel() ,
                            'Mvircen'            : Mvir_cen.ravel(),
                            'Rvircen'            : Rvir_cen.ravel() })
   
    return df_pair

def make_dataframe_for_joyplot(dataframe):
    """
    Make a dataframe with the RoundedPercentage - which is used only fot the joyplots.

    Parameters:
    dataframe = Dataframe, as obtained from the pairplot.

    Returns:
    df_joy - Dataframe. Used for making Joy plots with RoundedPercent as y-axis.

    """

    def roundup(x):
        """
        Round up number as ceil of 10. Used for dounded percentage values needed for joyplot.
        """
        return int(math.ceil(x / 10.0)) * 10 # Add whether it will be by increment of 5 or 10 or something third

    rounded_per = []
    for i in dataframe['Percent']:
        A = roundup(i)#_test = (central_mass.ravel()/g_m)*100
        rounded_per.append(A) 


    df_joy = pd.DataFrame({'GroupStellarMass'   : np.log10(g_st),
                            'GroupHIMass'        : np.log10(g_m),
                            'Percent'            : percentage,
                            'GroupSize'          : group_length,
                            'BTTcentral'         : BTT_cen.ravel() ,
                            'Mvircen'            : Mvir_cen.ravel(),
                            'Rvircen'            : Rvir_cen.ravel(),
                            'RoundedPercent'     : rounded_per})
                       
    return df_joy

@jit
def make_pair_plot(dataframe):
    """
    Make a pair plot.

    Parameters:
    ===========
    dataframe: Pandas dataframe. Should contin all the properties that I want for pairplot.
    
    Returns:
    ========
    Saves Plot.

    """
    fig = plt.figure(figsize=(13,13)) 
    ax = fig.add_subplot(1,1,1)
    

    sns.set(font_scale=1)
    sns.set_style("ticks", {"xtick.major.size":10, "ytick.major.size":10,
                            "xtick.minor.size":6,"ytick.minor.size":6})
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    
    
    g = sns.PairGrid(dataframe, diag_sharey=False)
    g.map_lower(sns.kdeplot)
    g.map_upper(sns.scatterplot)
    g.map_diag(sns.kdeplot, lw=3)

    # To hide upper scatter plot axies: axis
    #for i, j in zip(*np.triu_indices_from(g.axes, 1)):
    #    g.axes[i, j].set_visible(False)
    plt.savefig("./plots/Pair_plot_groups.png")

    #################################################### 
    print('Made pair plot, now making pair correlation plot')
    ####################################################
    
    # Make pair correlation plot
    fig2 = plt.figure(figsize=(8,8))                                                               
    ax = fig.add_subplot(1,1,1)

    corr = dataframe.corr()

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(corr, annot=True, cmap='viridis',  linewidths=.5, mask=mask, vmax=1,
            xticklabels=corr.columns,
            yticklabels=corr.columns)

    plt.savefig("./plots/Pair_correlation_plot_groups.png")

    return dataframe


def make_joy_plot(dataframe, group_size_joy, how_many_groups, colormap_one, colormap_two):
    """
    Make a joy plot for specific group, using two columns (two datasets) and sort them by third one.

    Paremeters:
    ==========
    dataframe: Pandas dataframe.
    group_size_joy: List of Integers. Specific length of the group.   
    how_many_groups: Integer. Number of different groups to plot.
    column_one: String. Column from the dataframe. One property.
    column_two: String. Column from the dataframe. One property.
    color_map_one: Name of the colormap. Example: `plt.cm.viridis`
    color_map_two: Name of the colormap.    

    Returns:
    Saves plot.

    """
    if how_many_groups == 1:
        
        print('One group size is selected')
        # Draw Plot
        plt.figure(figsize=(16,10), dpi= 80)
        
        fig, axes = joypy.joyplot(dataframe[dataframe['GroupSize'] == group_size_joy], column = ['GroupHIMass', 'GroupStellarMass'], by = 'RoundedPercent', ylim = 'own', figsize = (14,10), hist = False, bins = 30, overlap = 2, 
                                  colormap = [colormap_one, colormap_two], grid = True, legend = True)
        #axes = joypy.joyplot(df_pair[df_pair['GroupSize']==3], column=['GroupHIMass'], by="RoundedPercent", ylim='own', figsize=(14,10),hist=False, bins=50, overlap=2, colormap=[cm.cividis_r])

        # Decoration
        plt.title(r'\% of HI in central (rounded) vs. Group Mass', fontsize=22)
        axes[-1].set_xlabel(r' log M$_{\textrm{group}}$ [M$_{\odot}$]', fontsize=25)

        #plt.xaxis('A')
        plt.show()

        # Save the output
        fname = f'./plots/Joy_plot_group{group_size_joy}.png'
        plt.savefig(fname)
        print(f"Saved plot to {fname}")
       
    elif how_many_groups == 2:
        print('Insert')
    
    elif len(group_size_for_joy_plot) == 3:

        print('Selected are 3 group sizes')
        f, axes = plt.subplots(11,3, figsize=(16,10))
        first_column = [axes[i][0] for i in range(11)]
        second_column = [axes[i][1] for i in range(11)]
        third_column = [axes[i][2] for i in range(11)]

        #print('First groups is group of size:', group_size_joy[0])
        #print(dataframe)
        #print(dataframe[dataframe['GroupSize'] == group_size_joy[0]])
        joypy.joyplot(dataframe[dataframe['GroupSize'] == group_size_for_joy_plot[0]], column=['GroupHIMass', 'GroupStellarMass'], by="RoundedPercent", ylim='own', figsize=(14,10),hist=False, bins=50, overlap=1, 
                                  color=['#e0ecf4', '#fff7bc'], ax=first_column)
        joypy.joyplot(dataframe[dataframe['GroupSize'] == group_size_for_joy_plot[1]], column=['GroupHIMass', 'GroupStellarMass'], by="RoundedPercent", ylim='own', figsize=(14,10),hist=False, bins=50, overlap=1, 
                                  color=['#9ebcda', '#fec44f'], ax=second_column)
        joypy.joyplot(dataframe[dataframe['GroupSize'] == group_size_for_joy_plot[2]], column=['GroupHIMass', 'GroupStellarMass'], by="RoundedPercent", ylim='own', figsize=(14,10),hist=False, bins=50, overlap=1, 
                                  color=['#8c96c6', '#fe9929'], ax=third_column)

        axes[0][0].text(9.5, 1.4, "Groups of {0}".format(group_size_for_joy_plot[0]), color="k", fontsize=20)
        axes[0][1].text(10.4, 1.8, "Groups of {0}".format(group_size_for_joy_plot[1]), color="k", fontsize=20)
        axes[0][2].text(10.5, 1.7, "Groups of {0}".format(group_size_for_joy_plot[2]), color="k", fontsize=20)

        axes[0][0].text(6.5, -2, r'\% of HI in central (rounded)', rotation = 90, fontsize=20)

        #plt.title(r'\% of HI in central (rounded) vs. Group Mass', fontsize=22)

        plt.xlabel(r' log M$_{\textrm{group}}$ [M$_{\odot}$]', fontsize=25)

        plt.show()
        plt.tight_layout()
        # Save the output
        fname = f'./plots/Joy_plot_groups_of_size_{group_size_for_joy_plot[0]}_{group_size_for_joy_plot[1]}_{group_size_for_joy_plot[2]}.png'
        plt.savefig(fname)
        print(f"Saved plot to {fname}")
       

    else:
        print("Wrong group number. Either add it to the function or change it.")

    return


def find_groups_HI_in_central(define_percent_low, define_percent_limit):
    """
    Find groups based on the % of the HI in the central galaxy.
    
    Parameters:
    ===========
    define_percent_low: Integer/List if in for loop. Values of the lower limit of the % of HI in central.
    define_percent_limit: Integer/List if in the for loop. Values of the upper limit of the % of HI in central.

    Return:
    =======
    index_limit: List of of the index number in which the central galaxies satisfies the condition given with the define_percent_low/define_percent_high
 
    define_percent_low, define_percent_limit -- as above
    g_ind_limit: Nested list of integers. Indices  of the group/halo galaxies.  in which the central galaxies satisfies the condition given with the define_percent_low/define_percent_high
    c_ind_limit: Nested list of integers. Indices of the central galaxies.  in which the central galaxies satisfies the condition given with the define_percent_low/define_percent_high
    s_ind_limit: Nested list of integers. Indices of the satellite galaxies.  in which the central galaxies satisfies the condition given with the define_percent_low/define_percent_high
 

 
    
    Usage:
    ======
    Limit_ranges = ([0, 10], [10, 20], [20, 30], [30, 40], [40, 50], [50, 60], [60, 70], [70, 80], [80, 90], [90, 100])
    for i in Limit_ranges:
        define_percent_low = i[0]
        define_percent_limit = i[1]
  
  
    """

    # Find which items in a percentage list have central of HI percentage above the defined limit
    index_limit = []
    for idx, large_percentage in enumerate(percentage):
        if define_percent_low <= large_percentage <= define_percent_limit:  
            index_limit.append(idx)
    
    # For found items, find what are galaxy(halo) indices of those and save them
    # This way I get the indices of a halo where central has HI percentage above the defined limit
    g_ind_limit = []
    for idx, cen in enumerate(g_ind):
        for each_index in index_limit:
            if idx == each_index:
                g_ind_limit.append(cen)
                
    c_ind_limit = []
    for idx, cen in enumerate(c_ind):
        for each_index in index_limit:
            if idx == each_index:
                c_ind_limit.append(cen)
    
    s_ind_limit = []
    for idx, sat in enumerate(s_ind):
        for each_index in index_limit:
            if idx == each_index:
                s_ind_limit.append(sat)
    
    print(index_limit[0:10])
    print(c_ind_limit[0:10])
    print(g_ind_limit[0:10])
    print(s_ind_limit[0:10])
    return define_percent_low, define_percent_limit, g_ind_limit, c_ind_limit, index_limit, s_ind_limit

def compute_groups_HI_in_central(g_ind_limit, c_ind_limit, s_ind_limit):
    """
    Computed group properties. Groups are the ones that satisfy the condition of the % of HI in central.
    
    Parameters:
    ==========
    g_ind_limit: Nested list of integers. Indices  of the group/halo galaxies.  in which the central galaxies satisfies the condition given with the define_percent_low/define_percent_high
    c_ind_limit: Nested list of integers. Indices of the central galaxies.  in which the central galaxies satisfies the condition given with the define_percent_low/define_percent_high
    s_ind_limit: Nested list of integers. Indices of the satellite galaxies.  in which the central galaxies satisfies the condition given with the define_percent_low/define_percent_high
 
    Return:
    ======
    A number of group properties.
    g_m, g_st: List of floats. Respectively group HI mass and group stellar mass.
    p: List of floats. Percent of the HI in the central galaxy.
    hi_sat, st_sat: List of floats. Respectively HI mass and stellar mass of the satellites.
    central_mass, central_st_mass: List of floats. Respectivelly HI mass and stellar mass of the central galaxies.

    """

     # Compute central galaxies
    central_mass = np.sum(G['DiscHI'],axis=1)[c_ind_limit]*1e10/h
    central_st_mass = G['StellarMass'][c_ind_limit]*1e10/h
    
    # Compute group properties 
    g_m = []
    g_st = []
    p = []
    sat_m = []
    sat_sm = []
    #@jit
    #def compute_group_properties():
    for i in tqdm(g_ind_limit):
        g_mass = np.sum(G['DiscHI'],axis=1)[i]*1e10/h
        g_st_mass = G['StellarMass'][i]*1e10/h
        
        g_m.append(np.sum(g_mass, axis=0))
        g_st.append(np.sum(g_st_mass, axis=0))
    
    for i in s_ind_limit:
        g_mass_sat = np.sum(G['DiscHI'],axis=1)[i]*1e10/h
        g_st_mass_sat = G['StellarMass'][i]*1e10/h
        sat_m.append(g_mass_sat)
        sat_sm.append(g_st_mass_sat)
    
    # Calculate percentage   
    perc = (central_mass.ravel()/g_m)*100
    p.append(perc)
    
    # The list has to be flatten because I have not equal elements [2, 3, 4] [2, 3] [4, 5, 6, 6]
    # I use the chain to flatten the list and then convert back to list from chain format.
    chain2 = itertools.chain(*sat_m)
    chain1 = itertools.chain(*sat_sm)
    st_sat = np.log10(list(chain1))
    hi_sat = np.log10(list(chain2))


    return g_m, g_st, p, hi_sat, st_sat, central_mass, central_st_mass



def plot_limit_HI_in_central(limit_low, limit_high, central_st_mass, central_mass, st_sat, hi_sat):
    """
    Plot the MHI versus Mstar of the satellite and their central galaxies. Central galaxies satisfy given condition of the % of HI in the central galaxy. 
    
    Parameters:
    ==========
    hi_sat, st_sat: List of floats. Respectively HI mass and stellar mass of the satellites.
    central_mass, central_st_mass: List of floats. Respectivelly HI mass and stellar mass of the central galaxies.
    limit_low, limit_high: Integers/Floats. Limits as selected above, low and high of % in central galaxy. Using in a for loop to plot:

    Returns:
    ========
    Saves a number of plots.

    Usate:

    Limit_ranges = ([0, 10], [10, 20], [20, 30], [30, 40], [40, 50], [50, 60], [60, 70], [70, 80], [80, 90], [90, 100])

    for i in Limit_ranges:
        define_percent_low = i[0]
        define_percent_limit = i[1]
     
        limit_low, limit_high, limit_group, limit_central, limit_index, s_ind_limit = find_groups_HI_in_central(define_percent_low, define_percent_limit)
        g, s, p, sat_m, sat_sm, central_mass, central_st_mass = compute_groups_HI_in_central(limit_group, limit_central, s_ind_limit)
        plot_limit_HI_in_central(limit_low, limit_high, central_st_mass, central_mass, sat_sm, sat_m)
  
    """


    fig = plt.figure(figsize=(13,10))                                                               
    ax = fig.add_subplot(1,1,1)

    cm = plt.cm.get_cmap('YlGnBu')


    im = plt.scatter(np.log10(central_st_mass).ravel(), np.log10(central_mass).ravel(), s=120, alpha=0.8,  c=p[0].ravel(), cmap=cm, #color = 'white', edgecolor='#2c7fb8', 
                           linewidth=2, label=r'Rich centrals')
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', label=r'\% of the M$_{\mathrm{HI}}$ in central', pad=0.01)
    cbar.ax.tick_params(labelsize=20, size=28) 

    im = plt.scatter(st_sat, hi_sat, s=120, alpha = 0.8,
                            color='lightgrey', edgecolor='k', label=r'Their satellites', zorder=-2)

    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)

    ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
    ax.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)

    ax.legend(loc=3)
    leg = ax.get_legend()
    #leg.legendHandles[0].set_color('k')
    plt.xlim(8.7,12.2)
    plt.ylim(4, 11.1)
    plt.tight_layout()
    #plt.savefig('./plots/Limit_rich_centrals_and_satellites.png')

     # Save the output
    fname = f'./plots/Limit_{limit_low}_and_{limit_high}_rich_centrals_and_satellites.png'
    plt.savefig(fname)
    print(f"Saved plot to {fname}")
   




    #########################################################################################################
    ########################################### Handle the functions ########################################
    #########################################################################################################
    
if __name__ == "__main__":

    outdir = '/fred/oz042/rdzudzar/python/plots/' # where the plots will be saved
    if not os.path.exists(outdir): os.makedirs(outdir)

    Mass_cutoff = 0.06424
    number_of_files = 15
    h = 0.73
    group_size_for_two_sided = 2
    
    # Name of the colormaps and group size used fot the joy plot.
    cmap_one = cut_colormap(plt.cm.viridis_r, 0.25, 1)      
    cmap_two = cut_colormap(plt.cm.plasma_r, 0.25, 1)
    group_size_for_joy_plot = np.array([3, 4, 5])
    how_many_groups = 3    

    indir = '/fred/oz042/rdzudzar/simulation_catalogs/darksage/millennium_latest/output/' # directory where the Dark Sage data are

    NpartMed = 100 # minimum number of particles for finding relevant medians for minima on plots

    print("Reading in {0} galaxies file".format(number_of_files))
    G = create_G_dataframe(indir, number_of_files)

	# Get the indices and dictionary(ies) needed for plots
	# Indices - all
    store_all_indices = extract_all_indices(G)

	# Group dictionary
    groups = create_groups_dictionary(store_all_indices)
	
	# All centrals and single centrals
    store_cen_indices = extract_central_indices(G)
    single_gal_ind = single_central_galaxies(groups, G)
	

	#This is updated dictionary which one can parse to the plotting function:
    updated_dict = create_cen_sat_from_groups_dict(groups, store_cen_indices) 

    # For two-sided histogram richer/poorer 
    grp_length = 5 
    richer_central_ind, poorer_central_ind, richer_sat_ind, poorer_sat_ind, richer_central_s_m, richer_central_hi_m, poorer_central_s_m, poorer_central_hi_m,  richer_sat_hi_m, poorer_sat_hi_m, richer_sat_s_m, poorer_sat_s_m = find_richer_central_for_Nsized_group(grp_length)



    ########################################################################################################
    ########################################### Make all plots #############################################
    ########################################################################################################
    
    #plot_len_max(G)
    #plot_single_galaxies(G, single_gal_ind)	
    #plot_group_numbers_and_sizes(updated_dict)
    #plot_mhi_vs_ms_3x3(updated_dict)

    #two_sided_histogram_group_and_single(updated_dict, group_size_for_two_sided)
    #two_sided_histogram_groups(updated_dict, group_size_for_two_sided)
    #hist_Mhi_vs_Mstar_each_group(updated_dict)
 

    #find_richer_central_for_Nsized_group(grp_length)

    #Centrals more HI rich than the sum of their satellites
    #two_sided_histogram_rich(richer_central_s_m, richer_central_hi_m, poorer_central_s_m, poorer_central_hi_m) 
   

    #Central gals which are more HI rich than the sum of their satellites --- also marks position of their satellites
    # And vice versa
    #two_sided_histogram_rich_groups(grp_length, richer_central_s_m.ravel(), richer_central_hi_m.ravel(), poorer_central_s_m.ravel(), poorer_central_hi_m.ravel(),
    #                            richer_sat_s_m.ravel(), richer_sat_hi_m.ravel(), poorer_sat_s_m.ravel(), poorer_sat_hi_m.ravel())
    
    # Group indices
    c_ind, s_ind, g_ind = group_indices_for_percentage(updated_dict)
    
    # Compute group masses (HI and stellar) 
    g_m, g_st, percentage, BTT_cen, Mvir_cen, Rvir_cen, group_length = compute_group_properties(c_ind, s_ind, g_ind)
    
    plot_per_cent_of_HI_in_central(g_m, g_st, percentage)

    df_pairplot = make_dataframe_for_pairplot(g_st, g_m, percentage, group_length, BTT_cen, Mvir_cen, Rvir_cen)

    #make_pair_plot(df_pairplot)

    df_joy = make_dataframe_for_joyplot(df_pairplot)
       
    # This will break if there are no galaxies/groups of certain type, or go up to 100% 

    #make_joy_plot(df_joy, group_size_for_joy_plot, how_many_groups, cmap_one, cmap_two) 

    # For the Limit of HI in centrala find distribution of the central galaxies and theirr satellites 
    # Make a loop to get more values
    Limit_ranges = ([0, 10], [10, 20], [20, 30], [30, 40], [40, 50], [50, 60], [60, 70], [70, 80], [80, 90], [90, 100])

    for i in Limit_ranges:
        define_percent_low = i[0]
        define_percent_limit = i[1]
     
        limit_low, limit_high, limit_group, limit_central, limit_index, s_ind_limit = find_groups_HI_in_central(define_percent_low, define_percent_limit)
        g, s, p, sat_m, sat_sm, central_mass, central_st_mass = compute_groups_HI_in_central(limit_group, limit_central, s_ind_limit)
        plot_limit_HI_in_central(limit_low, limit_high, central_st_mass, central_mass, sat_sm, sat_m)


