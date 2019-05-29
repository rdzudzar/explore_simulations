from __future__ import print_function # Always do this >:( 
from __future__ import division
#%load_ext line_profiler

import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import trange
from tqdm import tqdm
import time
from numba import jit

# Import code for reading in the data from Dark Sage

from pylab import *
import os
import routines as r
import random

# Warnings are annoying
import warnings
warnings.filterwarnings("ignore")

Mass_cutoff = 0.06424

@jit
def galdtype_darksage(Nannuli=30):
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

def create_G_dataframe(n_files):
	
	n_files = 30
	###### USER NEEDS TO SET THESE THINGS ######
	indir = '/fred/oz042/rdzudzar/simulation_catalogs/darksage/millennium_latest/output/' # directory where the Dark Sage data are
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


	##### SET PLOTTING DEFAULTS #####
	fsize = 26
	matplotlib.rcParams.update({'font.size': fsize, 'xtick.major.size': 10, 'ytick.major.size': 10, 'xtick.major.width': 1, 'ytick.major.width': 1, 'ytick.minor.size': 5, 'xtick.minor.size': 5, 'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.linewidth': 1, 'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Times New Roman', 'legend.numpoints': 1, 'legend.columnspacing': 1, 'legend.fontsize': fsize-4, 'xtick.top': True, 'ytick.right': True})

	NpartMed = 100 # minimum number of particles for finding relevant medians for minima on plots

	outdir = '/fred/oz042/rdzudzar/python/' # where the plots will be saved
	if not os.path.exists(outdir): os.makedirs(outdir)
	######  =================== #####


	# Make a sub_section of dataframe G so that it is controlled by the LenMax
	print(len(G['LenMax']))
	G = G [G['LenMax']>=100 ]
	print(len(G['LenMax']))

	return G


def extract_central_indices(G):
	store_cen_indices = []
	store_cen_indices = np.where(G["Type"] == 0)[0]
	return store_cen_indices

def extract_all_indices(G):
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
def create_cen_sat_from_groups_dict(groups, store_cen_indices, my_mass_cutoff=Mass_cutoff):
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
                    
