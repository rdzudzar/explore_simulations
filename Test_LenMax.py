#!/usr/bin/env python
# coding: utf-8

# In[42]:


from __future__ import print_function # Always do this >:( 
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import itertools
from tqdm import trange
from tqdm import tqdm
import time

#LineProfiler
get_ipython().run_line_magic('load_ext', 'line_profiler')

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))


# In[43]:


from jupyterthemes.stylefx import set_nb_theme
set_nb_theme('solarizedl')
#set_nb_theme('solarizedd')


# In[44]:


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


# In[45]:


# Make plots for z=0 galaxies that are typically used to constrain Dark Sage.

from pylab import *
import os
import routines as r
import random

# Warnings are annoying
import warnings
warnings.filterwarnings("ignore")


###### USER NEEDS TO SET THESE THINGS ######
indir = '/fred/oz042/rdzudzar/simulation_catalogs/darksage/mini-millennium/output/' # directory where the Dark Sage data are
sim = 0 # which simulation Dark Sage has been run on -- if it's new, you will need to set its defaults below.
#   0 = Mini Millennium, 1 = Full Millennium, 2 = SMDPL

fpre = 'model_z0.000' # what is the prefix name of the z=0 files
files = range(8) # list of file numbers you want to read

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


# In[46]:


##### READ DARK SAGE DATA #####
DiscBinEdge = np.append(0, np.array([FirstBin*ExponentBin**i for i in range(Nannuli)])) / h
G = r.darksage_snap(indir+fpre, files, Nannuli=Nannuli)
######  ================= #####


# In[47]:


##### SET PLOTTING DEFAULTS #####
fsize = 26
matplotlib.rcParams.update({'font.size': fsize, 'xtick.major.size': 10, 'ytick.major.size': 10, 'xtick.major.width': 1, 'ytick.major.width': 1, 'ytick.minor.size': 5, 'xtick.minor.size': 5, 'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.linewidth': 1, 'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Times New Roman', 'legend.numpoints': 1, 'legend.columnspacing': 1, 'legend.fontsize': fsize-4, 'xtick.top': True, 'ytick.right': True})

NpartMed = 100 # minimum number of particles for finding relevant medians for minima on plots

outdir = '/fred/oz042/rdzudzar/python/test_len/' # where the plots will be saved
if not os.path.exists(outdir): os.makedirs(outdir)
######  =================== #####


# In[48]:


#From one notebook: %store data
#Import to another notebook: %store -r data

#%store -r G
h = 0.73


# In[49]:


# Make a sub_section of dataframe G so that it is controlled by the LenMax
print(len(G['LenMax']))
G = G [G['LenMax']>=100 ]
print(len(G['LenMax']))


# In[50]:


##### SET PLOTTING DEFAULTS #####
fsize = 26
matplotlib.rcParams.update({'font.size': fsize, 'xtick.major.size': 10, 'ytick.major.size': 10, 'xtick.major.width': 1, 'ytick.major.width': 1, 'ytick.minor.size': 5, 'xtick.minor.size': 5, 'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.linewidth': 1, 'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Times New Roman', 'legend.numpoints': 1, 'legend.columnspacing': 1, 'legend.fontsize': fsize-4, 'xtick.top': True, 'ytick.right': True})


# In[51]:


#Test
Mlen_central = G['LenMax'] [ (G['CentralGalaxyIndex']==G['GalaxyIndex']) & (G['StellarMass']>0.06424) ]/h
print(min(Mlen_central))

fig = plt.figure(figsize=(10,10))                                                               
ax = fig.add_subplot(1,1,1)

plt.hist( np.log10(Mlen_central), bins=80, color='lightgrey',  label=r'Central galaxies [cut]')
plt.axvline(2, 0, label='Len=100')

ax.set_xlabel(r'log Len', fontsize=25)
#ax.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)

plt.legend()
plt.show()


# ## Extract indices and masses of central galaxies

# In[52]:


Mass_cutoff = 0.06424 #Stellar mass cut

store_cen_indices = []
store_cen_indices = np.where(G["Type"] == 0)[0] # These are indices of the central galaxies

print(len(store_cen_indices))


# ## Extract indices and masses of ALL galaxies in a halo

# In[53]:


central_IDs, unique_counts = np.unique(G["CentralGalaxyIndex"], return_counts=True)
group_offset = np.cumsum(unique_counts)

count = 0
store_all_indices = []

argsort_central_gal_idx = np.argsort(G["CentralGalaxyIndex"])

for offset in group_offset:
    inds = np.arange(count, offset)
    my_list = argsort_central_gal_idx[inds]
    
    store_all_indices.append(my_list)
    
    count += len(my_list)

#print(empty[0:100])
    
for group in store_all_indices:
    if not np.all(G["CentralGalaxyIndex"][group] == G["CentralGalaxyIndex"][group][0]):
        print(G["CentralGalaxyIndex"][group])
#        print(group)


# # Create dictionary for group sizes 
# ### Based on the number of galaxies in a group

# In[54]:


groups = {} #initiate groups dictionary

#i = range(0, len(store_all_indices)) #gives range of halos

for item in range(len(store_all_indices)):

#for item in trange(len(store_all_indices)):
#for item in trange(100):
    indices = store_all_indices[item]
    halo_length = len(indices) #gives length of each halo (central+satellite)
    try:
        groups[halo_length].append(indices)
    except KeyError:
        groups[halo_length] = []
        groups[halo_length].append(indices)
#print(groups[3])    


# # Single galaxies

# In[55]:


#CENTRALS
Group_of_one = 1

Mstellar_single_gal = []
Mcoldgas_single_gal = []
Mvir_single_gal = []
BTT_single = []

#initiate condition for group
for group in groups[Group_of_one]:
    #if (G['StellarMass'][ G['Len']>=100 ]).any():
    #    continue
    if  (G["StellarMass"][group] < Mass_cutoff).any(): #creates array of booleans; if theese are true hit continue
        continue
    #Store single galaxies
    
    BTT_single_gal= (G['InstabilityBulgeMass'][group]*1e10/h + G['MergerBulgeMass'][group]*1e10/h) / ( G['StellarMass'][group]*1e10/h )
        
    Mstellar_single_gal.append(G["StellarMass"][group]*1e10/h+1)
    Mcoldgas_single_gal.append(np.sum(G['DiscHI'],axis=1)[group]*1e10/h+1)
    Mvir_single_gal.append(G["Mvir"][group]*1e10/h+1)


# In[56]:


print(len(Mstellar_single_gal))
fig = plt.figure(figsize=(10,10))                                                               
ax = fig.add_subplot(1,1,1)


plt.plot(np.log10(Mstellar_single_gal), np.log10(Mcoldgas_single_gal), 'o', 
         color='lightgrey', markeredgecolor='k', markersize=8, markeredgewidth=0.2, label='Single galaxies')


ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
ax.set_ylabel(r'log M$_{HI}$ [M$_{\odot}$]',fontsize=25)
plt.legend(loc=4)
plt.show()


# # Create cen & sat from groups dictionary

# In[57]:


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
            
            if (G["StellarMass"][group] < my_mass_cutoff).any() : #creates array of booleans; if theese are true hit continue
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
                    


# In[58]:


#This is updated dictionary which one can parse to the plotting function:
updated_dict = create_cen_sat_from_groups_dict(groups, store_cen_indices) 


# In[65]:


# Initiate figure and how large it will be
def mhi_vs_ms_3x3(groups_dict):
    
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
    
    
    fig, ax = plt.subplots(nrows=4, ncols=5, sharex='col', sharey='row', figsize=(21, 18))
    
    ax[0][0].plot(np.log10(Mstellar_single_gal), np.log10(Mcoldgas_single_gal), 'o', color='lightgrey',markersize=8, label='Single galaxies')
    
    # Put plot in row and columns:3by3 are 00, 01, 02, 10, 11, 12, 20, 21, 22 
    row_count = 0
    
    for size in range(1, 20, 1): #I'm planning to have 1-9 sized groups. Made to form trange(1, 9, 1) as 3x3 pannels 
        try:    
            if size % 5 == 0:
                row_count += 1
            this_ax = ax[row_count, size%5] #Axis. Created for plotting 3x3 plots
            
            group_size = size+1 # Select number of galaxies per group --- adding +1 because it is counting from 0.
        
            central_gals = groups_dict[group_size]["Centrals"]  #List of integers. Contains indices of central galaxies.  List is obtained through the dictionary. 
            sat_gals = groups_dict[group_size]["Satellites"] #List of integers. Contains indices of satellite galaxies. List is obtained through the dictionary.
        
            # Plot single galaxies           
            this_ax.plot(np.log10(Mstellar_single_gal), np.log10(Mcoldgas_single_gal), 'o', color='lightgrey',markersize=8)
            
                
            # Do plotting of groups of N-length; it will be placed on each subplot + +.
            # Centrals
            this_ax.plot(np.log10(G['StellarMass'][central_gals]*1e10/h+1), 
                             np.log10(np.sum(G['DiscHI'],axis=1)[central_gals]*1e10/h+1), 
                             'o', color='white', markeredgecolor='#1f78b4', markersize=8, markeredgewidth=2)
            # Satellites
            this_ax.plot(np.log10(G['StellarMass'][sat_gals]*1e10/h+1), 
                             np.log10(np.sum(G['DiscHI'],axis=1)[sat_gals]*1e10/h+1), 
                             'o', color='#1f78b4', markersize=8)
        
            # Add label for each sub-plot to know the group size and what is central/satellite
            this_ax.plot([], [], 'o', color='white', label='Groups of %.f' %group_size)
            this_ax.plot([], [], 'o', color='lightgrey', label='Single galaxies')
            this_ax.plot([], [], 'o', color='#1f78b4', markersize=8, label='Satellite')
            this_ax.plot([], [], 'o', color='white', markeredgecolor='#1f78b4', markersize=8, markeredgewidth=2, label='Central')
            #Add legend    
            leg = this_ax.legend(loc=4,frameon=True,fancybox=True, framealpha=0.6, fontsize=16)
        except KeyError:
            print('KeyError happended - plot size is larger than dictionary')
            continue
    # Add x and y axis labels only on the plot edges since there will be no space between panels
    for row in range(4):
        this_ax = ax[row,0]
        this_ax.set_ylabel(r'log M$_{HI}$ [M$_{\odot}$]',fontsize=25)
    for col in range(5):
        this_ax = ax[3,col]
        this_ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
    # Tight layout and remove spacing between subplots
    plt.tight_layout()
    # My first plot is only single galaxies so I add legend separately
    ax[0][0].legend(loc=4,frameon=True,fancybox=True, framealpha=0.6, fontsize=16)

    plt.subplots_adjust(wspace = 0.0, hspace = 0.0)
    plt.show()
    #return fig


# In[66]:


#Use line_profiler to test how fast is each line of the code

#put this line in front of funtion which you are calling:
#%lprun -f mhi_vs_ms_3x3 

mhi_vs_ms_3x3(updated_dict)


# In[ ]:





# In[ ]:





# In[ ]:




