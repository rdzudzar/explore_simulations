#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import trange
from tqdm import tqdm
import time


# In[2]:


np.__version__


# In[3]:


#LineProfiler

from __future__ import print_function # Always do this >:( 
from __future__ import division
get_ipython().run_line_magic('load_ext', 'line_profiler')


# In[4]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))


# In[5]:


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


# # Plot

# In[6]:


#print(range(30,60))


# In[7]:


# Make plots for z=0 galaxies that are typically used to constrain Dark Sage.

from pylab import *
import os
import routines as r
import random

# Warnings are annoying
import warnings
warnings.filterwarnings("ignore")


###### USER NEEDS TO SET THESE THINGS ######
indir = '/fred/oz042/rdzudzar/simulation_catalogs/darksage/millennium_latest/output/' # directory where the Dark Sage data are
sim = 1 # which simulation Dark Sage has been run on -- if it's new, you will need to set its defaults below.
#   0 = Mini Millennium, 1 = Full Millennium, 2 = SMDPL

fpre = 'model_z0.000' # what is the prefix name of the z=0 files
files = range(50) # list of file numbers you want to read

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


# In[8]:


##### READ DARK SAGE DATA #####
DiscBinEdge = np.append(0, np.array([FirstBin*ExponentBin**i for i in range(Nannuli)])) / h
G = r.darksage_snap(indir+fpre, files, Nannuli=Nannuli)
######  ================= #####


# In[9]:


##### SET PLOTTING DEFAULTS #####
fsize = 26
matplotlib.rcParams.update({'font.size': fsize, 'xtick.major.size': 10, 'ytick.major.size': 10, 'xtick.major.width': 1, 'ytick.major.width': 1, 'ytick.minor.size': 5, 'xtick.minor.size': 5, 'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.linewidth': 1, 'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Times New Roman', 'legend.numpoints': 1, 'legend.columnspacing': 1, 'legend.fontsize': fsize-4, 'xtick.top': True, 'ytick.right': True})

NpartMed = 100 # minimum number of particles for finding relevant medians for minima on plots

outdir = '/fred/oz042/rdzudzar/python/' # where the plots will be saved
if not os.path.exists(outdir): os.makedirs(outdir)
######  =================== #####


# In[10]:


#%store G


# ## Want to store output as Pandas dataframe; Have to convert multidimensional arrays into Pandas series first:

# In[11]:


import pandas as pd


# In[10]:


# This is a way to converte multi dimensional data into pd.Series and then load these into the pandas dataframe
Pos = []
for p in G['Pos']:
    Pos.append(p)
Pos_df = pd.Series(Pos, dtype=np.dtype("object"))

Vel = []
for v in G['Vel']:
    Vel.append(v)
Vel_df = pd.Series(Vel, dtype=np.dtype("object"))

Spin = []
for s in G['Spin']:
    Spin.append(s)
Spin_df = pd.Series(Spin, dtype=np.dtype("object"))

Disc_r = []
for d in G['DiscRadii']:
    Disc_r.append(d)
Disc_df = pd.Series(Disc_r, dtype=np.dtype("object"))

Disc_gas = []
for g in G['DiscGas']:
    Disc_gas.append(g)
Disc_gas_df = pd.Series(Disc_gas, dtype=np.dtype("object"))

Disc_stars = []
for g in G['DiscStars']:
    Disc_stars.append(g)
Disc_stars_df = pd.Series(Disc_stars, dtype=np.dtype("object"))

SpinStars = []
for g in G['SpinStars']:
    SpinStars.append(g)
SpinStars_df = pd.Series(SpinStars, dtype=np.dtype("object"))

SpinGas = []
for g in G['SpinGas']:
    SpinGas.append(g)
SpinGas_df = pd.Series(SpinGas , dtype=np.dtype("object"))

SpinClassicalBulge = []
for g in G['SpinClassicalBulge']:
    SpinClassicalBulge.append(g)
SpinClassicalBulge_df = pd.Series(SpinClassicalBulge, dtype=np.dtype("object"))

DiscHI = []
for g in G['DiscHI']:
    DiscHI.append(g)
DiscHI_df = pd.Series(DiscHI, dtype=np.dtype("object"))

DiscH2 = []
for g in G['DiscH2']:
    DiscH2.append(g)
DiscH2_df = pd.Series(DiscH2, dtype=np.dtype("object"))

DiscSFR = []
for g in G['DiscSFR']:
    DiscSFR.append(g)
DiscSFR_df = pd.Series(DiscSFR, dtype=np.dtype("object"))

DiscGasMetals = []
for g in G['DiscGasMetals']:
    DiscGasMetals.append(g)
DiscGasMetals_df = pd.Series(DiscGasMetals, dtype=np.dtype("object"))

DiscStarsMetals = []
for g in G['DiscStarsMetals']:
    DiscStarsMetals.append(g)
DiscStarsMetals_df = pd.Series(DiscStarsMetals, dtype=np.dtype("object"))


# # Storing DarkSage output as Pandas dataframe: DS

# In[10]:


DS = pd.DataFrame({'Type'   : G['Type'                      ],
'GalaxyIndex'               : G['GalaxyIndex'               ],
'HaloIndex'                 : G['HaloIndex'                 ],
'SimulationHaloIndex'       : G['SimulationHaloIndex'       ],
'TreeIndex'                 : G['TreeIndex'                 ],
'SnapNum'                   : G['SnapNum'                   ],
'CentralGalaxyIndex'        : G['CentralGalaxyIndex'        ],
'CentralMvir'               : G['CentralMvir'               ],
'mergeType'                 : G['mergeType'                 ],
'mergeIntoID'               : G['mergeIntoID'               ],
'mergeIntoSnapNum'          : G['mergeIntoSnapNum'          ],
'dT'                        : G['dT'                        ],
'Pos'                       : Pos_df,
'Vel'                       : Vel_df                       ,
'Spin'                      : Spin_df                      ,
'Len'                       : G['Len'                       ],
'LenMax'                    : G['LenMax'                    ],
'Mvir'                      : G['Mvir'                      ],
'Rvir'                      : G['Rvir'                      ],
'Vvir'                      : G['Vvir'                      ],
'Vmax'                      : G['Vmax'                      ],
'VelDisp'                   : G['VelDisp'                   ],
'DiscRadii'                 : Disc_df,
'ColdGas'                   : G['ColdGas'                   ],
'StellarMass'               : G['StellarMass'               ],
'MergerBulgeMass'           : G['MergerBulgeMass'           ],
'InstabilityBulgeMass'      : G['InstabilityBulgeMass'      ],
'HotGas'                    : G['HotGas'                    ],
'EjectedMass'               : G['EjectedMass'               ],
'BlackHoleMass'             : G['BlackHoleMass'             ],
'IntraClusterStars'         : G['IntraClusterStars'         ],
'DiscGas'                   : Disc_gas_df,
'DiscStars'                 : Disc_stars_df,
'SpinStars'                 : SpinStars_df,
'SpinGas'                   : SpinGas_df,
'SpinClassicalBulge'        : SpinClassicalBulge_df,
'StarsInSitu'               : G['StarsInSitu'               ],
'StarsInstability'          : G['StarsInstability'          ],
'StarsMergeBurst'           : G['StarsMergeBurst'           ],
'DiscHI'                    : DiscHI_df,
'DiscH2'                    : DiscH2_df,
'DiscSFR'                   : DiscSFR_df,
'MetalsColdGas'             : G['MetalsColdGas'             ],
'MetalsStellarMass'         : G['MetalsStellarMass'         ],
'ClassicalMetalsBulgeMass'  : G['ClassicalMetalsBulgeMass'  ],
'SecularMetalsBulgeMass'    : G['SecularMetalsBulgeMass'    ],
'MetalsHotGas'              : G['MetalsHotGas'              ],
'MetalsEjectedMass'         : G['MetalsEjectedMass'         ],
'MetalsIntraClusterStars'   : G['MetalsIntraClusterStars'   ],
'DiscGasMetals'             : DiscGasMetals_df,
'DiscStarsMetals'           : DiscStarsMetals_df,
'SfrFromH2'                 : G['SfrFromH2'                 ],
'SfrInstab'                 : G['SfrInstab'                 ],
'SfrMergeBurst'             : G['SfrMergeBurst'             ],
'SfrDiskZ'                  : G['SfrDiskZ'                  ],
'SfrBulgeZ'                 : G['SfrBulgeZ'                 ],
'DiskScaleRadius'           : G['DiskScaleRadius'           ],
'CoolScaleRadius'           : G['CoolScaleRadius'           ],
'StellarDiscScaleRadius'    : G['StellarDiscScaleRadius'    ],
'Cooling'                   : G['Cooling'                   ],
'Heating'                   : G['Heating'                   ],
'LastMajorMerger'           : G['LastMajorMerger'           ],
'LastMinorMerger'           : G['LastMinorMerger'           ],
'OutflowRate'               : G['OutflowRate'               ],
'infallMvir'                : G['infallMvir'                ],
'infallVvir'                : G['infallVvir'                ],
'infallVmax'                : G['infallVmax'                ]})


# In[11]:


DS.columns


# In[ ]:





# In[ ]:





# # My stuff

# ### Cut based on the stellar mass
# ### The HI mass: np.sum(G['DiscHI'],axis=1)

# In[12]:


#Mass resolution 8.6x10^8 Msun/h  
#Make a cut at stellar masses of logMstar=8.8 -- based on the 100 particle median mass -- shown above
#0.088*h is 0.06424 which then becomes cut of 8.8 when divided with h
Stellar_mass_cut = G['StellarMass'] [ G['StellarMass']>0.06424 ]/h
HI_mass_cut = np.sum(G['DiscHI'],axis=1) [ G['StellarMass']>0.06424 ]/h
Mvir_mass_cut = G['Mvir'] [ G['StellarMass']>0.06424 ]/h


# In[13]:


print(len(Stellar_mass_cut))


# In[14]:


#CENTRALS
Mstellar_central_galaxies_cut = G['StellarMass'] [ (G['CentralGalaxyIndex']==G['GalaxyIndex']) & (G['StellarMass']>0.06424) & (np.sum(G['DiscHI'],axis=1)!=0)]/h
Mcoldgas_central_galaxies_cut = np.sum(G['DiscHI'],axis=1) [ (G['CentralGalaxyIndex']==G['GalaxyIndex']) & (G['StellarMass']>0.06424) & (np.sum(G['DiscHI'],axis=1)!=0)]/h
Mvir_central_galaxies_cut = G['Mvir'] [ (G['CentralGalaxyIndex']==G['GalaxyIndex']) & (G['StellarMass']>0.06424) & (np.sum(G['DiscHI'],axis=1)!=0)]/h


# In[15]:


#SATELLITES
Mstellar_satellite_galaxies_cut = G['StellarMass'] [ (G['CentralGalaxyIndex']!=G['GalaxyIndex']) & (G['StellarMass']>0.06424) & (np.sum(G['DiscHI'],axis=1)!=0)]/h
Mcoldgas_satellite_galaxies_cut = np.sum(G['DiscHI'],axis=1) [ (G['CentralGalaxyIndex']!=G['GalaxyIndex']) & (G['StellarMass']>0.06424) & (np.sum(G['DiscHI'],axis=1)!=0)]/h
Mvir_satellite_galaxies_cut = G['Mvir'] [ (G['CentralGalaxyIndex']!=G['GalaxyIndex']) & (G['StellarMass']>0.06424) & (np.sum(G['DiscHI'],axis=1)!=0)]/h


# In[15]:


#TO STORE VARIABLES WHICH CAN BE USED IN ANOTHER NOTEBOOK
#central
get_ipython().run_line_magic('store', 'Mstellar_central_galaxies_cut')
get_ipython().run_line_magic('store', 'Mcoldgas_central_galaxies_cut')
get_ipython().run_line_magic('store', 'Mvir_central_galaxies_cut')
#satellite
get_ipython().run_line_magic('store', 'Mstellar_satellite_galaxies_cut')
get_ipython().run_line_magic('store', 'Mcoldgas_satellite_galaxies_cut')
get_ipython().run_line_magic('store', 'Mvir_satellite_galaxies_cut')


# # Plot MHI vs M* relation

# In[11]:


fig = plt.figure(figsize=(10,10))                                                               
ax = fig.add_subplot(1,1,1)

plt.plot( np.log10( (Mstellar_satellite_galaxies_cut*10**10)) ,np.log10( (Mcoldgas_satellite_galaxies_cut*10**10)), 'o', color='#1f78b4', markersize=5, label=r'Satellite galaxies [cut]')
#plt.plot(cut_stell*1e10/h, 'o', color='r')
plt.plot( np.log10( (Mstellar_central_galaxies_cut*10**10)) ,np.log10( (Mcoldgas_central_galaxies_cut*10**10)), 'o', color='lightgrey', markersize=5, label=r'Central galaxies [cut]')


ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
ax.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)

plt.legend()
plt.show()


# In[12]:


fig = plt.figure(figsize=(10,10))                                                               
ax = fig.add_subplot(1,1,1)

#plt.plot(cut_stell*1e10/h, 'o', color='r')
plt.hist( np.log10(Mcoldgas_central_galaxies_cut*1e10), bins=100, color='lightgrey', alpha=0.8, label=r'Central galaxies [cut]')
plt.hist(np.log10(Mcoldgas_satellite_galaxies_cut*1e10), bins=100, color='#1f78b4', alpha=1, label=r'Satellite galaxies [cut]')


ax.set_ylabel(r'N$_{\textrm{gal}}$', fontsize=25)
ax.set_xlabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)

plt.legend()
#plt.xlim(-0.1, 3)
plt.show()


# In[16]:


Mlen_central = G['LenMax'] [ (G['CentralGalaxyIndex']==G['GalaxyIndex']) & (G['StellarMass']>0.06424)& (G['LenMax']>=100)]/h
Mlen_all_central = G['LenMax'] [ (G['CentralGalaxyIndex']==G['GalaxyIndex']) & (G['StellarMass']>0.06424)]/h
print(len(Mlen_central))
print(len(Mlen_all_central))

Mlen_sat = G['LenMax'] [ (G['CentralGalaxyIndex']!=G['GalaxyIndex']) & (G['StellarMass']>0.06424) & (G['LenMax']>=100)]/h
Mlen_all_sat = G['LenMax'] [ (G['CentralGalaxyIndex']!=G['GalaxyIndex']) & (G['StellarMass']>0.06424) ]/h
print(len(Mlen_sat))
print(len(Mlen_all_sat))


# In[127]:



fig = plt.figure(figsize=(10,10))                                                               
ax = fig.add_subplot(1,1,1)

plt.hist( np.log10(Mlen_all_central), bins=50, color='grey',  label=r'Central all')
plt.hist( np.log10(Mlen_central), bins=50, color='lightgrey',  label=r'Central L>100')
plt.hist( np.log10(Mlen_all_sat), bins=50, color='blue',  label=r'Sat all')
plt.hist( np.log10(Mlen_sat), bins=50, color='lightblue',  label=r'Sat L>100')


plt.axvline(2, 0, label='Len=100', color='red')

ax.set_xlabel(r'log Len', fontsize=25)
#ax.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)

plt.legend()
plt.show()


# In[ ]:





# In[17]:


from matplotlib.ticker import NullFormatter

# the random data
x = np.log10( (Mstellar_satellite_galaxies_cut*10**10)) 
y = np.log10( (Mcoldgas_satellite_galaxies_cut*10**10))

w = np.log10( (Mstellar_central_galaxies_cut*10**10))
z = np.log10( (Mcoldgas_central_galaxies_cut*10**10))

nullfmt = NullFormatter()         # no labels

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(1, figsize=(8, 8))

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# the scatter plot:
axScatter.scatter(w, z, color='lightgrey', label='Centrals')
axScatter.scatter(x, y, color='#1f78b4', label='Satellites')

axScatter.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
axScatter.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)

# now determine nice limits by hand:
binwidth = 0.1
xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
lim = (int(xymax/binwidth) + 1) * binwidth

axScatter.set_xlim((8.8, lim))
axScatter.set_ylim((2, lim))

bins = np.arange(8.8, lim + binwidth, binwidth)
axHistx.hist(w, bins=bins, color='white', edgecolor='k', linewidth=1)
axHisty.hist(z, bins=bins, orientation='horizontal', color='white', edgecolor='k', linewidth=1.2)

axHistx.hist(x, bins=bins, color='#1f78b4',alpha=0.4, edgecolor='black', linewidth=1)
axHisty.hist(y, bins=bins, orientation='horizontal', color='#1f78b4', alpha=0.4, edgecolor='black', linewidth=1.2)

axHistx.set_ylabel(r'N$_{\textrm{gal}}$', fontsize=22)
axHisty.set_xlabel(r'N$_{\textrm{gal}}$', fontsize=22)

axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

axScatter.legend()


plt.show()


# In[14]:


fig = plt.figure(figsize=(10,10))                                                               
ax = fig.add_subplot(1,1,1)

plt.plot( np.log10( (Mstellar_satellite_galaxies_cut*1e10)) ,np.log10( (Mcoldgas_satellite_galaxies_cut*1e10)/(Mstellar_satellite_galaxies_cut*1e10)), 'o', color='#1f78b4', markersize=5, label=r'Satellite galaxies [cut]')
#plt.plot(cut_stell*1e10/h, 'o', color='r')
plt.plot( np.log10( (Mstellar_central_galaxies_cut*1e10)) ,np.log10( (Mcoldgas_central_galaxies_cut*1e10)/(Mstellar_central_galaxies_cut*1e10)), 'o', color='lightgrey', markersize=5, label=r'Central galaxies [cut]')

ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
ax.set_ylabel(r'log f$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)
 
plt.legend()
plt.show()


# In[15]:


fig = plt.figure(figsize=(10,10))                                                               
ax = fig.add_subplot(1,1,1)

plt.plot( np.log10( (Mstellar_satellite_galaxies_cut*1e10)) ,np.log10( (Mvir_satellite_galaxies_cut*1e10)), 'o', color='#1f78b4', markersize=5, label=r'Satellite [cut]')
#plt.plot(cut_stell*1e10/h, 'o', color='r')
plt.plot( np.log10( (Mstellar_central_galaxies_cut*1e10)) ,np.log10( (Mvir_central_galaxies_cut*1e10)), 'o', color='lightgrey', markersize=5, label=r'Central [cut]')


ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
ax.set_ylabel(r'log M$_{\textrm{vir}}$ [M$_{\odot}$]',fontsize=25)

plt.legend()
plt.show()


# ## Bulge-to-total ratio (All)
# ##### I want to see only late-type galaxies

# In[17]:


BTT = (G['InstabilityBulgeMass'] + G['MergerBulgeMass']) / ( G['StellarMass'] ) # Find bulge to total ratio

# Including a cut in stellar mass
BTT_Disc = BTT[ ( G['StellarMass']>0.06424 ) & (BTT <= 0.5) ] # Disc dominated
BTT_Bulge = BTT[ ( G['StellarMass']>0.06424 ) & (BTT > 0.5) ] # Bulge dominated

Disc_dominated_SM = G['StellarMass'][ (G['StellarMass']>0.06424) & (BTT <= 0.5) ]
Bulge_dominated_SM = G['StellarMass'][ (G['StellarMass']>0.06424) & (BTT > 0.5) ]


# In[17]:


fig = plt.figure(figsize=(10,10))                                                               
ax = fig.add_subplot(1,1,1)

plt.plot( np.log10( (Disc_dominated_SM*1e10)) ,BTT_Disc, 'o', color='#1f78b4', markersize=5, label=r'Disc dominated')
#plt.plot(cut_stell*1e10/h, 'o', color='r')
plt.plot( np.log10( (Bulge_dominated_SM*1e10)) ,BTT_Bulge, 'o', color='lightgrey', markersize=5, label=r'Bulge dominated')


ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
ax.set_ylabel(r'B/T',fontsize=25)

plt.legend()
plt.show()


# In[ ]:





# In[32]:


from matplotlib.ticker import NullFormatter

# the random data
x = np.log10( (Mstellar_satellite_galaxies_cut*10**10)) 
y = np.log10( (Mvir_satellite_galaxies_cut*1e10))

w = np.log10( (Mstellar_central_galaxies_cut*10**10))
z = np.log10( (Mvir_central_galaxies_cut*1e10))

nullfmt = NullFormatter()         # no labels

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(1, figsize=(8, 8))

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# the scatter plot:
axScatter.scatter(w, z, color='lightgrey', label='Centrals')
axScatter.scatter(x, y, color='#1f78b4', label='Satellites', alpha=0.8)

axScatter.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
axScatter.set_ylabel(r'log M$_{\textrm{vir}}$ [M$_{\odot}$]',fontsize=25)

# now determine nice limits by hand:
binwidth = 0.1
xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
lim = (int(xymax/binwidth) + 1) * binwidth

axScatter.set_xlim((8.8, 12))
axScatter.set_ylim((9.8, 14.5))

bins = np.arange(8.8, lim + binwidth, binwidth)
axHistx.hist(w, bins=bins, color='white', edgecolor='k', linewidth=1)
axHisty.hist(z, bins=bins, orientation='horizontal', color='white', edgecolor='k', linewidth=1.2)

axHistx.hist(x, bins=bins, color='#1f78b4',alpha=0.4, edgecolor='black', linewidth=1)
axHisty.hist(y, bins=bins, orientation='horizontal', color='#1f78b4', alpha=0.4, edgecolor='black', linewidth=1)

axHistx.set_ylabel(r'N$_{\textrm{gal}}$', fontsize=22)
axHisty.set_xlabel(r'N$_{\textrm{gal}}$', fontsize=22)

axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

axScatter.legend(loc=4)


plt.show()


# # Find for each central galaxy its satellites

# In[20]:


#Central_g_stellar_mass = G['StellarMass'] [ (G['CentralGalaxyIndex']==G['GalaxyIndex']) & (G['StellarMass']>0.06424) & (np.sum(G['DiscHI'],axis=1)!=0)]/h
#Central_g_hi_mass = np.sum(G['DiscHI'],axis=1) [ (G['CentralGalaxyIndex']==G['GalaxyIndex']) & (G['StellarMass']>0.06424) & (np.sum(G['DiscHI'],axis=1)!=0)]/h
#Central_g_vir_mass = G['Mvir'] [ (G['CentralGalaxyIndex']==G['GalaxyIndex']) & (G['StellarMass']>0.06424) & (np.sum(G['DiscHI'],axis=1)!=0)]/h
#
#
#print(G['CentralGalaxyIndex'] [ G['CentralGalaxyIndex']==G['GalaxyIndex']] )
#print('Length', len(G['CentralGalaxyIndex'] [ G['CentralGalaxyIndex']==G['GalaxyIndex']] ))
#
#print(G['GalaxyIndex'] [ G['CentralGalaxyIndex']!=G['GalaxyIndex']] )
#print('Length', len(G['GalaxyIndex'] [ G['CentralGalaxyIndex']!=G['GalaxyIndex']]))
#
#print(len(G['CentralGalaxyIndex']))
#print(len(G['GalaxyIndex']))
#print(len(np.unique( G['CentralGalaxyIndex'] )))
#print(len(np.unique( G['GalaxyIndex'])))


# ## Find galaxies

# In[28]:


Sat_inds = np.where( G['CentralGalaxyIndex']!=G['GalaxyIndex'])[0]
print('Sat_inds', Sat_inds[:28])
print(len(Sat_inds))
Central_inds = np.where(G['CentralGalaxyIndex']==G['GalaxyIndex'])[0]
print('Central_inds', Central_inds[:15])
print(len(Central_inds))
central_IDs = np.unique( G['CentralGalaxyIndex']  )
print('unique central ID', central_IDs[:10])


# ## Extract indices and masses of central galaxies

# In[18]:


store_cen_indices = []
store_cen_indices = np.where(G["Type"] == 0)[0]


# ## Extract indices and masses of ALL galaxies in a halo

# In[19]:


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


# In[20]:


print(store_cen_indices[0:5])
print(store_all_indices[0:5])


# # Create dictionary for group sizes 
# ### Based on the number of galaxies in a group

# In[20]:


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


# # Single galaxies

# In[21]:


# SINGLE CENTRALS
Mass_cutoff = 0.06424
Group_of_one = 1

single_gal_ind = []

#initiate condition for group
for group in groups[Group_of_one]:
    if (G["StellarMass"][group] < Mass_cutoff).any(): #creates array of booleans; if theese are true hit continue
        continue
    
    #Store indices of single galaxies 
    single_gal_idx = group
    single_gal_ind.append(single_gal_idx)


# In[22]:


Mstellar_single_gal = G["StellarMass"][single_gal_ind]*1e10/h
Mcoldgas_single_gal = np.sum(G['DiscHI'],axis=1)[single_gal_ind]*1e10/h
Mvir_single_gal = G["Mvir"][single_gal_ind]*1e10/h
BTT_single = (G['InstabilityBulgeMass'][single_gal_ind]*1e10/h + G['MergerBulgeMass'][single_gal_ind]*1e10/h) / ( G['StellarMass'][single_gal_ind]*1e10/h )


# In[22]:


print(len(G["StellarMass"][ (G["StellarMass"] > Mass_cutoff) & (G['Len']>=100) ]))
print(len(G["StellarMass"][ (G["StellarMass"] > Mass_cutoff)]))


# In[24]:



fig = plt.figure(figsize=(10,10))                                                               
ax = fig.add_subplot(1,1,1)


plt.plot(np.log10(Mstellar_single_gal), np.log10(Mcoldgas_single_gal), 'o', 
         color='lightgrey', markeredgecolor='k', markersize=8, markeredgewidth=0.2, label='Single galaxies')


ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
ax.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)
plt.legend(loc=4)
plt.show()


# # All galaxies in groups that are N=given N
# #### if **any galaxy in the group** has Mass less than `Mass_cutoff`, we don't plot the entire group

# In[23]:


Mass_cutoff = 0.06424


# In[26]:


print(G["StellarMass"][groups[15][1]] > Mass_cutoff)
print(G["StellarMass"][groups[15][1]])


# ### Separated centrals and satellites

# In[24]:


#print(G["StellarMass"][groups[4][0]] < Mass_cutoff)
#print(G["StellarMass"][groups[4][0]])

Group_size = 2 # select number of galaxies per group

fig = plt.figure(figsize=(10,10))                                                               
ax = fig.add_subplot(1,1,1)
plt.plot( np.log10( (Mstellar_central_galaxies_cut*10**10)) ,np.log10( (Mcoldgas_central_galaxies_cut*10**10)), 'o', color='lightgrey', markersize=5, label=r'Central galaxies [cut]')

#initiate condition for plot
for group in groups[Group_size]:
    if (G["StellarMass"][group] < Mass_cutoff).any(): #creates array of booleans; if theese are true hit continue
        continue
            
        # Do plotting.
    #plot all galaxies here; for loops below plots satellite and centrals separatelly
    #plt.plot(np.log10(G['StellarMass'][group]*1e10/h), np.log10(G['ColdGas'][group]*1e10/h), 'o', color='blue', markersize=12)
    
    for i in group: #find central galaxies and plot them as red
        if i in store_cen_indices: 
            #print(i)
            plt.plot(np.log10(G['StellarMass'][i]*1e10/h), np.log10(np.sum(G['DiscHI'],axis=1)[i]*1e10/h), 
                     'o', color='white', markeredgecolor='#1f78b4', markersize=8, markeredgewidth=2)
        else:
            if (G["StellarMass"][i] > Mass_cutoff).any(): # else find where galaxy is satellite; take into account mass cut
                plt.plot(np.log10(G['StellarMass'][i]*1e10/h), np.log10(np.sum(G['DiscHI'],axis=1)[i]*1e10/h), 
                         'o', color='#1f78b4', markersize=8)
        
ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
ax.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)
plt.plot([], [], 'o', color='#1f78b4', markersize=8, label='Satellite')
plt.plot([], [], 'o', color='white', markeredgecolor='#1f78b4', markersize=8, markeredgewidth=2, label='Central')

plt.legend(loc=4)
plt.show()


# # All central and satellite galaxies in groups of length 2+ and single galaxies

# In[41]:


#initiate figure and how large it will be
fig = plt.figure(figsize=(10,10))                                                               
ax = fig.add_subplot(1,1,1)

ax.scatter(np.log10(Mstellar_single_gal), np.log10(Mcoldgas_single_gal), marker='o', s=50, color='#fcc5c0', zorder=-3, alpha=0.4)

#put plot in row and columns:3by3 are 00, 01, 02, 10, 11, 12, 20, 21, 22 
row_count = 0
for size in trange(1, 9, 1): #I'm planning to have 1-9 sized groups

    Group_size = size+1 # select number of galaxies per group --- adding +1 because it is counting from 0.
    

    #plot centrals to be on each plot
  
        #initiate condition for plot
    for group in groups[Group_size]:
        if (G["StellarMass"][group] < Mass_cutoff).any(): #creates array of booleans; if theese are true hit continue
            continue
        
        for i in group: #find central galaxies and plot them as red
            if i in store_cen_indices:
            #print(i)
                ax.plot(np.log10(G['StellarMass'][i]*1e10/h), np.log10(np.sum(G['DiscHI'],axis=1)[i]*1e10/h), 
                     'o', color='white', markeredgecolor='#1f78b4', markersize=8, markeredgewidth=1)
            else:
                #if (G["StellarMass"][i] > Mass_cutoff).any(): # else find where galaxy is satellite; take into account mass cut
                ax.plot(np.log10(G['StellarMass'][i]*1e10/h), np.log10(np.sum(G['DiscHI'],axis=1)[i]*1e10/h), 
                             'o', color='#1f78b4', markersize=8)

    #add label for each sub-plot to know the group size and what is central/satellite
ax.plot([], [], 'o', color='#fcc5c0', label='Single galaxies')
ax.plot([], [], 'o', color='#1f78b4', markersize=8, label='Satellite')
ax.plot([], [], 'o', color='white', markeredgecolor='#1f78b4', markersize=8, markeredgewidth=2, label='Central')


ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
ax.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)

leg = ax.legend(loc=4,frameon=True,fancybox=True, framealpha=0.6, fontsize=16)

#add x and y axis labels only on the plot edges since there will be no space between panels

#tight layout and remove spacing between subplots
plt.tight_layout()
#plt.legend(loc=4)
plt.savefig('MHI_vs_Mstar_Groups.png')
plt.show()


# ### Now mark central and satellites in the N times N plot
# ###### Takes 00:00:4 to run

# In[42]:


#initiate figure and how large it will be
fig, ax = plt.subplots(nrows=3, ncols=3, sharex='col', sharey='row', figsize=(18, 15))

#put plot in row and columns:3by3 are 00, 01, 02, 10, 11, 12, 20, 21, 22 
row_count = 0
for size in trange(1, 9, 1): #I'm planning to have 1-9 sized groups
        
    if size % 3 == 0:
        row_count += 1
    this_ax = ax[row_count, size%3]
    
    Group_size = size+1 # select number of galaxies per group --- adding +1 because it is counting from 0.
    

    #plot centrals to be on each plot
    this_ax.plot(np.log10(Mstellar_single_gal), np.log10(Mcoldgas_single_gal), 'o', color='lightgrey',markersize=8)
  
        #initiate condition for plot
    for group in groups[Group_size]:
        if (G["StellarMass"][group] < Mass_cutoff).any(): #creates array of booleans; if theese are true hit continue
            continue
        
        for i in group: #find central galaxies and plot them as red
            if i in store_cen_indices:
            #print(i)
                this_ax.plot(np.log10(G['StellarMass'][i]*1e10/h), np.log10(np.sum(G['DiscHI'],axis=1)[i]*1e10/h), 
                     'o', color='white', markeredgecolor='#1f78b4', markersize=8, markeredgewidth=2)
            else:
                #if (G["StellarMass"][i] > Mass_cutoff).any(): # else find where galaxy is satellite; take into account mass cut
                this_ax.plot(np.log10(G['StellarMass'][i]*1e10/h), np.log10(np.sum(G['DiscHI'],axis=1)[i]*1e10/h), 
                             'o', color='#1f78b4', markersize=8)

    #add label for each sub-plot to know the group size and what is central/satellite
    this_ax.plot([], [], 'o', color='white', label='Groups of %.f' %Group_size)
    this_ax.plot([], [], 'o', color='lightgrey', label='Single galaxies')
    this_ax.plot([], [], 'o', color='#1f78b4', markersize=8, label='Satellite')
    this_ax.plot([], [], 'o', color='white', markeredgecolor='#1f78b4', markersize=8, markeredgewidth=2, label='Central')

    leg = this_ax.legend(loc=4,frameon=True,fancybox=True, framealpha=0.6, fontsize=16)

#add x and y axis labels only on the plot edges since there will be no space between panels
for row in range(3):
    this_ax = ax[row,0]
    this_ax.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)
for col in range(3):
    this_ax = ax[2,col]
    this_ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
#tight layout and remove spacing between subplots
plt.tight_layout()
#plt.legend(loc=4)
plt.subplots_adjust(wspace = 0.0, hspace = 0.0)
plt.savefig('MHI_vs_Mstar_Groups.png')
plt.show()


# ## Make upper plot muchfaster

# # Create cen & sat from groups dictionary

# In[24]:


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
            if (G['LenMax'][group] < 100).any():
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
                    


# In[25]:


#This is updated dictionary which one can parse to the plotting function:
updated_dict = create_cen_sat_from_groups_dict(groups, store_cen_indices) 


# ### Example of the usage of updated_dict -- taking "Groups"

# In[29]:


#Use dictionary and for each galaxy in "Groups" of 3 show central/satellite index
Grp_length = 69
for group_key in updated_dict[Grp_length]["Groups"].keys():
    central_idx = updated_dict[Grp_length]["Groups"][group_key]["Centrals"] #give indices
    central_mass = np.sum(G['DiscHI'],axis=1)[central_idx] #give masses of these galaxies
    
    satellite_inds = updated_dict[Grp_length]["Groups"][group_key]["Satellites"]
    sat_mass = np.sum(G['DiscHI'],axis=1)[satellite_inds]
    
    if (sat_mass > central_mass).any(): #check conditions
        print("Sat more massive than central")
    #print(sat_mass > central_mass)
    
    print("Central mass is {0} and sat masses are {1}".format(central_mass, sat_mass))
    #print(updated_dict[3]["Groups"][group_key]["Centrals"])

    
updated_dict[3]["Groups"]["Group_10"] #shows indices for 10th Group in Groups of length 3


# ### Check group sizes

# In[44]:


#for i in trange(1,100):
#    print(i, len(updated_dict[i]["Groups"].keys()))
#    
#print(updated_dict[20]['Groups'].keys())


# In[ ]:





# In[26]:


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
                         'o', color='white', markeredgecolor='#1f78b4', markersize=8, markeredgewidth=2)
        # Satellites
        this_ax.plot(np.log10(G['StellarMass'][sat_gals]*1e10/h), 
                         np.log10(np.sum(G['DiscHI'],axis=1)[sat_gals]*1e10/h), 
                         'o', color='#1f78b4', markersize=8)
    
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


# In[27]:


#Use line_profiler to test how fast is each line of the code

#put this line in front of funtion which you are calling:
#%lprun -f mhi_vs_ms_3x3 

mhi_vs_ms_3x3(updated_dict)


# # Make a two sided histogram

# In[33]:


def two_sided_histogram_group_and_single(x_cen, y_cen, x_sat, y_sat, x_single, y_single):
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
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))
    
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
    axScatter.set_ylim((5.8, 11.7))
    
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


    return plt.show()
    


# In[34]:


def two_sided_histogram_groups(x_cen, y_cen, x_sat, y_sat, color='#1f78b4'):
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
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))
    
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
    axScatter.set_ylim((5.8, 11.7))
    
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


    return plt.show()


# In[35]:


single = updated_dict[1]['Centrals']
central_2 = updated_dict[2]["Centrals"]
satellite_2 = updated_dict[2]["Satellites"]

x_cen = np.log10(G["StellarMass"][central_2]*1e10/h ) #gives array of central galaxy stellar masses 
y_cen = np.log10(np.sum(G['DiscHI'],axis=1)[central_2]*1e10/h+1) #BECAUSE log0 goes to -inf so I add 1! has to be checked/removed!
x_sat = np.log10(G["StellarMass"][satellite_2]*1e10/h) #gives array of satellite galaxy stellar masses 
y_sat = np.log10(np.sum(G['DiscHI'],axis=1)[satellite_2]*1e10/h+1)

x_single = np.log10(G["StellarMass"][single]*1e10/h) #gives array of satellite galaxy stellar masses 
y_single = np.log10(np.sum(G['DiscHI'],axis=1)[single]*1e10/h+1)


# In[36]:


two_sided_histogram_group_and_single(x_cen, y_cen, x_sat, y_sat, x_single, y_single)


# In[37]:


two_sided_histogram_groups(x_cen, y_cen, x_sat, y_sat)


# In[38]:


def hist_Mhi_vs_Mstar_with_singles(groups_dict):
    
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
            
        if size % 3 == 0:
            row_count += 1
        this_ax = ax[row_count, size%3]
        
        group_size = size+1 # Select number of galaxies per group --- adding +1 because it is counting from 0.
    
        central_gals = groups_dict[group_size]["Centrals"]
        sat_gals = groups_dict[group_size]["Satellites"]
        single_gals = groups_dict[1]['Centrals']
     
        # Do plotting of groups of N-length; it will be placed on each figure
        # Centrals and Satellites; x & y axis histogram
        # Using function two_sided_histogram to plot the histogram output so here I call the function and give (x, y), (x, y)
        two_sided_histogram_group_and_single(np.log10(G['StellarMass'][central_gals]*1e10/h),
                            np.log10(np.sum(G['DiscHI'],axis=1)[central_gals]*1e10/h+1), 
                            np.log10(G['StellarMass'][sat_gals]*1e10/h),
                            np.log10(np.sum(G['DiscHI'],axis=1)[sat_gals]*1e10/h+1),
                            np.log10(G['StellarMass'][single_gals]*1e10/h),
                            np.log10(np.sum(G['DiscHI'],axis=1)[single_gals]*1e10/h+1) ) 


# In[54]:


hist_Mhi_vs_Mstar_with_singles(updated_dict)


# In[35]:


def hist_Mhi_vs_Mstar(groups_dict):
    
    """
    N times the scatter plot with two datasets and their histograms on x and y axis.
    Two datasets are groups of lenght N++
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
    To use type: ``hist_Mhi_vs_Mstar(updated_dict) ``
                 Where updated_dict is the dictionary that is parsed to plotting function
                 ``updated_dict = create_cen_sat_from_groups_dict(groups, store_cen_indices)``
    
    """
       

                
                
    row_count = 0
    for size in trange(1, 9, 1): #I'm planning to have 1-9 sized groups
            
        if size % 3 == 0:
            row_count += 1
        this_ax = ax[row_count, size%3]
        
        group_size = size+1 # Select number of galaxies per group --- adding +1 because it is counting from 0.
    
        central_gals = groups_dict[group_size]["Centrals"]
        sat_gals = groups_dict[group_size]["Satellites"]
        single_gals = groups_dict[1]['Centrals']
     
        # Do plotting of groups of N-length; it will be placed on each figure
        # Centrals and Satellites; x & y axis histogram
        # Using function two_sided_histogram to plot the histogram output so here I call the function and give (x, y), (x, y)
        two_sided_histogram_groups(np.log10(G['StellarMass'][central_gals]*1e10/h),
                            np.log10(np.sum(G['DiscHI'],axis=1)[central_gals]*1e10/h+1), 
                            np.log10(G['StellarMass'][sat_gals]*1e10/h),
                            np.log10(np.sum(G['DiscHI'],axis=1)[sat_gals]*1e10/h+1) ) 


# In[56]:


hist_Mhi_vs_Mstar(updated_dict)


# # Properties of the group as the whole entity

# In[39]:


def two_sided_histogram(x_data, y_data, color='lightgrey', legend=legend, s=90):
    """
    Make a scatter plot with 1 dataset and then place histogram on both x and y axis for them. .
    Here I use logMhi vesus logMstar as the sum values for groups of N length.
    
    Parameters
    ==========
    x_data, y_data: x and y values of the first dataset. Floats. Can be list/array. 
                  In this case, I am using logMHI and logMstar of the sum a group.
    
    color='lightgrey': Default color.
                      This will change when I call a function that has defined its own colors.
    
    legend=legend: Default legend.
                   This will change when I call a function that has defined its own legends.
    
    s=90: Markersize. Default 90.
                This will change when I call a function with its defined markersizes.
   
    Returns
    =======
    A scatter plot with histogram on x and y axis for both datasets. 
    
    Usage
    =====
    To use type: ``two_sided_histogram(x_data, y_data)``
               I extracted satellite and central gal from dictionary using ``updated_dict`` and 
               group_size=2 for pairs -- can be any group_size number
               
                central_2 = updated_dict[2]["Centrals"]
                satellite_2 = updated_dict[2]["Satellites"]
                
                and then sum the x and y properties:
                
                x_sum = np.log10(G["StellarMass"][central_2]*1e10/h +(G["StellarMass"][satellite_2]*1e10/h) ) #gives array of the sum of the stellar masses 
                y_sum = np.log10(G["ColdGas"][central_2]*1e10/h + (G["ColdGas"][satellite_2]*1e10/h+1)) #gives array of the sum of the gas masses
    
                etc. just update central_2 with other keys to specific indices.
                
    """

    
    nullfmt = NullFormatter()         # no labels
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # start with a rectangular Figure
    plt.figure(1, figsize=(12, 12))
    
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    # the scatter plot:
    markersizes=90
    axScatter.scatter(x_data, y_data, color=color, edgecolor='k', s=s, linewidth=0.4, label=legend)
    
    axScatter.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
    axScatter.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)

    # now determine nice limits by hand:
    binwidth = 0.1
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(x))])
    lim = (int(xymax/binwidth) + 1) * binwidth
    
    axScatter.set_xlim((9, 11.8))
    axScatter.set_ylim((7.5, 11.5))
    
    bins = np.arange(7.5, lim + binwidth, binwidth)
    
    axHistx.hist(x_data, bins=bins, color=color,alpha=0.8, edgecolor='k', linewidth=1)
    axHisty.hist(y_data, bins=bins, orientation='horizontal', color=color, alpha=0.8, edgecolor='k', linewidth=1) 

    axHistx.set_ylabel(r'N$_{\textrm{groups}}$', fontsize=22)
    axHisty.set_xlabel(r'N$_{\textrm{groups}}$', fontsize=22)
    
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    
    
    legend = axScatter.legend(loc=3, fontsize=18)


    return #plt.show()


# ### All groups separatelly based on the length
# #### Group mass represented as sum of the galaxies which are in the group
# #### Takes long on big box

# In[38]:


#group_size = 2
#halo_2 = updated_dict[group_size]["All_groups"]
#
#x_values = []
#y_values = []
#
#for each_group in halo_2:
#   #print(each_group)
#    x_sum = np.log10( np.sum((G["StellarMass"][each_group]*1e10/h)) ) #gives array of the sum of the stellar masses 
#    y_sum = np.log10( np.sum((np.sum(G['DiscHI'],axis=1)[each_group]*1e10/h+1)) ) #gives array of the sum of the gas masses
#    x_values.append(x_sum)
#    y_values.append(y_sum)
#    
#    
#two_sided_histogram(x_values, y_values, legend='Group of %.f'%group_size)

    


# ### All groups together

# In[40]:


def hist_group_sum_Mhi_vs_Mstar(groups_dict, colors=['lightgrey','#feebe2','#fcc5c0','#fa9fb5','#f768a1','#dd3497','#ae017e','#7a0177'],
                               legends=['2', '3', '4', '5', '6', '7', '8', '9'], markersizes=[90,150,210,270,330,400,470,540,610]):
    
    """
    N times the scatter plot with dataset and their histograms on x and y axis.
    Datasets are groups of lenght N++
    Here I use logMhi vesus logMstar -- the sum values for the entire group
    
    
    Parameters
    ==========
    groups_dict: Dictionary. Keyed by the group size. Use: ``groups_dict[group_size]["Centrals"]``
                 Extracts groups with size N and the satellite & central galaxies
                 Created dictionary from (def create_cen_sat_from_groups_dict)
  
    colors=['r', 'b' ... ]: List of colors. To be replaced with default ones in two_sided_histogram.
                Here is used list of lenght 8 because I'm plotting N sized groups where N 2-9
    
    legends=['2', '3', '4', '5', '6', '7', '8', '9']: List of legends. To be places as a length of each group.
            Here is used list of lenght 8 because I'm plotting N sized groups where N 2-9
    
    markersizes=[90,150,210,270,330,400,470,540,610]: Array of integers. 
                Replaces the markersize for each iteration.
    
    Returns
    =======
    N times the scatter plot with dataset as a sum of the masses and their histograms on x and y axis.
    
    
    Usage
    =====
    To use type: ``hist_group_sum_Mhi_vs_Mstar(updated_dict) ``
                 Where updated_dict is the dictionary that is parsed to plotting function
                 ``updated_dict = create_cen_sat_from_groups_dict(groups, store_cen_indices)``
    
    """
       

    for size in trange(1, 9, 1): #I'm planning to have 1-9 sized groups

        group_size = size+1 # Select number of galaxies per group --- adding +1 because it is counting from 0.

        halo_groups = updated_dict[group_size]["All_groups"]
        
        x_values = []
        y_values = []

        for each_group in halo_groups:
        #print(each_group)
            x_sum = np.log10( np.sum((G["StellarMass"][each_group]*1e10/h)) ) #gives array of the sum of the stellar masses 
            y_sum = np.log10( np.sum((np.sum(G['DiscHI'],axis=1)[each_group]*1e10/h+1)) ) #gives array of the sum of the gas masses
            x_values.append(x_sum)
            y_values.append(y_sum)
        
        # Do plotting of groups of N-length; it will be placed on each figure
        # Centrals and Satellites; x & y axis histogram
        # Using function two_sided_histogram to plot the histogram output so here I call the function and give (x, y), (x, y)
        two_sided_histogram(x_values, y_values, colors[size-1], legends[size-1], markersizes[size-1]) # Call the function with specified colors to have different colors for each group.
                            
         


# In[60]:


hist_group_sum_Mhi_vs_Mstar(updated_dict)


# In[61]:


def histogram_wrapper(dataset_length):
    """
    """
    
    if dataset_length == 1:
        hist_group_sum_Mhi_vs_Mstar(updated_dict)
    elif dataset_length ==2:
        hist_Mhi_vs_Mstar(updated_dict)
    elif dataset_length == 3:
        hist_Mhi_vs_Mstar_with_singles(updated_dict)
    else:# Check for funky stuff and raise an error
        print("Wrong dataset length. Dataset length should be: 1, 2 or 3.")
        raise ValueError


# # Use dictionary based on "Groups" which also has info abour cen/sat
# ## Find where central galaxy is the most gas rich galaxy in each group

# In[41]:


#Use dictionary and for each galaxy in "Groups" of 3 show central/satellite index

richer_central_ind = []
poorer_central_ind = []

richer_sat_ind = []
poorer_sat_ind = []

Grp_length = 5

for group_key in tqdm(updated_dict[Grp_length]["Groups"].keys()):
    central_idx = updated_dict[Grp_length]["Groups"][group_key]["Centrals"] #give indices
    central_mass = np.sum(G['DiscHI'],axis=1)[central_idx] #give masses of these galaxies
    
    satellite_inds = updated_dict[Grp_length]["Groups"][group_key]["Satellites"]
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


# In[64]:


print(list(poorer_central_ind))

print(updated_dict[3]["Groups"]["Group_10"]) #shows indices for 10th Group in Groups of length 3

print(updated_dict[3]["Groups"][group_key]["Satellites"])

#print(G['ColdGas'][updated_dict[3]['Centrals']][:5])
#print(G['ColdGas'][updated_dict[3]['Satellites']][:5])
#print(G['ColdGas'][updated_dict[3]['All_galaxies']][:5])
#print(G['ColdGas'][updated_dict[3]['All_groups']][:5])


# In[42]:


#HI Masses, obtained from the indices: 
richer_central_hi_m = np.log10( (np.sum(G['DiscHI'],axis=1)[richer_central_ind] *1e10/h )+1)
poorer_central_hi_m = np.log10( (np.sum(G['DiscHI'],axis=1)[poorer_central_ind] *1e10/h )+1)

richer_sat_hi_m = np.log10( (np.sum(G['DiscHI'],axis=1)[richer_sat_ind] *1e10/h )+1)
poorer_sat_hi_m = np.log10( (np.sum(G['DiscHI'],axis=1)[poorer_sat_ind] *1e10/h )+1)

richer_central_s_m = np.log10( G['StellarMass'][richer_central_ind] *1e10/h +1)
poorer_central_s_m = np.log10( G['StellarMass'][poorer_central_ind] *1e10/h +1)

richer_sat_s_m = np.log10( G['StellarMass'][richer_sat_ind] *1e10/h +1)
poorer_sat_s_m = np.log10( G['StellarMass'][poorer_sat_ind] *1e10/h +1)


# In[43]:


b = richer_sat_hi_m.ravel() #RESHAPE 2D array into 1D
#print(b)


# In[40]:


fig = plt.figure(figsize=(10,10))                                                               
ax = fig.add_subplot(1,1,1)

binwidth=0.1
bins = np.arange(7.5, 11 + binwidth, binwidth)
    
#axHistx.hist(x_data, bins=bins, color=color,alpha=0.8, edgecolor='k', linewidth=1)
#axHisty.hist(y_data, bins=bins, orientation='horizontal', color=color, alpha=0.8, edgecolor='k', linewidth=1) 

plt.hist(richer_central_hi_m, bins=bins, color='#2c7fb8', edgecolor='k', linewidth=1, alpha=0.8, label='HI rich(er) central')
plt.hist(poorer_central_hi_m, bins=bins, color='#c7e9b4', edgecolor='k', linewidth=1, alpha=0.5, label='HI poor(er) central')

ax.set_ylabel(r'N$_{\mathrm{galaxies}}$', fontsize=25)
ax.set_xlabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)

plt.xlim(7,11)
plt.legend(loc=2)
plt.show()


# In[44]:


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
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # start with a rectangular Figure
    plt.figure(1, figsize=(12, 12))
    
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


    return plt.show()
    


# In[46]:


#Centrals more HI rich than the sum of their satellites
two_sided_histogram_rich(richer_central_s_m, richer_central_hi_m, poorer_central_s_m, poorer_central_hi_m) 


# In[45]:


def two_sided_histogram_rich_groups(x_cen_rich, y_cen_rich, x_cen_poor, y_cen_poor,
                                   x_sat_rich, y_sat_rich, x_sat_poor, y_sat_poor):
    """
    Make a scatter plot with 8 datasets and then place histogram on both x and y axis for them as rich/poor central groups. 
    Here I use logMhi vesus logMstar -- other data will need readjustment of the min/max.
    
    Parameters
    ==========
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
                                richer_sat_s_m.ravel(), richer_sat_hi_m.ravel(), poorer_sat_s_m.ravel(), poorer_sat_hi_m.ravel()) 
``
               I extracted satellite and central gal from dictionary using ``updated_dict`` and 
               group_size=2 for pairs -- can be any group_size number
                Indices:
                central_idx = updated_dict[3]["Groups"][group_key]["Centrals"]
                satellite_inds = updated_dict[3]["Groups"][group_key]["Satellites"]
                
                To this, condition is added which check the mass of the central vs. satellites.
                
                
    """

    
    nullfmt = NullFormatter()         # no labels
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # start with a rectangular Figure
    plt.figure(1, figsize=(12, 12))
    
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

    axScatter.legend(loc=3, fontsize=13)


    return plt.show()
    


# In[46]:


def two_sided_histogram_difference(x_cen_rich, y_cen_rich, x_cen_poor, y_cen_poor,
                                   x_sat_rich, y_sat_rich, x_sat_poor, y_sat_poor):
    """
    Make a scatter plot with 8 datasets and then place histogram on both x and y axis for them
    Here plot groups with small difference in stellar mass but large in HI mass between cen/sat. 
    Here I use logMhi vesus logMstar -- other data will need readjustment of the min/max.
    
    Parameters
    ==========
    x_cen_rich, y_cen_rich: x and y values of the first dataset - here are centrals with small Mstar diff and large
                            Mhi difference. Floats. Should be list/array; use .ravel() if needed
                            to reduce the dimensionality.  
                            In this case, I am using logMHI and logMstar of central galaxies.
    
    x_cen_poor, y_cen_poor: x and y values of the second dataset - other/normal groups. 
                            Should be list/array; use .ravel() if needed to reduce the dimensionality.
                            In this case, I am using logMHI and logMstar of satellite galaxies.                        
        
    x_sat_rich, y_sat_rich, x_sat_poor, y_sat_poor:
                            Same as above; just uses satellites around their respective central galaxy.
    
    Returns
    =======
    A scatter plot with histogram on x and y axis for both datasets. 
    
    Usage
    =====
    To use type: ``two_sided_histogram_rich_groups(richer_central_s_m.ravel(), richer_central_hi_m.ravel(), poorer_central_s_m.ravel(), poorer_central_hi_m.ravel(),
                                richer_sat_s_m.ravel(), richer_sat_hi_m.ravel(), poorer_sat_s_m.ravel(), poorer_sat_hi_m.ravel()) 
``
               I extracted satellite and central gal from dictionary using ``updated_dict`` and 
               group_size=2 for pairs -- can be any group_size number
                Indices:
                central_idx = updated_dict[3]["Groups"][group_key]["Centrals"]
                satellite_inds = updated_dict[3]["Groups"][group_key]["Satellites"]
                
                To this, condition is added which check the mass of the central vs. satellites.
                
                
    """

    
    nullfmt = NullFormatter()         # no labels
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # start with a rectangular Figure
    plt.figure(1, figsize=(12, 12))
    
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    # the scatter plot:

    axScatter.scatter(x_cen_poor, y_cen_poor, color='white', edgecolor='lightgrey', s=80, linewidth=2, label='Other central', alpha=0.8)
    axScatter.scatter(x_sat_poor, y_sat_poor, color='lightgrey', edgecolor='lightgrey', s=80, linewidth=2, label='Other central\'s satellite', alpha=0.8)

    axScatter.scatter(x_cen_rich, y_cen_rich, color='white', edgecolor='#2c7fb8', s=80, linewidth=2, label='Large diff central')
    axScatter.scatter(x_sat_rich, y_sat_rich, color='#2c7fb8', edgecolor='#2c7fb8', s=80, linewidth=2, label='Large diff central\'s satellite')

    
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

    axScatter.legend(loc=3, fontsize=13)


    return plt.show()
    


# In[47]:


def two_sided_histogram_difference_groups(x_cen_rich, y_cen_rich, x_cen_poor, y_cen_poor,
                                   x_sat_rich, y_sat_rich, x_sat_poor, y_sat_poor,
                                  x_group, y_group):
    """
    Make a scatter plot with 8 datasets and then place histogram on both x and y axis for them
    Here plot groups with small difference in stellar mass but large in HI mass between cen/sat. 
    Here I use logMhi vesus logMstar -- other data will need readjustment of the min/max.
    
    Parameters
    ==========
    x_cen_rich, y_cen_rich: x and y values of the first dataset - here are centrals with small Mstar diff and large
                            Mhi difference. Floats. Should be list/array; use .ravel() if needed
                            to reduce the dimensionality.  
                            In this case, I am using logMHI and logMstar of central galaxies.
    
    x_cen_poor, y_cen_poor: x and y values of the second dataset - other/normal groups. 
                            Should be list/array; use .ravel() if needed to reduce the dimensionality.
                            In this case, I am using logMHI and logMstar of satellite galaxies.                        
        
    x_sat_rich, y_sat_rich, x_sat_poor, y_sat_poor:
                            Same as above; just uses satellites around their respective central galaxy.
    
    Returns
    =======
    A scatter plot with histogram on x and y axis for both datasets. 
    
    Usage
    =====
    To use type: ``two_sided_histogram_rich_groups(richer_central_s_m.ravel(), richer_central_hi_m.ravel(), poorer_central_s_m.ravel(), poorer_central_hi_m.ravel(),
                                richer_sat_s_m.ravel(), richer_sat_hi_m.ravel(), poorer_sat_s_m.ravel(), poorer_sat_hi_m.ravel()) 
``
               I extracted satellite and central gal from dictionary using ``updated_dict`` and 
               group_size=2 for pairs -- can be any group_size number
                Indices:
                central_idx = updated_dict[3]["Groups"][group_key]["Centrals"]
                satellite_inds = updated_dict[3]["Groups"][group_key]["Satellites"]
                
                To this, condition is added which check the mass of the central vs. satellites.
                
                
    """

    
    nullfmt = NullFormatter()         # no labels
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # start with a rectangular Figure
    plt.figure(1, figsize=(12, 12))
    
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    # the scatter plot:
        
    axScatter.scatter(x_cen_poor, y_cen_poor, color='white', edgecolor='lightgrey', s=80, linewidth=2, label='Other central', alpha=0.6)
    axScatter.scatter(x_sat_poor, y_sat_poor, color='lightgrey', edgecolor='lightgrey', s=80, linewidth=2, label='Other central\'s satellite', alpha=0.6)
   
    
    axScatter.scatter(x_cen_rich, y_cen_rich, color='white', edgecolor='#2c7fb8', s=80, linewidth=2, label='Large diff central')
    axScatter.scatter(x_sat_rich, y_sat_rich, color='#2c7fb8', edgecolor='#2c7fb8', s=80, linewidth=2, label='Large diff central\'s satellite')

    #MARK THE RICH ONES
    markers = itertools.cycle(('o', 'o', 'o', 's', 's', 's','D', 'D', 'D','p', 'p', 'p', 'd', 'd', 'd',
                             'P', 'P', 'P', '*', '*', '*','<', '<', '<','v', 'v', 'v', '^', '^', '^',
                              'x', 'x', 'x')) 
 
    #col = matplotlib.cm.viridis(np.linspace(0, 1, len(x_group)))
    #colors = itertools.cycle(col)
    
    for i in range(0,len(x_group),1):
        axScatter.scatter(x_group[i],y_group[i], marker = next(markers), color = '',edgecolor='k', s=400)
        
    #marker = itertools.cycle(('o', 's', 'p', 'P', '*', 'h', 'D', 'd'))
    #axScatter.scatter(x_group, y_group, color='#fa9fb5', marker=next(marker), s=250, label='Groups', alpha=0.8)
    #print(len(x_group)) == 15 
    #print(len(x_group[1]))
    
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

    axScatter.legend(loc=3, fontsize=13)


    return plt.show()
    


# ## Groups Length-3

# In[50]:


# Central gals which are more HI rich than the sum of their satellites --- also marks position of their satellites
# And vice versa
two_sided_histogram_rich_groups(richer_central_s_m.ravel(), richer_central_hi_m.ravel(), poorer_central_s_m.ravel(), poorer_central_hi_m.ravel(),
                                richer_sat_s_m.ravel(), richer_sat_hi_m.ravel(), poorer_sat_s_m.ravel(), poorer_sat_hi_m.ravel()) 


# # Extract rich(er) and poor(er) from all size-d groups

# In[48]:


#Use dictionary and for each galaxy in "Groups" of 3 show central/satellite index

all_richer_central_ind = []
all_poorer_central_ind = []

all_richer_sat_ind = [] #satellites that belong to richer central
all_poorer_sat_ind = [] #satellites that belong to poorer central

for i in trange(3,10,1): #starting grom galaxy pairs
    for group_key in updated_dict[i]["Groups"].keys():
        central_idx = updated_dict[i]["Groups"][group_key]["Centrals"] #give indices
        central_mass = np.sum(G['DiscHI'],axis=1)[central_idx] #give masses of these galaxies
    
        satellite_inds = updated_dict[i]["Groups"][group_key]["Satellites"]
        sat_mass = np.sum(G['DiscHI'],axis=1)[satellite_inds]
    
        if (central_mass > sat_mass).all(): #check conditions
            print("Central is the most HI massive")
            all_richer_central_ind.append(central_idx) 
            all_richer_sat_ind.extend(satellite_inds) #it complains and complicates when i want to do .append
        else:
            print("Satellite is more HI massive thatn central")
            all_poorer_central_ind.append(central_idx)
            all_poorer_sat_ind.extend(satellite_inds)
        #print(sat_mass > central_mass)
    
        print("Central mass is {0} and sat masses are {1}".format(central_mass, sat_mass))
        print("")


# In[75]:


#HI Masses, obtained from the indices: 
all_richer_central_hi_m = np.log10( (np.sum(G['DiscHI'],axis=1)[all_richer_central_ind] *1e10/h )+1)
all_poorer_central_hi_m = np.log10( (np.sum(G['DiscHI'],axis=1)[all_poorer_central_ind] *1e10/h )+1)
all_richer_central_s_m = np.log10( G['StellarMass'][all_richer_central_ind] *1e10/h +1)
all_poorer_central_s_m = np.log10( G['StellarMass'][all_poorer_central_ind] *1e10/h +1)

#have to store satellites like this; otherwise it complains
all_richer_sat_hi_m = []
all_richer_sat_s_m = []

for item in all_richer_sat_ind:
    a_richer_sat_hi_m = np.log10( (np.sum(G['DiscHI'],axis=1)[item] *1e10/h )+1)
    a_richer_sat_s_m = np.log10( G['StellarMass'][item] *1e10/h +1)
    all_richer_sat_hi_m.append(a_richer_sat_hi_m)
    all_richer_sat_s_m.append(a_richer_sat_s_m)

all_poorer_sat_hi_m = []
all_poorer_sat_s_m = []
for item in all_poorer_sat_ind:    
    a_poorer_sat_hi_m = np.log10( (np.sum(G['DiscHI'],axis=1)[item] *1e10/h )+1)
    a_poorer_sat_s_m = np.log10( G['StellarMass'][item] *1e10/h +1)
    all_poorer_sat_hi_m.append(a_poorer_sat_hi_m)
    all_poorer_sat_s_m.append(a_poorer_sat_s_m)


# In[76]:


two_sided_histogram_rich(all_richer_central_s_m, all_richer_central_hi_m, 
                         all_poorer_central_s_m, all_poorer_central_hi_m) 


# In[77]:


two_sided_histogram_rich_groups(all_richer_central_s_m.ravel(), all_richer_central_hi_m.ravel(), 
                                all_poorer_central_s_m.ravel(), all_poorer_central_hi_m.ravel(),
                                all_richer_sat_s_m, all_richer_sat_hi_m, 
                                all_poorer_sat_s_m, all_poorer_sat_hi_m) 


# # See the distribution of groups where central galaxy is X% HI richer than the sum of the satellites around that central
# ### Percent = M_HI_central/ (Sum[ M_HI_satellites ])

# In[49]:


per_cent_with_pairs =[]


# In[50]:


#Use dictionary and for each galaxy in "Groups" of 3 show central/satellite index

perc_richer_central_ind = [] #storring where central/satellite is above/below some percentage
perc_poorer_central_ind = []

perc_richer_sat_ind = [] #satellites that belong to richer central
perc_poorer_sat_ind = [] #satellites that belong to poorer central

per_cent = [] # How much of HI mass is in central galaxy (in %)

central_hi_mass_per_cent = []
central_st_mass_per_cent = []
group_size_per_cent = []

BTT_central = []
BTT_satellite = []

group_HI_mass = []
group_St_mass = []


# In[51]:


#BTT_central[0]


# In[52]:


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


# In[53]:


# Compute central galaxies
central_mass = np.sum(G['DiscHI'],axis=1)[c_ind]*1e10/h
central_st_mass = G['StellarMass'][c_ind]*1e10/h


# In[54]:


# Compute group properties (~4min)
g_m = []
g_st = []
for i in tqdm(g_ind):
    g_mass = np.sum(G['DiscHI'],axis=1)[i]*1e10/h
    g_st_mass = G['StellarMass'][i]*1e10/h
    g_m.append(np.sum(g_mass))
    g_st.append(np.sum(g_st_mass))
    


# In[55]:


# Compute percentage
percentage = (central_mass.ravel()/g_m)*100

#check output
#print(central_mass[0:5].ravel())
#print(g_m[0:5])
#print(percentage)


# In[56]:



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


#plt.savefig('Groups_HIinCentral.png')


# In[57]:


#Bulge-to-total ratio
#BTT_sat = (G['InstabilityBulgeMass'][s_ind] + G['MergerBulgeMass'][s_ind]) / ( G['StellarMass'][s_ind] )
BTT_cen = (G['InstabilityBulgeMass'][c_ind] + G['MergerBulgeMass'][c_ind]) / ( G['StellarMass'][c_ind] )
Lenght_cen = G['Len'][c_ind]
Mvir_cen = np.log10( (G['Mvir'][c_ind])*1e10/h)
Rvir_cen = G['Rvir'][c_ind]


# ## Distribution below is for groups N=+3

# In[59]:


fig = plt.figure(figsize=(14,7))                                                               
ax = fig.add_subplot(1,2,1)

binwidth=5
bins = np.arange(0, 100 + binwidth, binwidth)
    

per_cent = np.nan_to_num(per_cent)
per_cent_with_pairs = np.nan_to_num(per_cent_with_pairs)

plt.hist(per_cent_with_pairs, bins=bins, color='#80cdc1', edgecolor='k', linewidth=1, alpha=0.8, label='N$\geq$2')
plt.hist(per_cent, bins=bins, color='#dfc27d', edgecolor='k', linewidth=1, alpha=0.8, label='N$\geq$3')

ax.set_ylabel(r'N$_{\mathrm{centrals}}$', fontsize=25)
ax.set_xlabel(r'\% of the M$_{\mathrm{HI}}$ in central',fontsize=25)

plt.xlim(-5,105)
plt.legend(loc=1)

ax2 = fig.add_subplot(1,2,2)

    
ax2.hist(per_cent_with_pairs, density=True, bins=bins, color='#80cdc1', edgecolor='k', linewidth=1, alpha=0.8, label='N$\geq$2')
ax2.hist(per_cent, density=True, bins=bins, color='#dfc27d', edgecolor='k', linewidth=1, alpha=0.8, label='N$\geq$3')

ax2.set_xlabel(r'\% of the M$_{\mathrm{HI}}$ in central',fontsize=25)
ax2.set_ylabel(r'Density$_{\mathrm{centrals}}$', fontsize=25)
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()

plt.xlim(-5,105)
plt.legend(loc=1)

plt.show()


# In[58]:


grp_length = []
for i in g_ind:
    size = len(i)
    grp_length.append(size)


# In[59]:


df_percent = pd.DataFrame({'GroupStellarMass'   : g_st,
                           'GroupHIMass'        : g_m,
                           'Percent'            : percentage,
                           'GroupSize'          : grp_length,
                           'BTT_central'        : BTT_cen.ravel().tolist(),# })
                           'Len_cen'            : Lenght_cen.ravel(),
                           'Mvir_cen'           : Mvir_cen.ravel(),
                           'Rvir_cen'           : Rvir_cen.ravel() })
#print(df_percent)


# In[60]:


print(df_percent['BTT_central'][0:4])


# In[61]:



fig = plt.figure(figsize=(13,10))                                                               
ax = fig.add_subplot(1,1,1)

cm = plt.cm.get_cmap('YlGnBu')
#Numpy doesn't know what you might have stored in a object array, 
df_percent['GroupStellarMass'] = df_percent['GroupStellarMass'].astype(np.float64) 
df_percent['GroupHIMass'] = df_percent['GroupHIMass'].astype(np.float64)

#BE CAREFUL HERE! Constructed sizes are OK for current groups; will need change
#s = 80*df_percent['Group_size']

sizes = [100,300,500,700,900]
im = plt.scatter(np.log10(df_percent['GroupStellarMass']), np.log10(df_percent['GroupHIMass']), s=100,
                        c=df_percent['Percent'], edgecolor='k', cmap=cm, label=r'Groups ($N\geq3$)')

fig.colorbar(im, ax=ax, orientation='vertical', label=r'\% of the M$_{\mathrm{HI}}$ in central',
            pad=0.01)

ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
ax.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)
ax.legend(loc=2)
leg = ax.get_legend()
leg.legendHandles[0].set_color('k')

#l1 = plt.scatter([],[], s=100, color='white',edgecolors='k')
#l2 = plt.scatter([],[], s=300, color='white',edgecolors='k')
#l3 = plt.scatter([],[], s=500, color='white',edgecolors='k')
#l4 = plt.scatter([],[], s=700, color='white',edgecolors='k')
#l5 = plt.scatter([],[], s=900, color='white',edgecolors='k')
#
#labels = ["3", "4", "5", "6", '7']
#
#leg = plt.legend([l1, l2, l3, l4, l5], labels, ncol=1, loc=3, framealpha=0.5)
#leg.set_title('Group size')                   
                   
#plt.ylim(6,11)
plt.savefig('Groups_HIinCentral.png')


# In[62]:


HI_per_cent = 50

fig = plt.figure(figsize=(13,10))                                                               
ax = fig.add_subplot(1,1,1)

cm = plt.cm.get_cmap('YlGnBu')
#Numpy doesn't know what you might have stored in a object array, 
df_percent['GroupStellarMass'] = df_percent['GroupStellarMass'].astype(np.float64) 
df_percent['GroupHIMass'] = df_percent['GroupHIMass'].astype(np.float64)

#BE CAREFUL HERE! Constructed sizes are OK for current groups; will need change
#s = 80*df_percent['Group_size']

sizes = [100,300,500,700,900]
im = plt.scatter(np.log10(df_percent['GroupStellarMass'][df_percent['Percent']<HI_per_cent]), np.log10(df_percent['GroupHIMass'][df_percent['Percent']<HI_per_cent]), s=100,
                        c=df_percent['Percent'][df_percent['Percent']<HI_per_cent], edgecolor='k', cmap=cm, label=r'Less than {0}\% of HI is in Central'.format(HI_per_cent))

fig.colorbar(im, ax=ax, orientation='vertical', label=r'\% of the M$_{\mathrm{HI}}$ in central',
            pad=0.01)

ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
ax.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)
ax.legend(loc=2)
leg = ax.get_legend()
leg.legendHandles[0].set_color('k')

#l1 = plt.scatter([],[], s=100, color='white',edgecolors='k')
#l2 = plt.scatter([],[], s=300, color='white',edgecolors='k')
#l3 = plt.scatter([],[], s=500, color='white',edgecolors='k')
#l4 = plt.scatter([],[], s=700, color='white',edgecolors='k')
#l5 = plt.scatter([],[], s=900, color='white',edgecolors='k')
#
#labels = ["3", "4", "5", "6", '7']
#
#leg = plt.legend([l1, l2, l3, l4, l5], labels, ncol=1, loc=3, framealpha=0.5)
#leg.set_title('Group size')                   
                   
#plt.ylim(6,11)
plt.savefig('Groups_HIinCentral.png')


# In[63]:



fig = plt.figure(figsize=(13,10))                                                               
ax = fig.add_subplot(1,1,1)

cm = plt.cm.get_cmap('YlGnBu')
#Numpy doesn't know what you might have stored in a object array, 
df_percent['GroupStellarMass'] = df_percent['GroupStellarMass'].astype(np.float64) 
df_percent['GroupHIMass'] = df_percent['GroupHIMass'].astype(np.float64)

#BE CAREFUL HERE! Constructed sizes are OK for current groups; will need change
#s = 80*df_percent['Group_size']

sizes = [100,300,500,700,900]
im = plt.scatter(np.log10(df_percent['GroupStellarMass'][df_percent['Percent']>HI_per_cent]), np.log10(df_percent['GroupHIMass'][df_percent['Percent']>HI_per_cent]), s=100,
                        c=df_percent['Percent'][df_percent['Percent']>HI_per_cent], edgecolor='k', cmap=cm, label=r'More than {0}\% of HI is in Central'.format(HI_per_cent))

fig.colorbar(im, ax=ax, orientation='vertical', label=r'\% of the M$_{\mathrm{HI}}$ in central',
            pad=0.01)

ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
ax.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)
ax.legend(loc=2)
leg = ax.get_legend()
leg.legendHandles[0].set_color('k')

#l1 = plt.scatter([],[], s=100, color='white',edgecolors='k')
#l2 = plt.scatter([],[], s=300, color='white',edgecolors='k')
#l3 = plt.scatter([],[], s=500, color='white',edgecolors='k')
#l4 = plt.scatter([],[], s=700, color='white',edgecolors='k')
#l5 = plt.scatter([],[], s=900, color='white',edgecolors='k')
#
#labels = ["3", "4", "5", "6", '7']
#
#leg = plt.legend([l1, l2, l3, l4, l5], labels, ncol=1, loc=3, framealpha=0.5)
#leg.set_title('Group size')                   
                   
#plt.ylim(6,11)
plt.savefig('Groups_HIinCentral.png')


# In[64]:



fig = plt.figure(figsize=(13,10))                                                               
ax = fig.add_subplot(1,1,1)

cm = plt.cm.get_cmap('YlGnBu')
#Numpy doesn't know what you might have stored in a object array, 
df_percent['GroupStellarMass'] = df_percent['GroupStellarMass'].astype(np.float64) 
df_percent['GroupHIMass'] = df_percent['GroupHIMass'].astype(np.float64)

#BE CAREFUL HERE! Constructed sizes are OK for current groups; will need change
#s = 80*df_percent['Group_size']

sizes = [100,300,500,700,900]
im = plt.scatter(np.log10(df_percent['GroupStellarMass']), df_percent['Percent'], s=300,
                        color='lightgrey', edgecolor='k', label=r'Groups ($N\geq3$) properties')

#fig.colorbar(im, ax=ax, orientation='vertical', label=r'\% of the M$_{\mathrm{HI}}$ in central',
#            pad=0.01)

ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
ax.set_ylabel(r'\% of the M$_{\mathrm{HI}}$ in central',fontsize=25)
ax.legend(loc=2)
leg = ax.get_legend()
leg.legendHandles[0].set_color('lightgrey')

#l1 = plt.scatter([],[], s=100, color='white',edgecolors='k')
#l2 = plt.scatter([],[], s=300, color='white',edgecolors='k')
#l3 = plt.scatter([],[], s=500, color='white',edgecolors='k')
#l4 = plt.scatter([],[], s=700, color='white',edgecolors='k')
#l5 = plt.scatter([],[], s=900, color='white',edgecolors='k')
#
#labels = ["3", "4", "5", "6", '7']
#
#leg = plt.legend([l1, l2, l3, l4, l5], labels, ncol=1, loc=3, framealpha=0.5)
#leg.set_title('Group size')                   
                   
#plt.ylim(6,11)
#plt.savefig('Groups_HIinCentral.png')


# # Fixed N-sized groups

# In[65]:


Nsize_group = 3

fig = plt.figure(figsize=(13,10))                                                               
ax = fig.add_subplot(1,1,1)

cm = plt.cm.get_cmap('YlGnBu')
#Numpy doesn't know what you might have stored in a object array, 
#df_percent['GroupStellarMass'] = df_percent['GroupStellarMass'].astype(np.float64) 
#df_percent['GroupHIMass'] = df_percent['GroupHIMass'].astype(np.float64)

#BE CAREFUL HERE! Constructed sizes are OK for current groups; will need change
#s = 80*df_percent['Group_size']

sizes = [100,300,500,700,900]
im = plt.scatter(np.log10(df_percent['GroupStellarMass'][df_percent['GroupSize']==Nsize_group]), np.log10(df_percent['GroupHIMass'][df_percent['GroupSize']==Nsize_group]), s=100,
                        c=df_percent['Percent'][df_percent['GroupSize']==Nsize_group], edgecolor='k', cmap=cm, label=r'Groups with {0} members'.format(Nsize_group))

fig.colorbar(im, ax=ax, orientation='vertical', label=r'\% of the M$_{\mathrm{HI}}$ in central',
            pad=0.01)

ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
ax.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)
ax.legend(loc=2)
leg = ax.get_legend()
leg.legendHandles[0].set_color('k')

#l1 = plt.scatter([],[], s=100, color='white',edgecolors='k')
#l2 = plt.scatter([],[], s=300, color='white',edgecolors='k')
#l3 = plt.scatter([],[], s=500, color='white',edgecolors='k')
#l4 = plt.scatter([],[], s=700, color='white',edgecolors='k')
#l5 = plt.scatter([],[], s=900, color='white',edgecolors='k')
#
#labels = ["3", "4", "5", "6", '7']
#
#leg = plt.legend([l1, l2, l3, l4, l5], labels, ncol=1, loc=3, framealpha=0.5)
#leg.set_title('Group size')                   
                   
#plt.ylim(6,11)
plt.savefig('Groups_HIinCentral.png')


# In[66]:


import seaborn as sns


# In[67]:



#import matplotlib.font_manager
#matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

#matplotlib.rcParams.update({'font.size': fsize, 'xtick.major.size': 10, 'ytick.major.size': 10, 'xtick.major.width': 1, 'ytick.major.width': 1, 'ytick.minor.size': 5, 'xtick.minor.size': 5, 'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.linewidth': 1, 'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Times New Roman', 'legend.numpoints': 1, 'legend.columnspacing': 1, 'legend.fontsize': fsize-4, 'xtick.top': True, 'ytick.right': True})


# In[68]:


High_percentage = 80
Low_percentage = 50



fig = plt.figure(figsize=(10,10))                                                               
ax = fig.add_subplot(1,1,1)

#sns.set_style("white")
#sns.set(font_scale=2)
sns.set_style("ticks", {"xtick.major.size":10, "ytick.major.size":10,
						"xtick.minor.size":6,"ytick.minor.size":6})
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
#matplotlib.rcParams.update({'font.size': fsize, 'xtick.major.size': 10, 'ytick.major.size': 10, 'xtick.major.width': 1, 'ytick.major.width': 1, 'ytick.minor.size': 5, 'xtick.minor.size': 5, 'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.linewidth': 1, 'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Times New Roman', 'legend.numpoints': 1, 'legend.columnspacing': 1, 'legend.fontsize': fsize-4, 'xtick.top': True, 'ytick.right': True})


#sns.set_style("white")
sns.kdeplot(np.log10(df_percent['GroupStellarMass'][df_percent['Percent'] >= High_percentage]), 
            np.log10(df_percent['GroupHIMass'][df_percent['Percent'] >= High_percentage]), cmap='Blues', label= 'Over {0}\% of HI is in Central'.format(High_percentage)) #shade=True, shade_lowest=False, 


#sns.kdeplot(np.log10(df_percent['GroupStellarMass'][(df_percent['Percent'] >= 50) & ((df_percent['Percent'] < 80)) ]), 
#            np.log10(df_percent['GroupHIMass'][(df_percent['Percent'] >= 50) & ((df_percent['Percent'] < 80))]), cmap='Greens', label='Rich')

sns.kdeplot(np.log10(df_percent['GroupStellarMass'][df_percent['Percent'] < Low_percentage]), 
            np.log10(df_percent['GroupHIMass'][df_percent['Percent'] < Low_percentage]), cmap='RdPu', label= 'Less than {0}\% of HI is in Central'.format(Low_percentage))


#sns.kdeplot(np.log10(df_percent['StellarM'][df_percent['GroupSize']==Nsize_group]), 
#            np.log10(df_percent['HImass'][df_percent['GroupSize']==Nsize_group]), 
#            cmap="Blues", shade=True, shade_lowest=True, )

ax.set_xlabel(r'log M$_{\star\textrm{group}}$ [M$_{\odot}$]', fontsize=25)
ax.set_ylabel(r'log M$_{\textrm{HI}\textrm{group}}$ [M$_{\odot}$]',fontsize=25)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')

plt.legend(loc=3)
leg = ax.get_legend()
leg.legendHandles[0].set_color('blue')
leg.legendHandles[1].set_color('plum')
#leg.legendHandles[2].set_color('plum')
#plt.ylim(9.1,11)
plt.show()


# In[69]:


High_percentage = 50
Low_percentage = 50
Nsize_group = 3


fig = plt.figure(figsize=(10,10))                                                               
ax = fig.add_subplot(1,1,1)

#sns.set_style("white")
#sns.set(font_scale=2)
sns.set_style("ticks", {"xtick.major.size":10, "ytick.major.size":10,
						"xtick.minor.size":6,"ytick.minor.size":6})
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
#matplotlib.rcParams.update({'font.size': fsize, 'xtick.major.size': 10, 'ytick.major.size': 10, 'xtick.major.width': 1, 'ytick.major.width': 1, 'ytick.minor.size': 5, 'xtick.minor.size': 5, 'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.linewidth': 1, 'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Times New Roman', 'legend.numpoints': 1, 'legend.columnspacing': 1, 'legend.fontsize': fsize-4, 'xtick.top': True, 'ytick.right': True})


#sns.set_style("white")
#sns.kdeplot(np.log10(df_percent['GroupStellarMass'][ (df_percent['Percent'] >= High_percentage) & (df_percent['GroupSize']==5) ]), 
#            np.log10(df_percent['GroupHIMass'][ (df_percent['Percent'] >= High_percentage) & (df_percent['GroupSize']==5) ]), cmap='Purples', 
#            label= 'Over {0}\% of HI is in Central (Group size = {1})'.format(High_percentage, Nsize_group)) #shade=True, shade_lowest=False, 

#sns.kdeplot(np.log10(df_percent['GroupStellarMass'][ (df_percent['Percent'] >= High_percentage) & (df_percent['GroupSize']==3) ]), 
#            np.log10(df_percent['GroupHIMass'][ (df_percent['Percent'] >= High_percentage) & (df_percent['GroupSize']==3) ]), cmap='Blues', 
#            label= 'Over {0}\% of HI is in Central (Group size = {1})'.format(High_percentage, Nsize_group)) #shade=True, shade_lowest=False, 

sns.kdeplot(np.log10(df_percent['GroupStellarMass'][ (df_percent['Percent'] >= High_percentage) & (df_percent['GroupSize']==Nsize_group) ]), 
            np.log10(df_percent['GroupHIMass'][ (df_percent['Percent'] >= High_percentage) & (df_percent['GroupSize']==Nsize_group) ]), cmap='Blues', 
            label= 'Over {0}\% of HI is in Central (Group size = {1})'.format(High_percentage, Nsize_group)) #shade=True, shade_lowest=False, 


sns.kdeplot(np.log10(df_percent['GroupStellarMass'][ (df_percent['Percent'] < Low_percentage) & (df_percent['GroupSize']==Nsize_group) ]), 
            np.log10(df_percent['GroupHIMass'][ (df_percent['Percent'] < Low_percentage) & (df_percent['GroupSize']==Nsize_group) ]), cmap='RdPu', 
            label= 'Less than {0}\% of HI is in Central (Group size = {1})'.format(Low_percentage, Nsize_group))



ax.set_xlabel(r'log M$_{\star\textrm{group}}$ [M$_{\odot}$]', fontsize=25)
ax.set_ylabel(r'log M$_{\textrm{HI}\textrm{group}}$ [M$_{\odot}$]',fontsize=25)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')

plt.legend(loc=3)
leg = ax.get_legend()
leg.legendHandles[0].set_color('blue')
leg.legendHandles[1].set_color('plum')
#leg.legendHandles[2].set_color('plum')
#plt.ylim(9.1,11)
plt.show()


# In[70]:


matplotlib.__version__ #has to be 3.0.3; For sure is not working with 2.2.2
#import numpy as np
import pylab
from scipy.stats import skewnorm
from numpy.random import normal, multivariate_normal
from chainconsumer import ChainConsumer

#%matplotlib inline


# In[71]:


High_percentage = 60
Low_percentage = 50
Nsize_group = 3

x, y = np.log10(df_percent['GroupStellarMass'][df_percent['Percent'] >= High_percentage]), np.log10(df_percent['GroupHIMass'][df_percent['Percent'] >= High_percentage])

parameters = ["$\mathrm{log}$ $\mathrm{M}_{\star}$", "$\mathrm{log}$ $\mathrm{M}_{\mathrm{HI}}$"]
c = ChainConsumer().add_chain([x, y], parameters=parameters, name='Over {0}\% of HI is in Central'.format(High_percentage))

#properties at: https://samreay.github.io/ChainConsumer/chain_api.html
# The default case, so you don't need to specify: sigma2d gives  levels for 1D Gaussians - ie confidence levels of 68% and 95%.
#if sigma2d True : onfidence levels for 2D Gaussians, where 1 and 2 σ represents 39% and 86% confidence levels respectively

c.configure(statistics='max', flip=False, sigma2d=False, diagonal_tick_labels=False,  linewidths=2, linestyles=['-'],
            label_font_size=20, tick_font_size=15, contour_labels='sigma', contour_label_font_size=20, shade_alpha=1, colors='#c994c7', spacing=0)   # max_ticks=4 ; smooth=0

fig = c.plotter.plot()
fig.set_size_inches(5.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
plt.show()


# In[72]:


High_percentage = 60
Low_percentage = 50
Nsize_group = 3

x1, y1 = np.log10(df_percent['GroupStellarMass'][df_percent['Percent'] >= High_percentage]), np.log10(df_percent['GroupHIMass'][df_percent['Percent'] >= High_percentage])
x2, y2 = np.log10(df_percent['GroupStellarMass'][df_percent['Percent'] < Low_percentage]), np.log10(df_percent['GroupHIMass'][df_percent['Percent'] < Low_percentage])

parameters = ["$\mathrm{log}$ $\mathrm{M}_{\star}$", "$\mathrm{log}$ $\mathrm{M}_{\mathrm{HI}}$"]

c = ChainConsumer()
c.add_chain([x1, y1], parameters=parameters, name='Over {0}\% of HI is in Central'.format(High_percentage))
c.add_chain([x2, y2], parameters=parameters, name='Less than {0}\% of HI is in Central'.format(Low_percentage))


#Change sigmas to only 1 and it will show only 95%
c.configure(statistics='max', linestyles=["-", "--"], linewidths=[1.0, 1.0],  sigma2d=False, sigmas=[1,2], diagonal_tick_labels=False,
            legend_kwargs={"loc": "upper left", "fontsize": 15},
            legend_color_text=False, legend_location=(0, 0),
           label_font_size=25, tick_font_size=15, contour_label_font_size=20) #colors=['#2b8cbe', '#f4a582'],

fig = c.plotter.plot(figsize=1.5)
fig.set_size_inches(4.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.


# In[73]:


High_percentage = 50
Low_percentage = 50
Nsize_group = 4

x1, y1 = np.log10(df_percent['GroupStellarMass'][ (df_percent['Percent'] >= High_percentage) & (df_percent['GroupSize']==3) ]), np.log10(df_percent['GroupHIMass'][ (df_percent['Percent'] >= High_percentage) & (df_percent['GroupSize']==3) ])
    
x2, y2 = np.log10(df_percent['GroupStellarMass'][ (df_percent['Percent'] >= High_percentage) & (df_percent['GroupSize']==4) ]), np.log10(df_percent['GroupHIMass'][ (df_percent['Percent'] >= High_percentage) & (df_percent['GroupSize']==4) ])

x3, y3 = np.log10(df_percent['GroupStellarMass'][ (df_percent['Percent'] >= High_percentage) & (df_percent['GroupSize']==5) ]), np.log10(df_percent['GroupHIMass'][ (df_percent['Percent'] >= High_percentage) & (df_percent['GroupSize']==5) ])

parameters = ["$\mathrm{log}$ $\mathrm{M}_{\star}$", "$\mathrm{log}$ $\mathrm{M}_{\mathrm{HI}}$"]

c = ChainConsumer()
c.add_chain([x1, y1], parameters=parameters, name="Rich N = 3")
c.add_chain([x2, y2], parameters=parameters, name="Rich N = 4")
c.add_chain([x3, y3], parameters=parameters, name="Rich N = 5")

c.configure(linestyles=["-", "--", "-"], linewidths=[1.0, 1.0, 1.0],
            legend_kwargs={"loc": "upper left", "fontsize": 15},
            legend_color_text=False, legend_location=(0, 0),
            colors=["#1E88E5", "#D32F2F", "#111111"],
            shade=[False, True, True],
            shade_alpha=[0.2, 0.1, 0.0], bar_shade=[False, True, True])
fig = c.plotter.plot()

fig.set_size_inches(4.5 + fig.get_size_inches())  # Resize fig for doco. You don't need this.


# # Pair plot

# In[74]:


df_pair = pd.DataFrame({'GroupStellarMass'      : np.log10(g_st),
                           'GroupHIMass'        : np.log10(g_m),
                           'Percent'            : percentage,
                           'GroupSize'          : grp_length,
                           'BTTcentral'         : BTT_cen.ravel() ,
                            'Mvircen'           : Mvir_cen.ravel(),
                           'Rvircen'           : Rvir_cen.ravel() })
                           #'LenCen'             : Lenght_cen.ravel()})


# In[75]:



sns.set(font_scale=1)
sns.set_style("ticks", {"xtick.major.size":10, "ytick.major.size":10,
						"xtick.minor.size":6,"ytick.minor.size":6})
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})


g = sns.PairGrid(df_pair, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_upper(sns.scatterplot)
g.map_diag(sns.kdeplot, lw=3)

# To hide upper scatter plot axies: axis
#for i, j in zip(*np.triu_indices_from(g.axes, 1)):
#    g.axes[i, j].set_visible(False)


# In[160]:



fig = plt.figure(figsize=(13,10))                                                               
ax = fig.add_subplot(1,1,1)

cm = plt.cm.get_cmap('YlGnBu')
#Numpy doesn't know what you might have stored in a object array, 
#df_percent['StellarM'] = df_percent['StellarM'].astype(np.float64) 
#df_percent['HImass'] = df_percent['HImass'].astype(np.float64)

#BE CAREFUL HERE! Constructed sizes are OK for current groups; will need change
#s = 80*df_percent['Group_size']

sizes = [100,300,500,700,900]
im = plt.scatter(np.log10(df_percent['GroupStellarMass']), np.log10(df_percent['GroupHIMass']), s=60*df_percent['GroupSize'],
                        c=df_percent['Percent'], edgecolor='k', cmap=cm, label=r'Groups ($N\geq3$) properties')

fig.colorbar(im, ax=ax, orientation='vertical', label=r'\% of the M$_{\mathrm{HI}}$ in central',
            pad=0.01)

ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
ax.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)
ax.legend(loc=4)
leg = ax.get_legend()
leg.legendHandles[0].set_color('k')

l1 = plt.scatter([],[], s=100, color='white',edgecolors='k')
l2 = plt.scatter([],[], s=300, color='white',edgecolors='k')
l3 = plt.scatter([],[], s=500, color='white',edgecolors='k')
l4 = plt.scatter([],[], s=700, color='white',edgecolors='k')
l5 = plt.scatter([],[], s=900, color='white',edgecolors='k')

labels = ["3", "4", "5", "6", '7']

leg = plt.legend([l1, l2, l3, l4, l5], labels, ncol=1, loc=4, framealpha=0.5)
leg.set_title('Group size')                   
                   
#plt.ylim(6,11)


# In[79]:


# Initiate figure and how large it will be
def HI_group_distribution(groups_dict):
    
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
    
    
    fig, ax = plt.subplots(nrows=4, ncols=4, sharex='col', sharey='row', figsize=(18, 15))
    
    #ax[0][0].plot(np.log10(Mstellar_single_gal), np.log10(Mcoldgas_single_gal), 'o', color='lightgrey',markersize=8, label='Single galaxies')
    sat_g = []
    # Put plot in row and columns:3by3 are 00, 01, 02, 10, 11, 12, 20, 21, 22 
    row_count = 0
    for size in trange(1, 16, 1): #I'm planning to have 1-9 sized groups. Made to form trange(1, 9, 1) as 3x3 pannels 
            
        if size % 4 == 0:
            row_count += 1
        this_ax = ax[row_count, size%4] #Axis. Created for plotting 3x3 plots
        
        group_size = size+1 # Select number of galaxies per group --- adding +1 because it is counting from 0.
    
        central_gals = groups_dict[group_size]["Centrals"]  #List of integers. Contains indices of central galaxies.  List is obtained through the dictionary. 
        sat_gals = groups_dict[group_size]["Satellites"] #List of integers. Contains indices of satellite galaxies. List is obtained through the dictionary.

        sat_mass = np.sum(G['DiscHI'],axis=1)[sat_gals]
        sat_g.append(sat_gals)
        
        sum_of_all_satellites = np.sum(sat_mass)
        print(sum_of_all_satellites)
        
        sum_satellite_stellar = np.sum(G['StellarMass'][sat_gals])
        
        percentage = (np.sum(G['DiscHI'],axis=1)[central_gals]/(np.sum(G['DiscHI'],axis=1)[central_gals]+sum_of_all_satellites))*100
        #per_cent.append(percentage[0])
        groupHI = (np.sum(G['DiscHI'],axis=1)[central_gals]*1e10/h+1) + sum_of_all_satellites*1e10/h
        groupSt = (G['StellarMass'][central_gals]*1e10/h+1) + sum_satellite_stellar*1e10/h
        
        # Do plotting of groups of N-length; it will be placed on each subplot + +.
        # Centrals

        im = this_ax.scatter(np.log10(groupSt), np.log10(groupHI), s=300,
                        c=percentage, edgecolor='k', cmap=cm)

        # Add label for each sub-plot to know the group size and what is central/satellite
        this_ax.plot([], [], 'o', color='white', label='Groups of %.f' %group_size)
        this_ax.plot([], [], 'o', color='lightgrey', label='Single galaxies')
        this_ax.plot([], [], 'o', color='#1f78b4', markersize=8, label='Satellite')
        this_ax.plot([], [], 'o', color='white', markeredgecolor='#1f78b4', markersize=8, markeredgewidth=2, label='Central')
        #Add legend    
        leg = this_ax.legend(loc=4,frameon=True,fancybox=True, framealpha=0.6, fontsize=16)
        
    # Add x and y axis labels only on the plot edges since there will be no space between panels
    for row in range(3):
        this_ax = ax[row,0]
        this_ax.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)
    for col in range(3):
        this_ax = ax[2,col]
        this_ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
    # Tight layout and remove spacing between subplots
    plt.tight_layout()
    # My first plot is only single galaxies so I add legend separately
    ax[0][0].legend(loc=4,frameon=True,fancybox=True, framealpha=0.6, fontsize=16)

    plt.subplots_adjust(wspace = 0.0, hspace = 0.0)
    plt.show()
    print(sat_g)
    #return fig


# # Bulge to total ratio (BTT)

# In[80]:


#BTT_cen.ravel().tolist()


# In[81]:



fig = plt.figure(figsize=(13,10))                                                               
ax = fig.add_subplot(1,1,1)

cm = plt.cm.get_cmap('YlGnBu')
#Numpy doesn't know what you might have stored in a object array, 
BTT_cen = BTT_cen.astype(np.float64) 
#df_percent['HImass'] = df_percent['HImass'].astype(np.float64)

#BE CAREFUL HERE! Constructed sizes are OK for current groups; will need change
#s = 80*df_percent['Group_size']
sizes = [100,300,500,700,900]
im = plt.scatter(np.log10(df_percent['GroupStellarMass']), np.log10(df_percent['GroupHIMass']), s=sizes,
                        c=BTT_cen.ravel().tolist(), edgecolor='k', cmap=cm, label=r'Groups ($N\geq3$) properties')

fig.colorbar(im, ax=ax, orientation='vertical', label=r'B/T of central galaxy in a group',
            pad=0.01)

ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
ax.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)
ax.legend(loc=3)
leg = ax.get_legend()
leg.legendHandles[0].set_color('k')

l1 = plt.scatter([],[], s=100, color='white',edgecolors='k')
l2 = plt.scatter([],[], s=300, color='white',edgecolors='k')
l3 = plt.scatter([],[], s=500, color='white',edgecolors='k')
l4 = plt.scatter([],[], s=700, color='white',edgecolors='k')
l5 = plt.scatter([],[], s=900, color='white',edgecolors='k')

labels = ["3", "4", "5", "6", '7']

leg = plt.legend([l1, l2, l3, l4, l5], labels, ncol=1, loc=3, framealpha=0.5)
leg.set_title('Group size')                   
                   
#plt.ylim(6,11)


# ## N times N with B/T

# In[161]:


# Initiate figure and how large it will be
def mhi_vs_ms_3x3_colorbar(groups_dict):
    
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
    
    cm = plt.cm.get_cmap('YlGnBu')
    ax[0][0].scatter(np.log10(Mstellar_single_gal), 
                     np.log10(Mcoldgas_single_gal), 
                     c = BTT_single, s=80, edgecolor='k', cmap=cm)
    
    # Put plot in row and columns:3by3 are 00, 01, 02, 10, 11, 12, 20, 21, 22 
    row_count = 0
    for size in trange(1, 20, 1): #I'm planning to have 1-9 sized groups. Made to form trange(1, 9, 1) as 3x3 pannels 
            
        if size % 5 == 0:
            row_count += 1
        this_ax = ax[row_count, size%5] #Axis. Created for plotting 3x3 plots
        
        group_size = size+1 # Select number of galaxies per group --- adding +1 because it is counting from 0.
    
        central_gals = groups_dict[group_size]["Centrals"]  #List of integers. Contains indices of central galaxies.  List is obtained through the dictionary. 
        sat_gals = groups_dict[group_size]["Satellites"] #List of integers. Contains indices of satellite galaxies. List is obtained through the dictionary.
        
        BTT_central = groups_dict[group_size]["Centrals"]
        BTT_satellite = groups_dict[group_size]["Satellites"]
        
        # Plot single galaxies           
        #this_ax.plot(np.log10(Mstellar_single_gal), np.log10(Mcoldgas_single_gal), 'o', color='lightgrey',markersize=8)
        
            
        # Do plotting of groups of N-length; it will be placed on each subplot + +.
        # Centrals
        
        #im = plt.scatter(np.log10(df_percent['StellarM']+1), np.log10(df_percent['HImass']*1e10/h+1), s=80*df_percent['GroupSize'],
        #                c=df_percent['BTT_central'], edgecolor='k', cmap=cm, label=r'Groups ($N\geq3$) properties')
        
        cm = plt.cm.get_cmap('YlGnBu')
        this_ax.scatter(np.log10(G['StellarMass'][central_gals]*1e10/h), 
                         np.log10(np.sum(G['DiscHI'],axis=1)[central_gals]*1e10/h), 
                     c = BTT[BTT_central], s=80, edgecolor='k', cmap=cm)    
                     
        # Satellites
        im = this_ax.scatter(np.log10(G['StellarMass'][sat_gals]*1e10/h), 
                         np.log10(np.sum(G['DiscHI'],axis=1)[sat_gals]*1e10/h), 
                         c = BTT[BTT_satellite], s=80, cmap=cm)
    
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
        this_ax = ax[2,col]
        this_ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
    # Tight layout and remove spacing between subplots
    plt.tight_layout()
    # My first plot is only single galaxies so I add legend separately
    
    # COLORBAR INSIDE SUBPLOT
    #from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    #cbaxes = inset_axes(ax[0][0], width="5%", height="50%", loc='center right') #loc=l, bbox_to_anchor=(0.6,0.5)
    #plt.colorbar(im, cax=cbaxes, ticks=[0.,0.5,1], orientation='vertical')
    #cbaxes.yaxis.set_ticks_position('left')
    
    ax[0][0].legend(loc=4,frameon=True,fancybox=True, framealpha=0.6, fontsize=20)

    plt.subplots_adjust(wspace = 0.0, hspace = 0.0)
    
    # COLORBAR for the entire plot
    plt.colorbar(im, ax=ax[:, 4], pad=0.01, label=r'B/T ratio')
    plt.show()
    #return fig


# In[162]:


mhi_vs_ms_3x3_colorbar(updated_dict)


# In[163]:


# Initiate figure and how large it will be
def BT_distribution(groups_dict):
    
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
    
    
    fig, ax = plt.subplots(figsize=(8,8))
    
    #ax[0][0].plot(np.log10(Mstellar_single_gal), np.log10(Mcoldgas_single_gal), 'o', color='lightgrey',markersize=8, label='Single galaxies')
    
    # Put plot in row and columns:3by3 are 00, 01, 02, 10, 11, 12, 20, 21, 22 
    
    BTT_c = []
    BTT_s = []
    
    for size in trange(1, 20, 1): #I'm planning to have 1-9 sized groups. Made to form trange(1, 9, 1) as 3x3 pannels 
    
        group_size = size+1 # Select number of galaxies per group --- adding +1 because it is counting from 0.
    
        central_gals = groups_dict[group_size]["Centrals"]  #List of integers. Contains indices of central galaxies.  List is obtained through the dictionary. 
        sat_gals = groups_dict[group_size]["Satellites"] #List of integers. Contains indices of satellite galaxies. List is obtained through the dictionary.
        
        BTT_central = groups_dict[group_size]["Centrals"]
        BTT_satellite = groups_dict[group_size]["Satellites"]
        
        BTT_c.extend(BTT[BTT_central])
        BTT_s.extend(BTT[BTT_satellite])
    
    binwidth=0.1
    bins = np.arange(0, 1.1, binwidth)
    plt.hist(BTT_s, bins=bins, color='#2c7fb8', edgecolor='k', linewidth=1, alpha=0.8, label='Satellite')
    plt.hist(BTT_c, bins=bins, color='white', edgecolor='#2c7fb8', linewidth=4, alpha=0.6, label='Central')

    
    leg = ax.legend(loc=1,frameon=True,fancybox=True, framealpha=0.6, fontsize=16)
    
    ax.set_xlabel(r'B/T ratio',fontsize=25)
    ax.set_ylabel(r'N$_{\textrm{galaxies}}$',fontsize=25)
    
    plt.tight_layout()

    plt.subplots_adjust(wspace = 0.0, hspace = 0.0)
    plt.xlim(-0.05,1.05)
    plt.show()
    #return fig


# In[164]:


BT_distribution(updated_dict)


# In[ ]:





# In[86]:


# Initiate figure and how large it will be
def mhi_vs_ms_colorbar(groups_dict):
    
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
    
    
    fig, ax = plt.subplots(figsize=(15, 12))
    
    #ax[0][0].plot(np.log10(Mstellar_single_gal), np.log10(Mcoldgas_single_gal), 'o', color='lightgrey',markersize=8, label='Single galaxies')
    
    # Put plot in row and columns:3by3 are 00, 01, 02, 10, 11, 12, 20, 21, 22 
    for size in trange(1, 9, 1): #I'm planning to have 1-9 sized groups. Made to form trange(1, 9, 1) as 3x3 pannels 
    
        group_size = size+1 # Select number of galaxies per group --- adding +1 because it is counting from 0.
    
        central_gals = groups_dict[group_size]["Centrals"]  #List of integers. Contains indices of central galaxies.  List is obtained through the dictionary. 
        sat_gals = groups_dict[group_size]["Satellites"] #List of integers. Contains indices of satellite galaxies. List is obtained through the dictionary.
        
        BTT_central = groups_dict[group_size]["Centrals"]
        BTT_satellite = groups_dict[group_size]["Satellites"]
        

        cm = plt.cm.get_cmap('YlGnBu')
        ax.scatter(np.log10(G['StellarMass'][central_gals]*1e10/h), 
                         np.log10(np.sum(G['DiscHI'],axis=1)[central_gals]*1e10/h), 
                     c = BTT[BTT_central], s=150, edgecolor='k', linewidth=1, cmap=cm)    
                     
        # Satellites
        ax.scatter(np.log10(G['StellarMass'][sat_gals]*1e10/h), 
                         np.log10(np.sum(G['DiscHI'],axis=1)[sat_gals]*1e10/h), 
                         c = BTT[BTT_satellite], s=150, cmap=cm)
    
        # Add label for each sub-plot to know the group size and what is central/satellite
        #ax.plot([], [], 'o', color='lightgrey', label='Single galaxies')
    ax.plot([], [], 'o', color='#1f78b4', markersize=10, label='Satellite')
    ax.plot([], [], 'o', color='white', markeredgecolor='k', markersize=10, markeredgewidth=3, label='Central')
        #Add legend  
        
    ax.set_xlabel(r'log M$_{\star}$ [M$_{\odot}$]', fontsize=25)
    ax.set_ylabel(r'log M$_{\textrm{HI}}$ [M$_{\odot}$]',fontsize=25)
    
    leg = ax.legend(loc=4,frameon=True,fancybox=True, framealpha=0.6, fontsize=16)

    plt.tight_layout()

    plt.subplots_adjust(wspace = 0.0, hspace = 0.0)
    
    # COLORBAR for the entire plot
    plt.colorbar(im, ax=ax, pad=0.01, label=r'B/T ratio')
    plt.show()
    #return fig


# In[87]:


mhi_vs_ms_colorbar(updated_dict)


# # How the group look like (Mvir/Rvir/Pos)
# #### Make a function to extract parameters

# In[165]:


for i in c_ind[0:5]:
    print(i, G['Rvir'][i][0:])


# In[166]:


for i in g_ind[0:5]:
    print(G['Pos'][i][0:])


# In[90]:


fig, ax = plt.subplots(figsize=(8,8))
for i in g_ind[0]:
    print(G['Pos'][i][0])
    
    
    plt.plot( (G['Pos'][i])[0], (G['Pos'][i])[1], 'ko')
    plt.plot( 14.069526, 497.66397, 'rD', markersize=15, alpha=0.1)
    
    circle1 = plt.Circle((14.069526, 497.66397), 0.27567366, color='k', fill=False, clip_on=False)
    
    circle2 = plt.Circle((13.939832, 497.74933), 0.05918148, color='b', fill=False, clip_on=False)
    circle3 = plt.Circle((14.123639, 497.45767), 0.06282725, color='b', fill=False, clip_on=False)
    
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    
    #plt.tight_layout()
    ax.set_aspect('equal')
    plt.xlim(14.069526-2*0.27567366,14.069526+2*0.27567366)
    plt.ylim(497.66397-2*0.27567366, 497.66397+2*0.27567366)
    
    ax.set_xlabel(r'[Mpc/h]', fontsize=25)
    ax.set_ylabel(r'[Mpc/h]',fontsize=25)


# In[91]:


#HI Masses, obtained from the indices: 
perc_richer_central_hi_m = np.log10( (np.sum(G['DiscHI'],axis=1)[perc_richer_central_ind] *1e10/h )+1)
perc_poorer_central_hi_m = np.log10( (np.sum(G['DiscHI'],axis=1)[perc_poorer_central_ind] *1e10/h )+1)
perc_richer_central_s_m = np.log10( G['StellarMass'][perc_richer_central_ind] *1e10/h +1)
perc_poorer_central_s_m = np.log10( G['StellarMass'][perc_poorer_central_ind] *1e10/h +1)

#have to store satellites like this; otherwise it complains
perc_richer_sat_hi_m = []
perc_richer_sat_s_m = []

for item in perc_richer_sat_ind:
    p_richer_sat_hi_m = np.log10( (np.sum(G['DiscHI'],axis=1)[item] *1e10/h )+1)
    p_richer_sat_s_m = np.log10( G['StellarMass'][item] *1e10/h +1)
    perc_richer_sat_hi_m.append(p_richer_sat_hi_m)
    perc_richer_sat_s_m.append(p_richer_sat_s_m)

perc_poorer_sat_hi_m = []
perc_poorer_sat_s_m = []
for item in perc_poorer_sat_ind:    
    p_poorer_sat_hi_m = np.log10( (np.sum(G['DiscHI'],axis=1)[item] *1e10/h )+1)
    p_poorer_sat_s_m = np.log10( G['StellarMass'][item] *1e10/h +1)
    perc_poorer_sat_hi_m.append(p_poorer_sat_hi_m)
    perc_poorer_sat_s_m.append(p_poorer_sat_s_m)


# In[92]:


two_sided_histogram_rich(perc_richer_central_s_m, perc_richer_central_hi_m, 
                         perc_poorer_central_s_m, perc_poorer_central_hi_m) 


# In[93]:


two_sided_histogram_rich_groups(perc_richer_central_s_m.ravel(), perc_richer_central_hi_m.ravel(), 
                                perc_poorer_central_s_m.ravel(), perc_poorer_central_hi_m.ravel(),
                                perc_richer_sat_s_m, perc_richer_sat_hi_m, 
                                perc_poorer_sat_s_m, perc_poorer_sat_hi_m) 


# # Central and satellite (similar Mstar; large MHI difference)
# ##### Takes too long to run

# In[84]:


#Use dictionary and for each galaxy in "Groups" of 3 show central/satellite index

diff_central_ind = [] #storring where central/satellite is above/below some percentage
norm_diff_central_ind = []

norm_diff_sat_ind = [] #satellites that belong to richer central
diff_sat_ind = [] #satellites that belong to poorer central

diff_groups = []
gr_key = []
#per_cent = [] # How much of HI mass is in central galaxy (in %)

for i in trange(3,10,1): #starting grom galaxy pairs
    for group_key in updated_dict[i]["Groups"].keys():
        #print(group_key) #group_1 etc
        central_idx = updated_dict[i]["Groups"][group_key]["Centrals"] #give indices
        central_mass = np.log10(np.sum(G['DiscHI'],axis=1)[central_idx]*1e10/h +1) #give masses of these galaxies
        central_star_mass = np.log10(G['StellarMass'][central_idx]*1e10/h +1)
        
        satellite_inds = updated_dict[i]["Groups"][group_key]["Satellites"]
        sat_mass = np.log10(np.sum(G['DiscHI'],axis=1)[satellite_inds]*1e10/h +1)
        sat_star_mass = np.log10(G['StellarMass'][satellite_inds]*1e10/h +1)
        
        sum_of_all_satellites = np.sum(sat_mass)
        percentage = (central_mass/(central_mass+sum_of_all_satellites))*100
        per_cent.append(percentage[0])
        
        diff_in_star_mass = abs(central_star_mass - sat_star_mass) #abs difference in stellar masses
        #print(central_star_mass)
        #print(sat_star_mass)
        #print('diff star', diff_in_star_mass)
        diff_in_hi_mass = abs(central_mass - sat_mass) #abs difference in HI mass
        #print('hi cent mass', central_mass)
        #print('hi sat mass', sat_mass)
        
        if ((diff_in_star_mass < 0.4).all() & (diff_in_hi_mass > 0.5 ).all()):
            print('diff_st', diff_in_star_mass)
            print('diff_hi', diff_in_hi_mass)
            
            #print('yes')#check conditions
#            print("Central has more HI than the sum of satellites")
            diff_central_ind.append(central_idx[0]) #[0] so its creates a list
            diff_sat_ind.extend(satellite_inds) #it complains and complicates when i want to do .append
            diff_groups.append(updated_dict[i]["Groups"][group_key]['Centrals'][0])
            diff_groups.extend(satellite_inds)
            gr_key.append(group_key)
        else:
#            print("Sum of the satellites are more massive than central")
            norm_diff_central_ind.append(central_idx[0])
            norm_diff_sat_ind.extend(satellite_inds)
#        #print(sat_mass > central_mass)
#    
#        print("Central mass is {0} and sat masses are {1}".format(central_mass, sat_mass))
#        print("")


# In[85]:


print(diff_groups)
print(gr_key)


# In[86]:


#HI Masses, obtained from the indices: 
diff_central_hi_m = np.log10( (np.sum(G['DiscHI'],axis=1)[diff_central_ind] *1e10/h )+1)
diff_central_s_m = np.log10( G['StellarMass'][diff_central_ind] *1e10/h +1)

#have to store satellites like this; otherwise it complains
diff_sat_hi_m = np.log10( (np.sum(G['DiscHI'],axis=1)[diff_sat_ind] *1e10/h )+1)
diff_sat_s_m = np.log10( G['StellarMass'][diff_sat_ind] *1e10/h +1)

#HI Masses, obtained from the indices: 
norm_diff_central_hi_m = np.log10( (np.sum(G['DiscHI'],axis=1)[norm_diff_central_ind] *1e10/h )+1)
norm_diff_central_s_m = np.log10( G['StellarMass'][norm_diff_central_ind] *1e10/h +1)

#have to store satellites like this; otherwise it complains
norm_diff_sat_hi_m = np.log10( (np.sum(G['DiscHI'],axis=1)[norm_diff_sat_ind] *1e10/h )+1)
norm_diff_sat_s_m = np.log10( G['StellarMass'][norm_diff_sat_ind] *1e10/h +1)


# In[87]:


two_sided_histogram_difference(diff_central_s_m.ravel(), diff_central_hi_m.ravel(), 
                                norm_diff_central_s_m.ravel(), norm_diff_central_hi_m.ravel(),
                                diff_sat_s_m, diff_sat_hi_m, 
                                norm_diff_sat_s_m, norm_diff_sat_hi_m) 


# In[88]:


item1 = 9
item2 = 12
diff_groups_sm = np.log10( G['StellarMass'][diff_groups] *1e10/h +1)
diff_groups_him = np.log10( (np.sum(G['DiscHI'],axis=1)[diff_groups] *1e10/h) +1)


# In[89]:



two_sided_histogram_difference_groups(diff_central_s_m.ravel(), diff_central_hi_m.ravel(), 
                                norm_diff_central_s_m.ravel(), norm_diff_central_hi_m.ravel(),
                                diff_sat_s_m, diff_sat_hi_m, 
                                norm_diff_sat_s_m, norm_diff_sat_hi_m,
                              diff_groups_sm, diff_groups_him) 


# In[100]:


print(diff_groups)
print(diff_central_ind)
print(gr_key)


# In[101]:


for i in gr_key:
    print(updated_dict[3]["Groups"][i]) #shows indices of Nth group in groups of 3


# In[102]:


#Divide array into n-arrays
#f = lambda diff_groups, n=3: [diff_groups[i:i+n] for i in range(0, len(diff_groups), n)]
#New_groups = f(diff_groups)
#print(New_groups)


# ## Place colormap for centrals

# In[103]:


import matplotlib.cm as cm


# In[ ]:




