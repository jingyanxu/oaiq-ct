
import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider
import numpy as np
import pandas as pd 
import os
import time 
import streamlit as st
#from skimage import measure
#import nibabel as nib
from numpy.random import default_rng
from sklearn import metrics

from streamlit_option_menu import option_menu

distribution_fun = ['Gaussian', 'Poisson', 'Uniform' ] 
menu_options  = ['ROC/AUC basics', 'Model_observers']
seed = int (time.time () )
seed = 12345
rng = default_rng (seed = seed)

def  disp_histogram (samples_c1, samples_c2, bins ) : 
  min_val = min (np.min (samples_c1), np.min (samples_c2)) 
  max_val = max (np.max (samples_c1), np.max (samples_c2)) 
  h1, b_edges = np.histogram (samples_c1, bins = bins, range  = (min_val, max_val )   )
  h2, b2_edges = np.histogram (samples_c2, bins = bins, range  = (min_val, max_val )   )

  figsize = (5,  3 ) 
  fig1, ax1 = plt.subplots (1,1, figsize = figsize)
  ax1.stairs (h1, b_edges, label = 'class 1') 
  ax1.stairs (h2, b_edges, label = 'class 2') 
  ax1.set_title ('histogram' ) 
  ax1.set_ylabel ('occurrences' ) 
  ax1.legend ()
  st.pyplot ( fig1 ) 

def  delong_roc (samples_c1, samples_c2) : 
  n1 = len (samples_c1)
  n2 = len (samples_c2)
  n1n2 = n1*n2
  aa, bb = np.meshgrid (samples_c1, samples_c2)
  pos_mask = (aa <= bb ) 
  auc1 = np.sum (np.where (aa < bb, 1, 0) )
  auc2 = np.sum (np.where (aa == bb, 0.5, 0) )
  auc = (auc1 + auc2 ) / n1n2  
#  st.write ('{} ==  {}'.format (n1, aa.shape [1]) ) 
#  st.write ('{} ==  {}'.format (n2, aa.shape [0]) ) 
  
  comp1 = np.sum ((np.sum (pos_mask, axis = 0)/n2  - auc )**2 ) /(n1  -1 )
  comp2 = np.sum ((np.sum (pos_mask, axis = 1)/n1  - auc  ) **2 ) /(n2 -1 )
  auc_std = np.sqrt (comp1/n1 + comp2/n2 )
  return auc , auc_std
def  disp_roc (samples_c1, samples_c2) : 
  
  y_true = np.concatenate (( np.zeros_like (samples_c1), np.ones_like (samples_c2) )) 
  y_score = np.concatenate (( samples_c1, samples_c2 )) 
  fpr, tpr, thresholds = metrics.roc_curve (y_true, y_score)
  roc_auc0 = metrics.auc (fpr, tpr)
  if roc_auc0 < 0.5 : 
    roc_auc = 1 - roc_auc0
    delong_auc, auc_stdv =  delong_roc (samples_c2, samples_c1)  
  else : 
    roc_auc = roc_auc0
    delong_auc, auc_stdv =  delong_roc (samples_c1, samples_c2)  
  figsize = (5,  3 ) 
  fig1, ax1 = plt.subplots (1,1, figsize = figsize)

  if roc_auc0 < 0.5 : 
    ax1.plot ( tpr, fpr ) 
  else : 
    ax1.plot (fpr, tpr ) 
  ax1.set_xlabel ('false positive rate')
  ax1.set_ylabel ('true positive rate')
#  ax1.set_xlim (0, 1)
#  ax1.set_ylim (0, 1)
  ax1.set_title ("auc: {:.3g} $\pm$ {:.3g}".format (delong_auc, auc_stdv))
#  ax1.axis ('equal')  
  st.pyplot ( fig1 ) 
#  st.write ('here we display roc')
  

def  tutorial_auc_roc ( ) :  
  st.header ('ROC/AUC basics')
  col1, col2 = st.columns ([1 ,1 ], gap = "large")
  with col1 : 
    st.subheader ('class 1')
  with col2 : 
    st.subheader ('class 2')

  col2a, col2b, col2c, col2d = st.columns ([1 ,1,  1, 1 ], gap = "large")
  with col2a : 
    options_c1 = st.selectbox ('distribution function', options = distribution_fun, index = 2, key = 'c1')
  with col2b : 
    num_sample_c1 = st.number_input ('# samples', min_value = 10, max_value = 100, value = 50, step=10, key='c1c1')
  with col2c : 
    options_c2 = st.selectbox ('distribution function', options = distribution_fun, index = 2, key = 'c2')
  with col2d : 
    num_sample_c2 = st.number_input ('# samples', min_value = 10, max_value = 100, value = 50, step=10, key='c2c2')

  col3a, col3b  = st.columns ([1 ,1  ], gap = "large")
  with col3a : 
    if (options_c1 == distribution_fun [0] )  :  # Gaussian
      mu1 = st.slider ('mean value', min_value = -5., max_value = 5., value = 0., step = 0.1, key='c0' )
      stdv1 = st.slider ('standard devivation', min_value = 1.,  max_value = 10., value = 1., step = 0.1, key='-c1')
      samples_c1 = rng.standard_normal (size = num_sample_c1) * stdv1 + mu1
    elif (options_c1 == distribution_fun [1] ): 
      mu1 = st.slider ('mean value (lambda)', min_value = 0.0, max_value = 10., value = 5., step = 0.5, key='c1c2' )
      samples_c1 = rng.poisson (lam = mu1, size = num_sample_c1) 
    elif (options_c1 == distribution_fun [2] ) :
      mu1 = st.slider ('range ', min_value = -10.0, max_value = 10.0, value = (-5., 2.5), step = 0.1, key='c1c1c1' )
      samples_c1 = rng.uniform (low = mu1[0], high = mu1[1], size = num_sample_c1) 

  with col3b : 
    if (options_c2 == distribution_fun [0] )  :  # Gaussian
      mu2 = st.slider ('mean value', min_value = -5., max_value = 5., value = 0., step = 0.1, key = 'c00' )
      stdv2 = st.slider ('standard devivation', min_value = 1.,  max_value = 10., value = 1., step = 0.1, key ='-c2')
      samples_c2 = rng.standard_normal (size = num_sample_c2) * stdv2 + mu2
    elif (options_c2 == distribution_fun [1] ): 
      mu2 = st.slider ('mean value (lambda)', min_value = 0.0, max_value = 10., value = 5., step = 0.5, key='c2c22')
      samples_c2 = rng.poisson (lam = mu2, size = num_sample_c2) 
    elif (options_c2 == distribution_fun [2] ) :

      mu2 = st.slider ('range', min_value = -10.0, max_value = 10.0, value = (-2.5, 5.), step = 0.1, key='c2c2c2')
      samples_c2 = rng.uniform (low = mu2[0], high = mu2[1], size = num_sample_c2) 

  col4a, col4b  = st.columns ([1 ,1  ], gap = "large")
  with col4a : 
    disp_histogram (samples_c1, samples_c2, bins =  (num_sample_c2 + num_sample_c1)// 4)
  with col4b : 
    disp_roc (samples_c1, samples_c2)
  if False : 
    gen_samples = st.button (label = 'click to generate samples', disabled = False)
    gen_roc = st.button (label = 'click to generate ROC curve', disabled = False)
    if (gen_samples == True) : 
      st.write ('gen sample button clicked')
    if (gen_roc == True) : 
      st.write ('gen roc clicked clicked')

with st.sidebar:
    selected = option_menu("Main Menu", menu_options , 
        icons=['house', 'gear'], menu_icon="cast", default_index=0)
if selected == menu_options [0] :   
  tutorial_auc_roc ( ) 
else : 
  st.write ('something else') 
