
from sys import argv, exit
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
#import pydicom as dicom
import os
import time 
import streamlit as st
from streamlit_option_menu import option_menu

import os
#import streamlit as st

os.environ ['PYTHONINSPECT'] = '1'

def disp_dcm ( im )  : 
#  ds = dicom.dcmread (dcmfname)  
  figsize = (2, 2) 
  fig1, ax1 = plt.subplots (1,1, figsize = figsize)
  ax1.imshow (im, cmap = plt.cm.gist_gray, vmin = 1024-200, vmax = 1024+200 ) 
  ax1.axis ('equal')
  ax1.axis ('off')

  st.pyplot(fig1)

#@st.cache
def  dcm_load ( this_fobj) : 
  ds = dicom.dcmread ( this_fobj , force=True )  
  im = ds.pixel_array 
  return im

if False :
  code  = '''
    data = np.fromfile (youfilename, )
    '''

fpath =  r"C:\Users\jxu\OneDrive - Johns Hopkins\proposals\Taguchi-BrainPerfusion\PatientData\Patient-woPHI"
fname = r"\p0001\CT\AUF1DY4O\WWQNSQQX"

if False : 
  with st.sidebar : 
    selected = option_menu (
      menu_title =  'main menu', 
      options = ['data upload', 'study design'], 
      )


st.subheader ('upload data')
col1, col2 = st.columns (2, gap = "large")
with col1 :
  train_fobj = st.file_uploader ("Upload training data set", type = ['dcm' ,'raw f32'], disabled = False  ) 
  if train_fobj is not None : 

    this_fobj = train_fobj
    filetype = os.path.splitext(this_fobj.name)[1]
    train_fdetails = {'filename': this_fobj.name, 
          'filetype':filetype } 
    st.write (train_fdetails)
    if (filetype.casefold () == '.dcm' )  : 
      im = dcm_load ( this_fobj)
      disp_dcm ( im )   


with col2 :
  test_data =st.file_uploader ("Upload testing data set" , type = ['dcm', 'raw f32'], disabled = False) 

st.markdown('---')

st.subheader('task definition')
task_params = {
  'task' : st.radio ( label =  'Task definition', 
                  options = ('detection w/ known location (ROC)', 'detection w/ unknown location (LROC)'), 
                  index = 0 , label_visibility="collapsed", horizontal = True)
  }

st.subheader('observer types')
obs_params = {
    'name' : st.radio (label = 'Observer Type',  
                          options =  ('IO', 'NPW', 'CHO') , 
                          index = 2, label_visibility = "collapsed", horizontal = True), 
  }

if obs_params['name'] == 'CHO' : 

  st.subheader('CHO parameters')
  col1, col2, col3 = st.columns (3, gap = "large")
  with col1 :
    cho_type = st.radio ('Type of channels',  options  = ('Gabor', 'Square Symmetric', 'DOG')  ) 
  with col2 :
    cho_num_channels = st.selectbox ('Number of channels',  options = (20, 30, 40) ) 
  with col3 :
    cho_internal_noise =  st.slider ('Internal noise', min_value = 0.0, max_value  = 1., step = 0.1  ) 

st.markdown ('---')

col1, col2 = st.columns (2, gap = "large")
with col1 : 
  train_obs = st.button ('train observer')
  if (train_obs) : 
    st.write ('train button clicked')
with col2 : 
  test_obs = st.button ('test observer')
  if (test_obs) : 
    st.write ('test button clicked')

  
#st.title ('menu options')


