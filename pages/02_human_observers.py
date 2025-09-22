
from sys import argv, exit
import random
import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider
import numpy as np
import pandas as pd 
import os
import time 
import streamlit as st
from skimage import measure
import nibabel as nib

#from PIL import Image
#from streamlit_option_menu import option_menu
#from streamlit_drawable_canvas import st_canvas
#from bokeh.plotting import figure, show

import os
#import streamlit as st

os.environ ['PYTHONINSPECT'] = '1'

st.sidebar.subheader ('Choose task')
task_options = ('detection w/ known location (ROC)' , 'detection w/o known location (LROC)') 
task_params = st.sidebar.radio ( label =  'choose task', 
                        options = task_options , 
                        index = 0 , horizontal = False, label_visibility = "collapsed" )

st.sidebar.subheader ('Choose session')
reader_params = st.sidebar.radio ( label =  'choose session', 
                        options = ('training', 'testing'), 
                        index = 0 , horizontal = True, label_visibility = 'collapsed' )

# ROI shape and size
ROI_shape_options  = ('Box', 'Circle') 
pen_color = st.sidebar.color_picker (label = 'pick pen color', 
        value ='#aaeebb', disabled = (task_params == task_options[0]) )
roi_shape = st.sidebar.selectbox (label = 'ROI shape',  options = ROI_shape_options, index = 0,
          disabled = (task_params == task_options [0])) 
roi_size = st.sidebar.number_input (label = 'ROI size',  min_value = 10, max_value = 30, value = 10, step=5, 
          disabled = (task_params == task_options [0])) 

with st.sidebar.expander ('rating interpretations:') : 
  if False : 
    st.write ('0 -- 1 :  definitely absent\n', \
     '1 -- 2 :  more likely absent', \
    '2 -- 3 :  more likely present', \
    '3 -- 4 :  definitely present' ) 
  html_string = "<p>0 -- 1: definitely absent<br>1 -- 2: more likely absent <br>2 -- 3: more likely present <br>3 -- 4: definitely present</p>"  
  st.markdown (html_string, unsafe_allow_html=True)


#@st.cache
def load_im (train_fnames) : 
  im_tuple = ()
  for iname in train_fnames : 
    ds = nib.load ( iname ).get_fdata () 
  #im = im_normalize (
    im1 = np.transpose ( ds , axes = [2,1, 0] ) #  , vmin = 1024-200, vmax = 1024+200 )
    im_tuple += (im1,)
  print ('total number of images {}'.format(len (im_tuple)))  
  return im_tuple

def display_training (im3,  view_label,  ntotal) : 

  figsize = (5, 5*110/212) 
  fig1, ax1 = plt.subplots (1,1, figsize = figsize)

  ax1.axis ('equal')
  ax1.axis ('off')
  ax1.imshow (im3[0,:, :], cmap = plt.cm.gist_gray, vmin = 0-200, vmax = 0+200 )   
  if  ( task_params == task_options[1])  : 
    xx = [xpos - roi_size/2., xpos +roi_size/2. , xpos + roi_size/2.,  xpos-roi_size/2., xpos-roi_size/2.] 
    yy = [ypos - roi_size/2., ypos -roi_size/2.,  ypos + roi_size/2., ypos+roi_size/2., ypos-roi_size/2.]
    ax1.plot (xx, yy, color = pen_color)
  if view_label : 
    contours_cyst = measure.find_contours (im3[2, :, :], 0.5 ) 
    ax1.plot (contours_cyst[0][:,1], contours_cyst[0][:, 0], color = 'r',linewidth=1)

#  fig1.text ( 0.4, 0.95, 'training {}/{}'.format(idx+1, ntotal), fontsize = 8, color = (38/255,39/255,48/255))
  fig1.text ( 0.4, 0.95, 'training {}/{}'.format(idx+1, ntotal), fontsize = 8, color = (0, 0, 0))
  fig1.text ( 0.6, 0.95, 'training {}/{}'.format(idx+1, ntotal), fontsize = 8, color = 'w')
  ax1.autoscale_view('tight')
  fig1.patch.set_alpha (0.0)

  st.pyplot ( fig1 ) 

  return 

def train_session (im_tuple, idx, ntotal )  : 
  display_training (im_tuple [idx],  view_label, ntotal)

  return idx  #, done_training

load_options = ('start over', 'finish training')  
#done_training = False 
  
rel_path = r"pages/images"
train_fpath = os.getcwd () + rel_path 

train_fnames = [ rel_path + '/' + i for i in os.listdir (rel_path) ]
im_tuple = load_im (train_fnames )
#train_fnames =  os.listdir (rel_path ) 
ntotal = len (train_fnames) 
#ntotal = 5 

if 'done_training' not in st.session_state  : 
  st.session_state ['done_training'] =  False  # done_training 
else : 
  done_training = st.session_state ['done_training'] 

if 'idx' not in st.session_state  : 
  idx = 0  
  st.session_state ['idx'] = idx 
else : 
  idx = st.session_state ['idx']  
if 'result' not in st.session_state : 
  st.session_state ['result'] = [] 

#start_over = True 
#finish = False 
#load_next = False 

def change_index_state () : 
  if (st.session_state['idx'] < ntotal-1) : 
    st.session_state ['idx'] += 1 
#    len_list = len (st.session_state ['result']  )
  else : 
    done_training = True 
    st.session_state ['done_training'] =  done_training  # done_training 

def init_index_state () : 
  idx = 0
  st.session_state ['idx'] = idx 
  st.session_state ['result'] = [] 
  # done_training = False 
  st.session_state ['done_training'] =  False # done_training 

def finish_training () : 
  #done_training = True 
  st.session_state ['done_training'] = True #  done_training 

#st.write (idx+1, ntotal, st.session_state['idx']+1, (idx < ntotal), len (st.session_state['result']) )

col1, col2, col3, col4 = st.columns ([1 ,1, 1, 1], gap = "large")
with col1 : 
  load_next = st.button ( label =  'load next', disabled =  False , on_click = change_index_state )
with col2 : 
  start_over = st.button ( label =  'start over', disabled =  False, on_click = init_index_state  )
with col3 : 
  finish = st.button ( label =  'finish', disabled =  False, on_click = finish_training  )
with col4 : 
  view_label = st.checkbox ('view label', disabled = (task_params == task_options[0]))

col1a, col2a, col3a = st.columns ([1 ,1, 1], gap = "large")
with col1a : 
  xpos =  st.slider ('horizontal position (x)', min_value = 0, max_value  = 212, value = 106, step = 1, 
          disabled = ( task_params == task_options[0] )) 

with col2a : 
  ypos =  st.slider ('vertical position (y)', min_value = 0, max_value  = 110, value = 55, step = 1 , 
          disabled = (task_params == task_options[0] )) 
with col3a : 
  rating =  st.slider ('confidence rating (0: low, 4: high)', min_value = 0.0, max_value  = 4., 
              value = 2.0, step = 0.1 ) 


st.markdown ('---')

results_list = []

if (load_next  ) : 
      if (idx < ntotal-1) : 
      # idx +=  1
        pass 
      else : 
        done_training = True 

  #    xpos_list.append (xpos)  
  #    ypos_list.append (ypos)  
      len_list = len (st.session_state ['result']  )
      if ( len_list < ntotal ) : 
        st.session_state['result'].append (np.array ( [xpos, ypos, rating]) )  

#st.write (idx+1, ntotal, st.session_state['idx']+1, (idx < ntotal), len (st.session_state['result']))

df_columns = ['xpos', 'ypos', 'rating'] 
df_format_columns = ['{:.1f}', '{:.1f}', '{:.2f}'] 


df_format_dict  = dict ([(df_columns [i], df_format_columns[i] ) for i in  range(len (df_columns )) ]  )

col3a, col3b = st.columns ([2 ,1], gap = "large")
with col3a : 
  if reader_params == 'training'  : 

    if (idx < ntotal) : 
      _ = train_session (im_tuple, idx , ntotal)
#      st.session_state ['idx'] = idx 
#      st.session_state ['done_training'] =  done_training 

with col3b : 
  if (st.session_state ['done_training'] == True)  : 
      st.write ('training session completed')  
      df = pd.DataFrame ( st.session_state['result'], columns = df_columns)
      if (task_params == task_options [0] ) :
        st.dataframe ( df [ [ df_columns[-1]]].style.format ( df_format_dict) ) 
      else : 
        st.dataframe ( df.style.format ( df_format_dict) ) 

      if (task_params == task_options [0])  :      
        roc = st.button (label = 'ROC analysis', disabled = False)      
      else : 
        roc = st.button (label = 'LROC analysis', disabled = False)      

          
if reader_params == 'testing'  : 
  idx0 = st.write('start testing')
  

  
#st.title ('menu options')


