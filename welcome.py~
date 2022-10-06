
from sys import argv, exit
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os
import time 
import streamlit as st
from streamlit_option_menu import option_menu

import os
#import streamlit as st

os.environ ['PYTHONINSPECT'] = '1'


fpath =  r"C:\Users\jxu\OneDrive - Johns Hopkins\proposals\Taguchi-BrainPerfusion\PatientData\Patient-woPHI"
fname = r"\p0001\CT\AUF1DY4O\WWQNSQQX"

  
#st.title ('menu options')



st.subheader ('Welcome to OAIQ for CT image formation')
jobs_list = ['ML/DL/computer vision', 'Medical Imaging', 'CT', 'PET', 'SPECT', 'Radiologist'] 

st.markdown ('---')
user = st.text_input (label = 'enter name', 
            placeholder = 'enter name (id)', label_visibility = 'collapsed')
st.text ('Your background')

user_bg = st.multiselect (label = 'your profession', options = jobs_list, 
      default = None, label_visibility = 'collapsed')
if (len (user_bg ) > 0 )   : 
  st.write ('Awesome!')  


st.text ('Years of experience in your background')
yoe_list = ['< 5 years', '< 10 years', '>= 10 years']
user_yoe = st.selectbox (label = 'yoe', options = yoe_list, 
      index = 0, label_visibility = 'collapsed')

st.text ('Are you familiar with objective assessment of image quality? (OAIQ)')
oaiq_list = ['No', 'Somewhat', 'Expert']
user_oaiq = st.selectbox (label = 'familiarity', options = oaiq_list, 
      index = 0, label_visibility = 'collapsed')

st.text ('Anything comments/suggestions/bugs about this web page?')
text_out =  st.text_area (label = 'anything else', max_chars = 50, placeholder = 'comments, suggestions, bugs' , 
            label_visibility  = 'collapsed' )

st.markdown ('---')
update = st.button ('update user') 

if 'user' not in st.session_state : 
  st.session_state['user']  = user
if update : 
  st.session_state ['user'] = user


st.sidebar.write(f"Hello {st.session_state['user']}")
#with st.expander ('user profile') : 
#  st.write(st.session_state)
