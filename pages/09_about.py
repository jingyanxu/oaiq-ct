
from sys import argv, exit
import random
import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider
import numpy as np
import pandas as pd 
import os
import time 
import streamlit as st
import skimage 
import nibabel as nib

#from PIL import Image
#from streamlit_option_menu import option_menu
#from streamlit_drawable_canvas import st_canvas
#from bokeh.plotting import figure, show

import os
#import streamlit as st

os.environ ['PYTHONINSPECT'] = '1'



rel_path = r".\pages\images_jpeg"
imfpath = os.getcwd () + rel_path 
fname = 'zenodo_summary.jpg'
#st.text ('The images used in human observers studies are obtained from ')

#link='check out this [link](https://retailscope.africa/)'
#url = 'https://zenodo.org/record/4621057/export/hx#.YzxsLLTMKUk'
url = 'https://doi.org/10.5281/zenodo.4621057'
link = 'The images used in human observer studies are obtained from [Zenodo repo]({link})'.format (link = url)
st.markdown(link, unsafe_allow_html=True)
with st.expander ('view sample images') : 
  im = skimage.io.imread  ('{}/{}'.format (imfpath, fname))
  st.image (im)
  st.markdown(
      "The title refers to the patient idx and slice idx.  \n"
      "The images are vertically minimally cropped to maintain the same size.  \n"  
      "red -- pancreas  \n"
      "green -- cyst"
  ) 
st.markdown ('---')  
