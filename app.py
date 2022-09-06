import os
import streamlit as st
from PIL import Image

from inference import get_predictions

st.title('Person characteristic prediction Demo')

sample_files = os.listdir('./data/sample_images')
tot_index = len(sample_files)
sample_path = './data/sample_images'

if 'image_index' not in st.session_state:
    st.session_state['image_index'] = 4

if 'which_button' not in st.session_state:
    st.session_state['which_button'] = 'sample_button'

upload_col, sample_col = st.columns(2)
with upload_col:
    uploaded_file = st.file_uploader("Select a picture from your computer(png/jpg) :", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image')
        use_uploaded_image = st.button("Use uploaded_image")
        if use_uploaded_image is True:
            st.session_state['which_button'] = 'upload_button'

with sample_col:
    st.write("Select one from these available samples: ")
    current_index = st.session_state['image_index']
    current_image = Image.open(os.path.join(sample_path, sample_files[current_index]))
    prev = st.button('prev_image')
    next = st.button('next_image')
    if prev:
        current_index = (current_index - 1) % tot_index
    if next:
        current_index = (current_index + 1) % tot_index
    st.session_state['image_index'] = current_index
    sample_image = Image.open(os.path.join(sample_path, sample_files[current_index]))
    st.image(sample_image, caption='Chosen image')

    use_sample_image = st.button("Use this sample")
    if use_sample_image is True:
        st.session_state['which_button'] = 'sample_button'

predict_clicked = st.button("Get prediction")
if predict_clicked:
    which_button = st.session_state['which_button']
    if which_button == 'sample_button':
        predictions = get_predictions(sample_image)
    elif which_button == 'upload_button':
        predictions = get_predictions(img)
