import streamlit as st
from PIL import Image

st.title('Person characteristic prediction Demo')

uploaded_file = st.file_uploader("Select a picture from your computer(png/jpg) :", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image')
