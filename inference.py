import streamlit as st
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from model import AgePrediction

path = './lightning_logs/resnet101_weighted_no_scheduling/17daip11/checkpoints/epoch=11-total-f1score=0.748.ckpt'


@st.experimental_memo
def load_trained_model(model_path):
    model = AgePrediction(None, None, None)
    model.load_from_checkpoint(model_path)
    return model


@st.experimental_memo
def get_predictions(input_image):
    model = load_trained_model(path)
    transforms = Compose([Resize((256, 256)), ToTensor(),
                          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    all_predictions = model(input_image, transforms)
    return all_predictions


if __name__ == '__main__':
    pass