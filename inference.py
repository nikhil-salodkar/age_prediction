import streamlit as st
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from model import AgePrediction

path = './lightning_logs/UTK_Age_Prediction/796l8aot/checkpoints/epoch=8-total-f1score=0.760.ckpt'
age_dict = {
    0: '0 to 10', 1: '10 to 20', 2: '20 to 30', 3: '30 to 40', 4: '40 to 50', 5: '50 to 60',
    6: '60 to 70', 7: '70 to 80', 8: 'Above 80'
}
sex_dict = {0: 'Male', 1: 'Female'}
race_dict = {
    0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others (like Hispanic, Latino, Middle Eastern etc)'
}


@st.experimental_memo
def load_trained_model(model_path):
    age_weight = torch.tensor([0.7704, 1.5936, 0.3426, 0.6154, 1.2710, 1.2029, 2.2643, 3.8366, 4.7582],
                              dtype=torch.float32)
    sex_weight = torch.tensor([0.9581, 1.0458], dtype=torch.float32)
    race_weight = torch.tensor([0.4716, 1.0567, 1.3464, 1.1974, 2.8132], dtype=torch.float32)
    model = AgePrediction.load_from_checkpoint(model_path, age_weights=age_weight, sex_weights=sex_weight,
                                       race_weights=race_weight)
    return model


def get_predictions(input_image):
    model = load_trained_model(path)
    transforms = Compose([Resize((256, 256)), ToTensor(),
                          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    all_predictions = model(input_image, transforms)

    pred_dict = {
        'Predicted Age range': (age_dict[all_predictions[0][1][0]], age_dict[all_predictions[0][1][1]]),
        'Age Probability': all_predictions[0][0],
        'Predicted Sex': sex_dict[all_predictions[1][0]],
        'Sex Probability': all_predictions[1][1],
        'Predicted Race': (race_dict[all_predictions[2][1][0]], race_dict[all_predictions[2][1][1]]),
        'Race Probability': all_predictions[2][0],
    }
    return pred_dict
