## Problem Statement
Experiment with development, training and hyperparameter tuning of multitask,
multiclass classification model to classify attributes including sex, race, and 
age group.
## Dataset and Preprocessing
The training Dataset used is UTK face dataset.

Source: https://susanqq.github.io/UTKFace/

**Dataset summary**: 
This dataset has wild images for Race, Gender, and age annotated:
Sex : Male, Female
Age : 0 to 116
Race :  White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern)

Some look into the dataset further pre-processing and examples can be found in the
notebook UTKFace_and_wild_dataset_prep.ipynb

## Modelling and training
Information about model architecture used can be found in model.py which uses
ResNet architecture as backbone with different linear layers acting as head for
the respective tasks.

**Fine-tuning and Hyperparameter tuning:**
Pre-trained ResNet model architectures trained on ImageNet are fine-tuned.

Wandb sweep is utilized to perform hyperparameter tuning. The configuration of
tuning is found in sweep_configuration.yaml and logic to run each sweep is present
in train.py

To run the sweep in the terminal execute the following commands:
1. ```wandb sweep --project <wandb-project-name> sweep_configuration.yaml```
2. ```wandb agent <the generated id by above command>```

## Evaluation
Extensive hyperparameter tuning is not done and there is scope of improvement.
Some results can be found [here](https://wandb.ai/nikhilsalodkar/UTK_Age_Prediction/sweeps/zxnuansh?workspace=user-nikhilsalodkar).
## Demo
A demo link build using streamlit and deployed on huggingface is [here](https://huggingface.co/spaces/niks-salodkar/Age-Prediction-Demo).
You might have to restart the huggingface space and wait a short while to try the demo.

You can run your own streamlit app using ```streamlit run app.py``` after necessary packages installation.

## Requirements
The required packages can be viewed in reqs.txt. This file could include extra packages
which might not be necessary. A new conda environment is recommended if you want to
test out on your own.