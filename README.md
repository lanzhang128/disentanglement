# Overview
TBD
## Dataset
We provide our synthetic datasets used in this paper via Google Drive: https://drive.google.com/file/d/1lXxy7YIirtEIG7fqH4WOOfpS1TgFuvoI/view?usp=sharing. 

To view how we generate synthetic datasets, please dig into the dataset folder under this project.

## Python Files Description
### Environment Requirements
tensorflow==2.4.1, numpy==1.19.5, pandas==1.1.5, scikit-Learn==0.24.1, nltk==3.5

### Training

### Evaluation

### Disentanglement Scores

Please type "python XXX.py -h" for usage. To train VAEs, please create a "Dataset" folder to put datasets in and a "model" folder to store models.
#### training.py
VAE training. The dataset path is the relative path under Dataset directory. The trained model path is going to be the relative path under model directory.
#### modeling.py
Model Architecture with basic training and test methods.
#### quantity.py
Basic quantitative evaluation for VAE models including KL, Reconstruction Loss, Active Units. For models trained on synthetic datasets, it will also report disentanglement scores.
#### quality.py
Basic qualitative evaluation for VAE models including mean vector reconstruction and homotopy.
#### disentanglement.py
Using disentanglement metrics to caclulate disentanglement scores for representations of synthetic datasets.
#### ideal_generation.py
Using ideal representations of toy dataset to train and evaluate a LSTM generator.
