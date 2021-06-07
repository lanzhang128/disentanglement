# Overview
TBD
## Dataset
We provide our synthetic datasets used in this paper via Google Drive: https://drive.google.com/file/d/1lXxy7YIirtEIG7fqH4WOOfpS1TgFuvoI/view?usp=sharing. 

To view how we generate synthetic datasets, please dig into the **"Dataset"** folder under this repository.

## Python Files Description
### Environment Requirements
tensorflow==2.4.1, numpy==1.19.5, pandas==1.1.5, scikit-Learn==0.24.1, nltk==3.5

### Training
Before training a model, please create a **"Dataset"** folder to put datasets in and a **"model"** folder to store models. 

### Evaluation

### Disentanglement Scores
The disentanglement.py can be used directly to caculate disentanglement scores for representations of test set of synthetic datasets. In order to do this, you need to download synthetic datasets using the link above and put them under **"Dataset"** folder like in this repository.

Run:
```
python disentanglement.py -tm [metric] -s [seed] -d [dataset] -f [filepath]
```
Each *metric* is represented by an integer: 

**0**: metric of [Higgins et al., 2016](https://openreview.net/forum?id=Sy2fzU9gl); **1**: metric of [Kim & Mnih, 2018](http://proceedings.mlr.press/v80/kim18b.html); **2**: metric of [Chen et al., 2018](https://proceedings.neurips.cc/paper/2018/file/1ee3dfcd8a0645a25a35977997223d22-Paper.pdf), **3**: metric of [Ridgeway & Mozer, 2018](https://proceedings.neurips.cc/paper/2018/file/2b24d495052a8ce66358eb576b8912c8-Paper.pdf); **4**: metric of [Eastwood & Williams, 2018](https://openreview.net/forum?id=By-7dz-AZ); **5**: metric of [Kumar et al., 2018](https://openreview.net/forum?id=H1kG7GZAW).

*seed* is the random seed. *dataset* is the name of synthetic datasets: 'toy', 'ynoc', or 'pos'. *filepath* is the path of a *.csv* file which contains representations.

#### Example usage
We present sample *.csv* files of representations under **Examples** folder.
|Metric|ynoc.csv|toy.csv|toy_ideal1.csv|toy_ideal2.csv|
|------|------|------|------|------|
|[Higgins et al., 2016](https://openreview.net/forum?id=Sy2fzU9gl)|42.30%|51.40%|100.00%|100.00%|
|[Kim & Mnih, 2018](http://proceedings.mlr.press/v80/kim18b.html)|31.35%|50.68%|100.00%|100.00%|
|[Chen et al., 2018](https://proceedings.neurips.cc/paper/2018/file/1ee3dfcd8a0645a25a35977997223d22-Paper.pdf)|0.0249|0.0852|0.8105|0.0573|
|[Ridgeway & Mozer, 2018](https://proceedings.neurips.cc/paper/2018/file/2b24d495052a8ce66358eb576b8912c8-Paper.pdf)|0.9998|0.9952|1.0000|1.0000|
|[Eastwood & Williams, 2018](https://openreview.net/forum?id=By-7dz-AZ)|0.0053|0.0107|0.6647|0.6345|
|[Kumar et al., 2018](https://openreview.net/forum?id=H1kG7GZAW)|0.0086|0.0103|0.0468|0.0398|

**Note**: because of the randomness, you may obtain slightly different results with different machines and random seeds.

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
