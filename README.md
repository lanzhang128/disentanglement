# Overview
TBD
## Dataset
We provide our synthetic datasets used in this paper via Google Drive: https://drive.google.com/file/d/1lXxy7YIirtEIG7fqH4WOOfpS1TgFuvoI/view?usp=sharing. 

To view how we generate synthetic datasets, please dig into the **"Dataset"** folder under this repository.

## Python Files Description
### Environment Requirements
tensorflow==2.4.1, numpy==1.19.5, pandas==1.1.5, scikit-Learn==0.24.1, nltk==3.5

### Training VAEs
Before training a model, please create a **"Dataset"** folder to put datasets in and a **"model"** folder to store models. 
To train a model, run:
```
python training.py -e [embedding dim] -r [RNN dim] -z [latent space dim] -b [batch size] -lr [learning rate] -mt [model type] -zm [coupling method] -beta [beta value for Beta-VAE] -C [C value for CCI-VAE] -s [seed] --epochs [epoch number] --datapath d [dataset path] --mpath [model path]
```
The *dataset path* is the relative path under **Dataset** directory. The *model path* is the relative path under **model** directory, we suggest to use format.

We provide three *model types* in *modeling.py*: 'AE' (AutoEncoder with unidirectional LSTM encoder and decoder); 'VAE' ( Variational AutoEncoder with unidirectional LSTM encoder and decoder); 'BiVAE' ( Variational AutoEncoder with Bidirectional LSTM encoder and Unidirectional LSTM decoder). You can set different *beta* and *C* to obtain a vanilla [VAE](http://arxiv.org/abs/1312.6114), [Beta-VAE](https://openreview.net/forum?id=Sy2fzU9gl), or [CCI-VAE](https://arxiv.org/abs/1804.03599).

We also provide 4 different *coupling method* as illusated in figure below: **0**: Initialisation; **1**: Concatenation; **2**: Initialisation and Concatenation; **3**: Concatenation with out word embeddings.

![coupling method](/figures/coupling.PNG)

### Evaluation of VAEs
We provide both quantitative and qualitative evalutaion for models trained via *training.py*.

For quantitative evaluation, run:
```
python quantity.py -s [seed] --mpath [model path]
```
This will provide basic quantitative evaluation for VAE models including KL, Reconstruction Loss, Active Units. For models trained on synthetic datasets, it can also report disentanglement scores of six disentanglement metrics if you set the *model path* in the format "[dataset]_XXX".

For qualitative evaluation, run:
```
python quality.py -tm [test mode] -s [seed] --mpath [model path]
```
By setting *test mode*, this file can provide two basic qualitative evaluation for VAE models: **0**: mean vector reconstruction and BLEU scores cacluation; **1**: normal and dimension-wise homotopy (a 3D case dimension-wise example is illustrated below).

![homotopy](/figures/homotopy.PNG)

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
