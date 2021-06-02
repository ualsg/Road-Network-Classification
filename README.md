
# Classification of Urban Morphology with Deep Learning: Application on Urban Vitality

This repository is the official implementation of [Classification of Urban Morphology with Deep Learning: Application on Urban Vitality](https://arxiv.org/abs/2105.09908). It includes the major codes (written in Python) involved in the paper. We also offer some tractable tutorials in Notebook to show how to use our two modules, `CRHD generator` and `Morphoindex generator`. `CRHD generator` can automatically produce Colored Road Hierarchy Diagram (CRHD) for a given urban area. `Morphoindex generator` can automatically generate both traditional morphological indices based on built environment Shapefiles and road network class probabilities based on our road network classifier.

## Requirements

To use our `CRHD generator`, you need to install the requirements:

```setup
pip install osmnx
pip install geopandas
pip install matplotlib
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
