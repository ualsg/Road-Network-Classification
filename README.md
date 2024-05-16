
# Classification of Urban Morphology with Deep Learning: Application on Urban Vitality

![Graphical Abstract](./grabs.jpg)

This repository is the official implementation of [Classification of Urban Morphology with Deep Learning: Application on Urban Vitality](https://www.sciencedirect.com/science/article/abs/pii/S0198971521001137). It includes the major codes (written in Python) involved in the paper. We also offer some tractable tutorials in Notebook to show how to use our two modules, `CRHD generator` and `Morphoindex generator`. `CRHD generator` can automatically produce Colored Road Hierarchy Diagram (CRHD) for a given urban area. `Morphoindex generator` can automatically generate both traditional morphological indices based on built environment Shapefiles and road network class probabilities based on our road network classification model.

## Requirements

To use `CRHD generator`, you need to install the requirements:

```setup
pip install osmnx
pip install geopandas
pip install matplotlib
```
To use `Morphoindex generator`, you need to install the additional requirements:

```setup
pip install tensorflow
pip install keras
pip install cv2
pip install numpy
```
If you want to use our Morphoindex generator to calculate road network class probabilities, you should also download `config.py`, `MODEL.py` and `Build_model.py` togehther with `morphoindex_generator.py`, and put them in the same filepath. Also, make sure you have downloaded our pretrained model which you can find below.

## Tutorials
To let you quickly understand how to use our tools, we prepared some easy tutorials for you to have a glance:

[CRHD generator tutorial](https://github.com/ualsg/Road-Network-Classification/blob/main/tutorials/crhd_generator_tutorial.ipynb)

[Morphoindex generator tutorial](https://github.com/ualsg/Road-Network-Classification/blob/main/tutorials/mophoindex_generator_tutorial.ipynb)

## Pre-trained Model

You can download our pretrained models here:

- [Road network classification model](https://drive.google.com/file/d/1N7T9lN4TL5r8EqduZfWv22ROZO4zp_FN/view?usp=sharing) trained on our labelled image set using ResNet-34 architecture, learning rate as 0.0005, batch size as 2. 


## Results

Our model achieves the following performance on the testing set:

**Confusion matrix and ROC curves:**

![image](https://github.com/ualsg/Road-Network-Classification/blob/main/images/results.png)

## Paper

A [paper](https://doi.org/10.1016/j.compenvurbsys.2021.101706) about the work is available.

If you use this work in a scientific context, please cite this article.

Chen W, Wu AN, Biljecki F (2021): Classification of Urban Morphology with Deep Learning: Application on Urban Vitality. Computers, Environment and Urban Systems 90: 101706.

```
@article{2021_ceus_dl_morphology,
  author = {Wangyang Chen and Abraham Noah Wu and Filip Biljecki},
  doi = {10.1016/j.compenvurbsys.2021.101706},
  journal = {Computers, Environment and Urban Systems},
  pages = {101706},
  title = {Classification of Urban Morphology with Deep Learning: Application on Urban Vitality},
  url = {https://doi.org/10.1016/j.compenvurbsys.2021.101706},
  volume = {90},
  year = 2021
}
```

## Contact

[Chen Wangyang](https://ual.sg/authors/wangyang/), [Urban Analytics Lab](https://ual.sg), National University of Singapore, Singapore

