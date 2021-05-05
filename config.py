# -*- coding: utf-8 -*-

import os
import random
import sys
class DefaultConfig():

    try:
        model_name = sys.argv[1]
    except:
        print("use default model ResNet34, see config.py")
        model_name = "ResNet34"
    project_dir = os.getcwd()
    source_data_path = os.path.join(project_dir, 'data', 'training_set')
    test_data_path = os.path.join(project_dir,'data', 'testing_set')
    checkpoints = os.path.join(project_dir, 'model')

    label_imgs = {}
    for label in [p for p in os.listdir(source_data_path) if not p.startswith('.')]:
        if label not in label_imgs:
            label_imgs[label] = []
        label_dir = os.path.join(source_data_path, label)
        for img_path in [p for p in os.listdir(label_dir) if not p.startswith('.')]:
            label_imgs[label].append(os.path.join(label_dir, img_path))


    # data_dic = {'train': {}, 'test': {}}
    # train_percent = 0.9
    # for label, paths in label_imgs.items():
    #     random.shuffle(paths)
    #     data_dic['train'][label] = paths[:int(len(paths) * train_percent)]
    #     data_dic['test'][label] = paths[:int(len(paths) * train_percent)]
    with open('train_class_idx.txt', 'w') as f:
        f.write('\n'.join(label_imgs.keys()))
    normal_size = 224
    epochs = 10
    batch_size = 2
    classNumber = len(label_imgs)  # see dataset
    channles = 3  # or 3 or 1 or 6
    lr = 0.001

    lr_reduce_patience = 5
    early_stop_patience = 10

    data_augmentation = False
    monitor = 'val_loss'
    cut = False
    rat = 0.1  # if cut,img[slice(h*self.rat,h-h*self.rat),slice(w*self.rat,w-w*self.rat)]


config = DefaultConfig()