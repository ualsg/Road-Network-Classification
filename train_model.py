
from __future__ import print_function
from config import config
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import keras
import glob,cv2

from keras.preprocessing.image import img_to_array,ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard

import tensorflow as tf
config1 = tf.compat.v1.ConfigProto()
config1.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config1)



import sys
sys.setrecursionlimit(10000)

from Build_model import Build_model

class Train(Build_model):
    def __init__(self,config):
        Build_model.__init__(self,config)

    def get_file(self,path):
        ends = os.listdir(path)[0].split('.')[-1]
        img_list = glob.glob(os.path.join(path , '*.'+ends))
        return img_list

    def deal_image(self,img_path,channel=3):
        image = cv2.imread(img_path)
        gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if channel == 1:
            return gray_img
        thre_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_TRUNC)[1]
        mask = np.where(thre_img > 190)
        image[mask] = 255
        return image

    def deal_divide_image(self, img_path, channel=3):
        image1 = cv2.imread(img_path)
        image2 = image1.copy()
        gray_img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        if channel == 1:
            return gray_img
        thre_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_TRUNC)[1]
        mask1 = np.where(np.logical_or(thre_img < 130, thre_img > 190))
        mask2 = np.where(thre_img > 180)
        image1[mask1], image2[mask2] = 255, 255
        return image1, image2

    def load_data(self):
        label_imgs = config.label_imgs
        images_data ,images_data_2, labels_idx, labels= [],[],[],[]
        for label,files in label_imgs.items():
            for file in files:
                if self.channles == 3:
                    img = self.deal_image(file)
                    _, w, h = img.shape[::-1]
                    if self.cut:
                        img = img[slice(int(h * self.rat), int(h - h * self.rat)), slice(int(w * self.rat), int(w - w * self.rat))]
                    img = cv2.resize(img, (self.normal_size, self.normal_size))
                    img = img_to_array(img)
                    images_data.append(img)
                elif self.channles == 1:
                    img = self.deal_image(file, 0)
                    w, h = img.shape[::-1]
                    if self.cut:
                        img = img[slice(int(h * self.rat), int(h - h * self.rat)), slice(int(w * self.rat), int(w - w * self.rat))]
                    img = cv2.resize(img, (self.normal_size, self.normal_size))
                    img = img_to_array(img)
                    images_data.append(img)
                elif self.channles == 6:
                    img_minor, img_major = self.deal_divide_image(file)
                    _, w, h = img_major.shape[::-1]
                    if self.cut:
                        img_major = img_major[slice(int(h * self.rat), int(h - h * self.rat)), slice(int(w * self.rat), int(w - w * self.rat))]
                        img_minor = img_minor[slice(int(h * self.rat), int(h - h * self.rat)), slice(int(w * self.rat), int(w - w * self.rat))]

                    img_major = cv2.resize(img_major, (self.normal_size, self.normal_size))
                    img_minor = cv2.resize(img_minor, (self.normal_size, self.normal_size))
                    img_major = img_to_array(img_major)
                    img_minor = img_to_array(img_minor)
                    images_data.append(img_major)
                    images_data_2.append(img_minor)

                labels.append(label)

        with open('train_class_idx.txt', 'r') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
            for label in labels:
                idx = lines.index(label.rstrip())
                labels_idx.append(idx)
        images_data = np.array(images_data, dtype='float32') / 255.0
        labels = to_categorical(np.array(labels_idx),num_classes=self.classNumber)
        X_train, X_test, y_train, y_test = train_test_split(images_data,labels,train_size=0.9, random_state=66)
        if self.channles==6:
            images_data_2 = np.array(images_data_2, dtype='float32') / 255.0
            X_train_2, X_test_2, y_train, y_test = train_test_split(images_data_2, labels, train_size=0.9, random_state=66)
            return [X_train, X_train_2], [X_test, X_test_2], y_train, y_test
        return X_train, X_test, y_train, y_test

    def mkdir(self,path):
        if not os.path.exists(path):
            return os.mkdir(path)
        return path

    def train(self,X_train, X_test, y_train, y_test, model):
        print("*"*50)
        print("-"*20+"train",config.model_name+"-"*20)
        print("*"*50)

        tensorboard=TensorBoard(log_dir=self.mkdir(os.path.join(self.checkpoints,self.model_name) ))

        lr_reduce = keras.callbacks.ReduceLROnPlateau(monitor=config.monitor,
                                                      factor=0.1,
                                                      patience=config.lr_reduce_patience,
                                                      verbose=1,
                                                      mode='auto',
                                                      cooldown=0)
        early_stop = keras.callbacks.EarlyStopping(monitor=config.monitor,
                                                   min_delta=0,
                                                   patience=config.early_stop_patience,
                                                   verbose=1,
                                                   mode='auto')
        checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(self.mkdir( os.path.join(self.checkpoints,self.model_name) ),self.model_name+'.h5'),
                                                     monitor=config.monitor,
                                                     verbose=1,
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     mode='auto',
                                                     period=1)

        if self.data_augmentation:
            print("using data augmentation method")
            data_aug = ImageDataGenerator(
                rotation_range=5,  # 图像旋转的角度
                width_shift_range=0.2,  # 左右平移参数
                height_shift_range=0.2,  # 上下平移参数
                zoom_range=0.3,  # 随机放大或者缩小
                horizontal_flip=True,  # 随机翻转
            )

            data_aug.fit(X_train)
            model.fit_generator(
                data_aug.flow(X_train, y_train, batch_size=config.batch_size),
                steps_per_epoch=X_train.shape[0] // self.batch_size,
                validation_data=(X_test, y_test),
                shuffle=True,
                epochs=self.epochs, verbose=1, max_queue_size=1000,
                callbacks=[early_stop,checkpoint,lr_reduce,tensorboard],
            )
        else:
            model.fit(x=X_train,y=y_train,
                      batch_size=self.batch_size,
                      validation_data=(X_test,y_test),
                      epochs=self.epochs,
                      callbacks=[early_stop,checkpoint,lr_reduce,tensorboard],
                      shuffle=True,
                      verbose=1)

    def start_train(self):
        X_train, X_test, y_train, y_test=self.load_data()
        if self.channles==6:
            model = Build_model(config).build_mymodel()
        else:
            model = Build_model(config).build_model()
        self.train(X_train, X_test, y_train, y_test, model)

    def remove_logdir(self):
        self.mkdir(self.checkpoints)
        self.mkdir(os.path.join(self.checkpoints,self.model_name))
        events = os.listdir(os.path.join(self.checkpoints,self.model_name))
        for evs in events:
            if "events" in evs:
                os.remove(os.path.join(os.path.join(self.checkpoints,self.model_name),evs))

    def mkdir(self,path):
        if os.path.exists(path):
            return path
        os.mkdir(path)
        return path


def main():
    config.model_name = 'ResNet-34'
    config.lr_reduce_patience = 3
    config.channles = 6
    config.batch_size = 2
    config.epochs = 40
    config.lr = 0.0005
    train = Train(config)
    train.remove_logdir()
    train.start_train()
    print('-----Done------')

if __name__=='__main__':
    main()
    #print(config.label_imgs)

