import tensorflow as tf
import numpy as np
from random import shuffle

class DataAugmentation():

    def __init__(self):
        pass

    def fit(self, X_train, Y_train, iter=1):
        X_train_aug, Y_train_aug = X_train, Y_train
        for i in range(iter):
            aug = np.empty((len(X_train), 256, 256, 3))
            for index, img in enumerate(X_train):
                img_rand = self.imageRandomize(img)
                aug[index] = img_rand
            X_train_aug = np.concatenate((X_train_aug, aug), axis=0)
            Y_train_aug = np.concatenate((Y_train_aug, Y_train), axis=0)

        index = [i for i in range(len(X_train_aug))]
        shuffle(index)
        X_train_aug = X_train_aug[index]
        Y_train_aug = Y_train_aug[index]

        return X_train_aug, Y_train_aug

    def imageRandomize(self, img):
        img = tf.image.resize(img, [300, 300])
        img = tf.image.random_crop(img, [256, 256, 3])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img

if __name__ == '__main__':
    pass