
from __future__ import print_function
from config import config
import sys,copy,shutil
import cv2
import os,time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import numpy as np

import tensorflow as tf
config1 = tf.compat.v1.ConfigProto()
config1.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config1)

from Build_model import Build_model

class PREDICT(Build_model):
    def __init__(self,config):
        Build_model.__init__(self,config)

        try:
            className = sys.argv[2]
        except:
            print("use default className")
            className = "gridiron"
            # className = "organic"

        self.className = className
        self.test_data_path = os.path.join(config.test_data_path,self.className)

    def classes_id(self):
        with open('train_class_idx.txt','r') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        return lines

    def mkdir(self,path):
        if os.path.exists(path):
            return path
        os.mkdir(path)
        return path

    def deal_image(self,img_path,channel=3):
        image = cv2.imread(img_path)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    def Predict(self):
        start = time.time()
        if self.channles==3:
            model = Build_model(self.config).build_model()
        elif self.channles==6:
            model = Build_model(self.config).build_mymodel()
        if os.path.join(os.path.join(self.checkpoints,self.model_name),self.model_name+'.h5'):
            print('weights is loaded')
        else:
            print('weights is not exist')
        model.load_weights(os.path.join(os.path.join(self.checkpoints,self.model_name),self.model_name+'.h5'))

        testset_imgs = {}
        img_names = []
        for label in [p for p in os.listdir(config.test_data_path) if not p.startswith('.')]:
            if label not in testset_imgs:
                testset_imgs[label] = []
            label_dir = os.path.join(config.test_data_path, label)
            for img_path in [p for p in os.listdir(label_dir) if not p.startswith('.')]:
                testset_imgs[label].append(os.path.join(label_dir, img_path))
                img_names.append(img_path)

        data_list, data_list_2, ground_truths_idx, ground_truths = [], [], [], []

        for label, files in testset_imgs.items():
            for file in files:
                if self.channles == 3:
                    img = self.deal_image(file)
                    _, w, h = img.shape[::-1]
                    if self.cut:
                        img = img[slice(int(h * self.rat), int(h - h * self.rat)), slice(int(w * self.rat), int(w - w * self.rat))]
                    img = cv2.resize(img, (self.normal_size, self.normal_size))
                    img = img_to_array(img)
                    data_list.append(img)
                elif self.channles == 6:
                    img_minor, img_major = self.deal_divide_image(file)
                    _, w, h = img_major.shape[::-1]
                    if self.cut:
                        img_major = img_major[slice(int(h * self.rat), int(h - h * self.rat)), slice(int(w * self.rat),int(w - w * self.rat))]
                        img_minor = img_minor[slice(int(h * self.rat), int(h - h * self.rat)), slice(int(w * self.rat),int(w - w * self.rat))]

                    img_major = cv2.resize(img_major, (self.normal_size, self.normal_size))
                    img_minor = cv2.resize(img_minor, (self.normal_size, self.normal_size))
                    img_major = img_to_array(img_major)
                    img_minor = img_to_array(img_minor)
                    data_list.append(img_major)
                    data_list_2.append(img_minor)

                ground_truths.append(label)

        data_list = np.array(data_list, dtype='float32') / 255.0
        if self.channles==6:
            data_list_2 = np.array(data_list_2, dtype='float32') / 255.0


        #img_name_list = []
        i,j,tmp = 0,0,[]
        for img_id, img in enumerate(data_list):
            #img = np.array([img_to_array(img)],dtype='float')/255.0
            if self.channles==3:
                pred = model.predict(data_list[img_id:img_id + 1]).tolist()[0]
            if self.channles==6:
                pred = model.predict([data_list[img_id:img_id+1], data_list_2[img_id:img_id+1]]).tolist()[0]
            label = self.classes_id()[pred.index(max(pred))]
            confidence = max(pred)
            print('image name:',img_names[img_id], '   predict label    is: ',label)
            #print('predict confidect is: ',confidence)

            if label != ground_truths[img_id]:
                print('_____________wrong label______________', label, 'correct:', ground_truths[img_id])
                i+=1
            else:
                j+=1

        accuracy = (1.0*j/ (1.0*len(data_list)))*100.0
        print("accuracy:{:.5}%".format(str(accuracy) ))
        print('Done')
        end = time.time()
        print("usg time:",end - start)

        #with open("accuacy.txt","a") as f:
            #f.write(config.model_name+","+self.className+","+"{:.5}%".format(str(accuracy))+"\n")

def main():
    config.model_name = 'ResNet34-4class6'
    config.channles = 3
    config.normal_size =224
    predict = PREDICT(config)
    predict.Predict()



if __name__=='__main__':
    main()
