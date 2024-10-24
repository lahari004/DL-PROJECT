import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
class PreProcess_Data:
    def visualization_images(self, dir_path, nimages):
        fig, axs = plt.subplots(4, 4, figsize=(8, 8))
        dpath = dir_path
        count = 0
        for i in os.listdir(dpath):
            train_class = os.listdir(os.path.join(dpath, i))
            for j in range(nimages):
                img = os.path.join(dpath, i, train_class[j])
                img = cv2.imread(img)
                axs[count][j].title.set_text(i)
                axs[count][j].imshow(img)
            count += 1
            fig.tight_layout()
        plt.show(block=True)
    def preprocess(self, dir_path):
        dpath = dir_path
        train = []
        label = []
        for i in os.listdir(dpath):
            train_class = os.listdir(os.path.join(dpath, i))
            for j in train_class:
                img = os.path.join(dpath, i, j)
                train.append(img)
                label.append(i)
        print('Number of train images : {}\n'.format(len(train)))
        print('Number of train images labels: {}\n'.format(len(label)))
        project_df = pd.DataFrame({'Image': train, 'Labels': label})
        print(project_df)
        return project_df, train, label