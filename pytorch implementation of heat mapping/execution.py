import numpy
import pandas as pd
import utils
import lrp_vgg16
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
import torch
from PIL import Image
from matplotlib.colors import ListedColormap

torch.cuda.get_device_name()
#tf.config.list_physical_devices('GPU')

directory = 'C:\d drive\design credit\pytorch implementation of heat mapping\image dataset\imagenet-sample-images'
heat_maps = []

def heatmap(R,sx,sy):
    b = 10*((numpy.abs(R)**3.0).mean()**(1.0/3))
    
    my_cmap = plt.cm.seismic(numpy.arange(plt.cm.seismic.N))
    my_cmap[:,0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx,sy))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('off')
    plt.imshow(R,cmap=my_cmap,vmin=-b,vmax=b,interpolation='nearest')
    plt.show()

def plot_pixel_distribution(img_file):
    img = plt.imread(img_file)
    fig = plt.figure(figsize=(16,8))
    fig.add_subplot(1,2,1)
    plt.title("image")
    plt.imshow(img)
    plt.xticks(numpy.array([]))
    plt.yticks(numpy.array([]))
    plt.show()

'''def plot_image_grid(images_files):
    fig = plt.figure(figsize=(8,8))
    images = [tf.contrib.keras.preprocessing.image.load_img(img) for img in images_files]
    for x in range(4):
        for y in range(4):
            ax = fig.add_subplot(4 ,4 ,4*y+x+1)
            plt.imshow(images[4*y+x])
            plt.xticks(numpy.array([]))
            plt.yticks(numpy.array([]))
    plt.show()'''

for filename in os.listdir(directory):
    if filename.endswith(".JPEG"):
        #do smth
        #image = cv2.imread(filename)
        image = Image.open(os.path.join(directory,filename))
        #img_rsz = cv2.resize(image, (224,224))    #resizing according to input dimesion of vgg16
        img_rsz = image.resize((224,224))
        img = numpy.array(img_rsz)[...,::-1]/255.0
        map = lrp_vgg16.lrp_heatmap(img)
        heat_maps.append(map)
        continue
    else:
        continue
#plot_image_grid(heat_maps[:16])
fig = plt.figure(figsize=(20,20))
for rs in heat_maps[:100]:
    b = 10*((numpy.abs(rs)**3.0).mean()**(1.0/3))
    
    my_cmap = plt.cm.seismic(numpy.arange(plt.cm.seismic.N))
    my_cmap[:,0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    for x in range(10):
        for y in range(10):
            ax = fig.add_subplot(10 ,10 ,10*y+x+1)
            plt.imshow(rs,cmap=my_cmap,vmin=-b,vmax=b,interpolation='nearest')
            plt.xticks(numpy.array([]))
            plt.yticks(numpy.array([]))
    plt.show()