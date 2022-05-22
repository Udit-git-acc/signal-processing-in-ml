import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
#import easystart
#hm = easystart.heat_map

directory = 'C:\d drive\design credit\RISE\RISE\output'
fig = plt.figure(figsize=(20,20))
for x in range(8):
    for y in range(8):
        ax = fig.add_subplot(8 ,8 ,8*y+x+1)
        plt.imshow(Image.open(os.path.join(directory,os.listdir(directory)[8*y+x])), interpolation='nearest', aspect='auto')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()

        

