from msilib.schema import Directory
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from PIL import Image

class Model():
    def __init__(self):
        K.set_learning_phase(0)
        self.model = ResNet50()
        self.input_size = (224, 224)
        
    def run_on_batch(self, x):
        return self.model.predict(x)

def load_img(path):
    #img = image.load_img(path, target_size=model.input_size)
    img = Image.open(os.path.join(directory,path))
    img_rsz = img.resize((224,224))
    x = np.array(img_rsz)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img_rsz, x

def generate_masks(N, s, p1):
    cell_size = np.ceil(np.array(model.input_size) / s)
    up_size = (s + 1) * cell_size

    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float32')

    masks = np.empty((N, *model.input_size))

    for i in tqdm(range(N), desc='Generating masks'):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x:x + model.input_size[0], y:y + model.input_size[1]]
    masks = masks.reshape(-1, *model.input_size, 1)
    return masks

batch_size = 100

def explain(model, inp, masks):
    preds = []
    #global preds
    # Make sure multiplication is being done for correct axes
    masked = inp * masks
    for i in tqdm(range(0, N, batch_size), desc='Explaining'):
        preds.append(model.run_on_batch(masked[i:min(i+batch_size, N)]))
    preds = np.concatenate(preds)
    sal = preds.T.dot(masks.reshape(N, -1)).reshape(-1, *model.input_size)
    sal = sal / N / p1
    return sal

def class_name(idx):
    return decode_predictions(np.eye(1, 1000, idx))[0][0][1]

model = Model()

N = 2000
s = 8
p1 = 0.5
masks = generate_masks(2000, 8, 0.5)
directory = 'C:\d drive\design credit\pytorch implementation of heat mapping\image dataset'
heat_map = []

count = 0
for filename in os.listdir(directory)[:100]:
    idx = count
    img, x = load_img(os.path.join(directory,filename))
    sal = explain(model, x, masks)
    heat_map.append(sal[count])
    count+=1

fig = plt.figure(figsize=(20,20))
for x in range(10):
    for y in range(10):
        ax = fig.add_subplot(10 ,10 ,10*y+x+1)
        plt.imshow(os.listdir(directory)[10*y+x])
        plt.imshow(heat_map[10*y+x], cmap='jet', alpha=0.5)
        plt.colorbar()
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()


