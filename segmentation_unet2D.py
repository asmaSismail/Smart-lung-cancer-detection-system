
from tensorflow.keras.optimizers import Adam
from keras_unet_collection import losses
from keras_unet_collection.models import unet_2d
import keras.backend as K
import numpy as np 
import matplotlib.pyplot as plt



SIZE=128
SIZE=(SIZE, SIZE, 1) 
#defining U-net 2D for segmentation
def create_model():
    model = unet_2d(SIZE,filter_num=[64,128,256,512],n_labels=1,stack_num_down=2 ,stack_num_up=2 ,
                activation='ReLU',output_activation='Sigmoid',batch_norm=True, 
                pool='max' , unpool='nearest', name='unet') 
    return model 