import tensorflow as tf
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model



def conv2d_block(input_tensor, n_filters, kernel_size = 3, pooling=False, deconv=False):    
    x = Conv2D(n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer = 'he_normal', padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(n_filters, kernel_size = (kernel_size, kernel_size),kernel_initializer = 'he_normal', padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    if pooling == True:
        p = MaxPool2D((2, 2))(x)
        p = tf.keras.layers.Dropout(0.3)(p)
        return x, p
    elif deconv == True:
        d = tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size = (3, 3),strides = 2, padding = 'same')(x)
        return d
    else:
        return x

def encoder(inputs): 
  f1, p1 = conv2d_block(inputs, n_filters=64,kernel_size=3,pooling=True)  
  print(p1)
  f2, p2 = conv2d_block(p1, n_filters=128,kernel_size=3,pooling=True)
  print(p2)  
  f3, p3 = conv2d_block(p2, n_filters=256,kernel_size=3,pooling=True)  
  print(p3)
  f4, p4 = conv2d_block(p3, n_filters=512,kernel_size=3,pooling=True)  
  print(p4)
  return p4, (f1, f2, f3, f4)



def bottleneck(inputs):  
  bottle_neck = conv2d_block(inputs, n_filters=1024,deconv = True)   
  print(bottle_neck)  
  return bottle_neck

def deconv2d_block(input_tensor, n_filters=64, kernel_size = 3, deconv=False):    
    
    x = Conv2D(n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer = 'he_normal', padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)    
    n_filters=n_filters/2    
    x = Conv2D(n_filters, kernel_size = (kernel_size, kernel_size),kernel_initializer = 'he_normal', padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)    
    
    if deconv == True:
        d = tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size = (3, 3),strides = 2, padding = 'same')(x)
        return d
    else:
        return x

def decoder_block(inputs, conv_output, n_filters=64, kernel_size=3, strides=3, dropout=0.3): 
  c = tf.keras.layers.concatenate([inputs, conv_output])   
  d = deconv2d_block(c, n_filters=n_filters, kernel_size=3,deconv= True)  
  return d


def decoder(inputs, convs, output_channels):  
  f1, f2, f3, f4 = convs
  f6 = decoder_block(inputs, f4, n_filters=1024,strides=2)
  print(f6)
  f7 = decoder_block(f6, f3,n_filters=512,strides=2)
  print(f7)
  f8 = decoder_block(f7, f2, n_filters=256,strides=2)
  print(f8)  
  f9 = decoder_block(f7, f2, n_filters=128,strides=2)
  print(f9)  
  
  outputs = tf.keras.layers.Conv2D(output_channels, kernel_size=1, activation='softmax')(f9)
  print(outputs )
  return outputs

def Unet(): 
  inputs = tf.keras.layers.Input(shape=(128, 128,3,))
  encoder_output, convs = encoder(inputs)
  bottle_neck = bottleneck(encoder_output)  
  outputs = decoder(bottle_neck, convs, output_channels=3)  
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  return model