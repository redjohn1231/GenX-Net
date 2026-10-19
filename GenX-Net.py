import numpy as np
from numpy.linalg import norm

import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras import backend as K

from tensorflow.keras.models import load_model
from image_preprocess import generate_data


class Generation_FeaturesLayer(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(GeneratingFeaturesLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
        self.conv = Conv2D(filters, kernel_size, padding='same', activation='relu')
       
        self.global_pool = GlobalAveragePooling2D()
        
    def get_config(self):
        config = super(GeneratingFeaturesLayer, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
        })
        return config

    def build(self, input_shape):
        super(GeneratingFeaturesLayer, self).build(input_shape)

    def call(self, inputs):
       
        F = self.conv(inputs)
        
       
        M = self.global_pool(F)
        M = tf.expand_dims(tf.expand_dims(M, 1), 1)  
        
       
        F_embedded = F * M 
        
       
        input_shape = tf.shape(F_embedded)  
        batch_size, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        
       
        F_embedded_reshaped = tf.reshape(F_embedded, (batch_size, H * W, C))
        
        
        F_norm = tf.norm(F_embedded_reshaped, axis=-1, keepdims=True)
        cosine_similarity = tf.einsum('bik,bjk->bij', F_embedded_reshaped, F_embedded_reshaped) / (
            tf.matmul(F_norm, F_norm, transpose_b=True) + 1e-8)

        
        weighted_sum = tf.einsum('bij,bjk->bik', cosine_similarity, F_embedded_reshaped)
        
        
        weighted_sum = tf.cast(weighted_sum, tf.float32)
        C = tf.cast(C, tf.float32)
        output = weighted_sum / C
        
        C = tf.cast(C, tf.int32)
       
        output = tf.reshape(output, (batch_size, H, W, C))
        
        return output
    

    def compute_output_shape(self, input_shape):
        return input_shape


def conv(x,c,k,s,d):
    x = SeparableConv2D(c, kernel_size=(k,k), strides=(s,s), padding='same', 
                        dilation_rate=(d, d),
                        depth_multiplier=1,
                        use_bias=True, 
                        depthwise_initializer='glorot_uniform', 
                        pointwise_initializer='glorot_uniform',
                        bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.5)(x)
    return x

class MatMulLayer(Layer):
    def __init__(self, **kwargs):
        super(MatMulLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.matmul(inputs[0], inputs[1])



def multi_scale_module(x):
    x = conv(x,32,3,1,4)
    x = AveragePooling2D(pool_size=(2,2),strides=2)(x)
    
    x = conv(x,64,3,1,4)
    x = AveragePooling2D(pool_size=(2,2),strides=2)(x)
    
    x = conv(x,128,3,1,2)
    x = AveragePooling2D(pool_size=(2,2),strides=2)(x)
    
    x = conv(x,256,3,1,2)
    x = AveragePooling2D(pool_size=(2,2),strides=2)(x)
    
    x = conv(x,256,3,1,1)
    x = AveragePooling2D(pool_size=(2,2),strides=2)(x)
    
    return x

def GFE_branch(x):
    
    M = Generation_FeaturesLayer(filters=512, kernel_size=1)(x)
    
    return M

def Detection_branch(x,M):

    F_h = conv(x,512,1,1,1)
    F_h = conv(F_h,512,3,1,1)
    F_h = AveragePooling2D(pool_size=(2,2),strides=2)(F_h)
    
   
    q = Reshape((F_h.shape[1]*F_h.shape[2],512))(F_h)
    k = Reshape((M.shape[1]*M.shape[2],M.shape[3]))(M)
    v = k
    
    k=Permute((2,1))(k)
    a = MatMulLayer()([q, k])
    b = MatMulLayer()([a, v])
    result = Softmax()(b)
    
    return result


def backbone(x):
    
    x_pf=multi_scale_module(x)
    
    M = GFE_branch(x_pf)
    
    result = Detection_branch(x_pf,M)
    
    flatten = Flatten()(result)
    
    flatten = Dense(128)(flatten)
    result = Dense(9,activation='softmax')(flatten)
    
    return result






