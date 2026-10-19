import numpy as np
from numpy.linalg import norm

import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras import backend as K

from image_preprocess_1 import generate_data_1
from keras.callbacks import ReduceLROnPlateau

from network import backbone, Generation_FeaturesLayer, MatMulLayer
from Setting import data
from Setting import hyperparameters

from training_callbacks import ConvergenceCallback


inputs = Input(shape=(hyperparameters.height, hyperparameters.width, 3))
output = backbone(inputs)
model = keras.Model(inputs, output)
model.summary()

initial_lr = hyperparameters.lr

model.compile(optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.8,
                              patience=1,
                              mode='auto',
                              verbose=1,
                              min_lr=1e-6)



convergence_callback = ConvergenceCallback(initial_lr=initial_lr,
                                           loss_threshold=1e-2,
                                           consecutive_epochs=5,
                                           lr_ratio=0.1)

model.fit(generate_data_1(data.path_1,data.label,
                        height=hyperparameters.height,
                        width=hyperparameters.width,
                        batch_size=hyperparameters.batch_size_t,
                        num=hyperparameters.num),
          steps_per_epoch=hyperparameters.steps_t,
          epochs=hyperparameters.epochs,

          validation_data=generate_data_1(data.path_2,data.label,
                                      height=hyperparameters.height,
                                      width=hyperparameters.width,
                                      batch_size=hyperparameters.batch_size_v,
                                      num=hyperparameters.num ),
          validation_steps=hyperparameters.steps_v,

          callbacks=[reduce_lr,
                     convergence_callback])

model.save("./test.h5")





