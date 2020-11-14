import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import datetime

raw_data_file = 'C:/Users/hui94/Desktop/AI project/mobile price/train.csv'
raw_data_arr = np.loadtxt(raw_data_file,skiprows = 1,delimiter= ',')
n, m = raw_data_arr.shape

#Normalized Data
mean_ = raw_data_arr.mean(axis = 0)
range_ = np.max(raw_data_arr,axis = 0) - np.min(raw_data_arr,axis = 0)
# normal_arr = (data_arr - mean_)/range_
# normal_arr = normal_arr / 255

#Define first 70% of data as train data
data_arr = raw_data_arr[0:int(n*0.7),0:20]
normal_arr = (data_arr - mean_[0:20])/range_[0:20]

label_arr = raw_data_arr[0:int(n*0.7),-1]

#Define later 30% of data as test data
test_arr = raw_data_arr[int(n*0.7):,0:20]
norm_test_arr = (test_arr - mean_[0:20])/range_[0:20]

test_label_arr = raw_data_arr[int(n*0.7):,-1]

#%%
#Train Neural Network Model
regular_lambda = 0.0001
model = tf.keras.Sequential([
    #tf.keras.layers.Flatten(input_shape = (n,m)),
    tf.keras.layers.Dense(m-1, activation='relu',kernel_regularizer= keras.regularizers.L2(l= regular_lambda)),
    tf.keras.layers.Dense(400, activation='relu',kernel_regularizer= keras.regularizers.L2(l= regular_lambda)),
    tf.keras.layers.Dense(4)])

model.compile(optimizer = 'adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

log_dir = "C:/Users/hui94/Desktop/AI project/digit recognition/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(normal_arr,label_arr, epochs = 20,callbacks=[tensorboard_callback])

#%%
test_loss, test_acc = model.evaluate(norm_test_arr,  test_label_arr, verbose=2)

print('\nTest accuracy %:', test_acc * 100)
