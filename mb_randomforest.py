import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.ops import resources
from tensorflow.contrib.tensor_forest.python import tensor_forest
import time

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

X_train = normal_arr
y_train = label_arr
X_test = norm_test_arr
y_test = test_label_arr


# Parameters
num_steps = 100 # Total steps to train
num_classes = 4
num_features = m-1 
num_trees = 100 
max_nodes = 1000 

tf.reset_default_graph()

# Input and Target placeholders 
X = tf.placeholder(tf.float32, shape=[None, num_features])
Y = tf.placeholder(tf.int64, shape=[None])

# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_classes=num_classes, num_features=num_features, num_trees=num_trees, max_nodes=max_nodes).fill()


# Build the Random Forest

forest_graph = tensor_forest.RandomForestGraphs(hparams)

# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# Measure the accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Initialize the variables (i.e. assign their default value) and forest resources
init_vars = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))


# Start TensorFlow session

sess = tf.Session()

# Run the initializer
sess.run(init_vars)

# Training
time0 = time.time()
for i in range(1, num_steps + 1):

    _, l = sess.run([train_op, loss_op], feed_dict={X: X_train, Y: y_train})

    if i % 50 == 0 or i == 1:

        acc = sess.run(accuracy_op, feed_dict={X: X_train, Y: y_train})

        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))
time1 = time.time()

train_time = time1-time0
print('Test Time:', train_time)

# Test Model
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: X_test, Y: y_test}))


