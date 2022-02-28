'''
COMP-4740 Project 1: Convolutional Neural Network on MNIST Dataset
Submitted by: Saffa Alvi, Nour ElKott 
March 1, 2022 

This file contains the source code for our CNN architecture and 
shows the application of our model to the MNIST dataset.
'''

import tensorflow as tf
import numpy as np
import random

# Public API for tf.keras.datasets.mnist namespace - import TensorFlow and MNIST dataset under the Keras API
mnist = tf.keras.datasets.mnist # 0-9, 28x28 images, 1 colour channel

(X_train, Y_train), (X_test, Y_test) = mnist.load_data() 

print("Train dataset size: ", X_train.shape)
print("Test dataset size: ", X_test.shape)

# visualize dataset
import matplotlib.pyplot as plt
plt.imshow(X_train[0])     # display index 0 of training group as an image
plt.show()    

X_train = tf.keras.utils.normalize(X_train, axis=1) 
X_test = tf.keras.utils.normalize(X_test, axis=1)   

# need to reshape the data as keras needs 4D datasets, and ours are 3D right now
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# new reshaped dataset
print(X_test.shape)
print(X_train.shape)

# build the model
model = tf.keras.models.Sequential() # most common model

# add l2 regularization - not used 
# l2 = tf.keras.regularizers.l2(0.00015)

# add the layers

# hidden layers
model.add(tf.keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu, input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
# add dropout regularization
tf.keras.layers.Dropout(0.30)

model.add(tf.keras.layers.Conv2D(256, (3,3), activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
# add dropout regularization
tf.keras.layers.Dropout(0.40)

# flattens out the input layer
model.add(tf.keras.layers.Flatten()) 

# output layer
# last dense layer must have 10 neurons as we have 10 classes
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
# add dropout regularization
tf.keras.layers.Dropout(0.30)

# compile and fit the model
# add learning rate
opt = tf.keras.optimizers.Adam(learning_rate=0.002)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=3) # epoch is the number of passes through the entire training dataset

# calculate validation loss and accuracy using the test dataset
valLoss, valAcc = model.evaluate(X_test, Y_test)
print(valLoss, valAcc)

# display a summary of the model
model.summary()

predictions = model.predict([X_test])
print(predictions) # prints the probability distributions

# Show the handwritten digit and the model prediction
w, x, y, z = X_test.shape

while(1):
    
    # choose random sample from test dataset
    num1 = random.randint(0, x)
    
    # show sample chosen
    plt.imshow(X_test[num1])
    plt.show()
    
    # show model prediction
    print("Model Prediction: ", np.argmax(predictions[num1]))
    
    # Pause when 'q' is entered
    cont = input('Paused - press ENTER to continue, q to exit: ')
    if cont == 'q':
        break
