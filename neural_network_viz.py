# This code trains a convolutional neural network on the CIFAR-10 dataset to 
# classify images into 10 categories, and then creates an interactive visualization using the plotly library to display the accuracy and loss over time during training.

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np


# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=50,
                    validation_data=(x_test, y_test),
                    shuffle=True)


# Create a line chart of the accuracy and loss over time
fig = make_subplots(rows=1, cols=2, subplot_titles=('Accuracy', 'Loss'))

fig.add_trace(go.Scatter(x=np.arange(len(history.history['accuracy'])),
                         y=history.history['accuracy'], name='Training'), row=1, col=1)
fig.add_trace(go.Scatter(x=np.arange(len(history.history['val_accuracy'])),
                         y=history.history['val_accuracy'], name='Validation'), row=1, col=1)

fig.add_trace(go.Scatter(x=np.arange(len(history.history['loss'])),
                         y=history.history['loss'], name='Training'), row=1, col=2)
fig.add_trace(go.Scatter(x=np.arange(len(history.history['val_loss'])),
                         y=history.history['val_loss'], name='Validation'), row=1, col=2)

fig.update_layout(title='Training History')

# Show the plot
fig.show()
