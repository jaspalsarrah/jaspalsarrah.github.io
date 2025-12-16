---
layout: post
title: Fruit Classification Using A Convolutional Neural Network
image: "/posts/CNN.png"
tags: [Python, Deep Learning, Convolution Neural Network]
---

In this post I'm going to be classifying images of fruit using a convolutional neural network.

---
First I will import the required packages.

```ruby
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
from keras import backend as K
import os
```
---
Next I will setup the flow for the training and validation data.

Initially I will begin with the data flow parameters.

```ruby
training_data_dir = 'data/training'
validation_data_dir = 'data/validation'
batch_size = 32
img_width = 128
img_height = 128
num_channels = 3
num_classes = 6
```
For the training and validation I will set up the generators. I have used augmentation to maximise training examples to generalise will and avoid overfitting.

```ruby
training_generator = ImageDataGenerator(rescale = 1./255,
                                        rotation_range = 20,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        zoom_range = 0.1,
                                        horizontal_flip = True,
                                        brightness_range = (0.5, 1.5),
                                        fill_mode = 'nearest')

validation_generator = ImageDataGenerator(rescale = 1./255)
```
---
Next I will set up the flow of images for both training and validation sets.

```ruby
training_set = training_generator.flow_from_directory(directory = training_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')

validation_set = validation_generator.flow_from_directory(directory = validation_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')
```
---
Below is the setup for the architecture network using Keras tuner to identify optimal parameters.

```ruby
def build_model(hp):
    model = Sequential()
    
    # 1st layer
    model.add(Conv2D(filters = hp.Int("Input_Conv_Filters", min_value = 32, max_value = 128, step = 32), kernel_size = (3,3), strides = (1,1), padding = 'same', input_shape = (img_width, img_height, num_channels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    
    for i in range(hp.Int("n_Conv_Layers", min_value = 1, max_value = 3, step = 1)):
    
        model.add(Conv2D(filters = hp.Int(f"Conv_{i}_Filters", min_value = 32, max_value = 128, step = 32), kernel_size = (3,3), strides = (1,1), padding = 'same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D())
    
    model.add(Flatten())
    
    for j in range(hp.Int("n_Dense_Layers", min_value = 1, max_value = 4, step = 1)):
    
        # Dense layer
        model.add(Dense(hp.Int(f"Conv_{j}_Neurons", min_value = 32, max_value = 128, step = 32)))
        model.add(Activation('relu'))
        
        if hp.Boolean("Dropout"):
            model.add(Dropout(0.5))
    
    # Output layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    # compile network
    
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = hp.Choice("Optimizer", values = ['adam', 'RMSProp']),
                  metrics = ['accuracy'])
    
    return model

tuner = RandomSearch(hypermodel = build_model,
                     objective = 'val_accuracy',
                     max_trials = 3,
                     executions_per_trial = 2,
                     directory = 'tuner-results',
                     project_name = 'fruit-cnn',
                     overwrite = True)

tuner.search(x = training_set,
             validation_data = validation_set,
             epochs = 5,
             batch_size = 32)

# top networks
tuner.results_summary()

# best network - hyperparameters
tuner.get_best_hyperparameters()[0].values

# summary of best network architecture
tuner.get_best_models()[0].summary()
```
---
Using the parameters identified above I will use them to set-up the final network architecture.

```ruby
model = Sequential()

# 1st layer
model.add(Conv2D(filters = 96, kernel_size = (3,3), strides = (1,1), padding = 'same', input_shape = (img_width, img_height, num_channels)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

# 2nd layer
model.add(Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())

# 3rd layer
model.add(Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())

# Dense layer
model.add(Dense(160))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# compile network

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# view network architecture

model.summary()
```
---
We are now ready to train our network.
```ruby
num_epochs = 50
model_filename = 'models/fruits_cnn_tuned.keras'

# callbacks

save_best_model = ModelCheckpoint(filepath = model_filename,
                                  monitor = 'val_accuracy',
                                  mode = 'max',
                                  verbose = 1,
                                  save_best_only = True,
                                  save_weights_only=False)

# train the network

history = model.fit(x = training_set,
                    validation_data = validation_set,
                    batch_size = batch_size,
                    epochs = num_epochs,
                    callbacks = [save_best_model])
```
---
We can visualise and validate performance using matplotlib and finding the best epoch performance.

```ruby
import matplotlib.pyplot as plt

# plot validation results
fig, ax = plt.subplots(2, 1, figsize=(15,15))
ax[0].set_title('Loss')
ax[0].plot(history.epoch, history.history["loss"], label="Training Loss")
ax[0].plot(history.epoch, history.history["val_loss"], label="Validation Loss")
ax[1].set_title('Accuracy')
ax[1].plot(history.epoch, history.history["accuracy"], label="Training Accuracy")
ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation Accuracy")
ax[0].legend()
ax[1].legend()
plt.show()

# get best epoch performance for validation accuracy
max(history.history['val_accuracy'])
```
---
Having trained the network we can now make predications on the test set.
```ruby
# import required packages

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
from os import listdir


# parameters for prediction

model_filename = 'models/fruits_cnn_tuned.keras'
img_width = 128
img_height = 128
labels_list = ['apple', 'avocado', 'banana', 'kiwi', 'lemon', 'orange']

# load model

model = load_model(model_filename)

# image pre-processing function

def preprocess_image(filepath):
    
    image = load_img(filepath, target_size = (img_width, img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0) # Add new dimension to array which represents to number of images
    image = image * (1./255)
    
    return image

# image prediction function

def make_prediction(image):
    
    class_probs = model.predict(image)
    predicted_class = np.argmax(class_probs)
    predicted_label = labels_list[predicted_class]
    predicted_prob = class_probs[0][predicted_class]
    
    return predicted_label, predicted_prob

# loop through test data

source_dir = 'data/test/'
folder_names = ['apple', 'avocado', 'banana', 'kiwi', 'lemon', 'orange']
actual_labels = []
predicted_labels = []
predicted_probabilities = []
filenames= []

for folder in folder_names:
    
    images = listdir(source_dir + '/' + folder)
    
    for image in images:
        
        processed_image = preprocess_image(source_dir + '/' + folder + '/' + image)
        predicted_label, predicted_probability = make_prediction(processed_image)
        
        actual_labels.append(folder)
        predicted_labels.append(predicted_label)
        predicted_probabilities.append(predicted_probability)
        filenames.append(image)
        
# create dataframe to analyse

predictions_df = pd.DataFrame({"actual_label" : actual_labels,
                               "predicted_label" : predicted_labels,
                               "predicted_probability" : predicted_probabilities,
                               "filename" : filenames})

predictions_df['correct'] = np.where(predictions_df['actual_label'] == predictions_df['predicted_label'],1,0)
```
---
We can measure the test accuracy using a confusion matrix
```ruby
test_set_accuracy = predictions_df['correct'].sum() / len(predictions_df)
print(test_set_accuracy)

# confusion matrix (raw numbers)

confusion_matrix = pd.crosstab(predictions_df['predicted_label'], predictions_df['actual_label'])
print(confusion_matrix)

# confusion matrix (percentages)

confusion_matrix = pd.crosstab(predictions_df['predicted_label'], predictions_df['actual_label'], normalize = 'columns')
print(confusion_matrix)
```
The result have a 98% accuracy.

This is very useful for grocery businesses which want to automate the weighing of loose fruit and vegetables by idenitfying what has been placed on the scale.
It saves the need for customers to navigate through the menu and creating a queue.
