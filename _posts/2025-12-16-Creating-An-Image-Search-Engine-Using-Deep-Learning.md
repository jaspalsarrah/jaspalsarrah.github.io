---
layout: post
title: Creating An Image Search Engine Using Deep Learning
image: "/posts/search_image.jpg"
tags: [Python, Deep Learning, Convolution Neural Network]
---

In this post I'm going to be creating an image search engine using deep learning

---
First I will import the required packages.

```ruby
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from os import listdir
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pickle
```
---
Next I will bring in the pre-trained VGG16 model excluding the top and save this as a file.

```ruby
# image parameters

img_width = 224
img_height = 224
num_channels = 3

# network architecture

vgg = VGG16(input_shape = (img_width, img_height, num_channels), include_top = False, pooling = 'avg')

model = Model(inputs = vgg.input, outputs = vgg.layers[-1].output)
```
I will save this model file.

```ruby
model.save('models/vgg16_search_engine.keras')
```
---
Next I will create the preprocess and featurise functions.

```ruby
# image pre-processing function

def preprocess_image(filepath):
    
    image = load_img(filepath, target_size = (img_width, img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0) # Add new dimension to array which represents to number of images
    image = preprocess_input(image)
    
    return image

# featurise image

def featurise_image(image):
    
    feature_vector = model.predict(image)
    
    return feature_vector
```
---
In the following step I will featurise the base images.

```ruby
# source directory for base images

source_dir = 'data/'

# empty objects to append to

filename_store = []
feature_vector_store = np.empty((0,512))

# pass in & featurise base image set

for image in listdir(source_dir):
    
    print(image)
    
    # append image filename for future lookup
    filename_store.append(source_dir + image)
    
    # preprocess the image
    preprocessed_image = preprocess_image(source_dir + image)
    
    # extract the feature vector
    feature_vector = featurise_image(preprocessed_image)
    
    # append feature vector for similarity calculations
    feature_vector_store = np.append(feature_vector_store, feature_vector, axis = 0)
```
---
The objects created above will be save for future use.

```ruby
pickle.dump(filename_store, open('models/filename_store.p', 'wb'))
pickle.dump(feature_vector_store, open('models/feature_vector_store.p', 'wb'))
```
---
Finally I passed in an image to return similar images.
```ruby
# load in required objects

model = load_model('models/vgg16_search_engine.keras', compile = False)

filename_store = pickle.load(open('models/filename_store.p', 'rb'))

feature_vector_store = pickle.load(open('models/feature_vector_store.p', 'rb'))

# search parameters

search_results_n = 8
search_image = 'search_image_02.jpg'
        
# preprocess & featurise search image

preprocessed_image = preprocess_image(search_image)
search_feature_vector = featurise_image(preprocessed_image)
        
# instantiate nearest neighbours logic

image_neighbors = NearestNeighbors(n_neighbors = search_results_n, metric = 'cosine')

# apply to our feature vector store

image_neighbors.fit(feature_vector_store)

# return search results for search image (distances & indices)

image_distances, image_indices = image_neighbors.kneighbors(search_feature_vector)

# convert closest image indices & distances to lists

image_indices = list(image_indices[0])
image_distances = list(image_distances[0])


# get list of filenames for search results

search_result_files = [filename_store[i] for i in image_indices]
```
---
The images were then plotted using matplotlib.

```ruby
plt.figure(figsize=(12,9))
for counter, result_file in enumerate(search_result_files):    
    image = load_img(result_file)
    ax = plt.subplot(3, 3, counter+1)
    plt.imshow(image)
    plt.text(0, -5, round(image_distances[counter],3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```
---
This is very useful for grocery businesses which wants customers to be able to find similar products.
