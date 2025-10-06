---
layout: post
title: Compressing Feature Space For Classification Using PCA
image: "/posts/pca.jpg"
tags: [Python, Machine Learning, Principal Component Analysis, Random Forest, Classification]
---

In this post I'm going to be using Principal Component Analysis along with Random Forest Trees to predict whether customers will purchase a music album by a particular artist based on there listening habits and whether they bought the previous album by the same artist.

---
First I will import the required packages.

```ruby
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```
---
Next I will import the sample data, which contains a column on if the last album by the same artist was purchased and a hundred columns with each representing the time spent listening to a particular artitst. However, we do not know which artist as they are just labelled with a numerical suffix.

```ruby
data_for_model = pd.read_csv("data/sample_data_pca.csv")
data_for_model.drop("user_id", axis=1, inplace=True)
```
---
I will shuffle the data to ensure any order is broken
```ruby
data_for_model = shuffle(data_for_model, random_state=42)
```
---
Next I will check for missing values. As there are a low number I have opted to drop them.
```ruby
data_for_model.dropna(how="any", inplace=True)
```
The data will be split into input and output variable(s)
```ruby
X = data_for_model.drop(["purchased_album"], axis=1)
y = data_for_model["purchased_album"]
```
Now I will split the data into training and test sets.
```ruby
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
```
The data will need to be standardised as different features may have different units.
```ruby
scale_standard = StandardScaler()

X_train = scale_standard.fit_transform(X_train)
X_test = scale_standard.transform(X_test)
```
I will apply PCA and set the number of components parameter to none. This is so that I am using all variables to begin with to determine explained variance. At this stage I will also only fit to see the variance.
```ruby

pca = PCA(n_components = None, random_state = 42)
pca.fit(X_train)

explained_variance = pca.explained_variance_ratio_
explained_variance_cumulative = pca.explained_variance_ratio_.cumsum()
```
---
I will now plot the explained variance across components. 
```ruby
pca = PCA(n_components = None, random_state = 42)
pca.fit(X_train)

explained_variance = pca.explained_variance_ratio_
explained_variance_cumulative = pca.explained_variance_ratio_.cumsum()
```
---
From the plot menioned above we can see that around 20-25 components explain approximately 75% of the variance. I will use the figure of 0.75 as the parameter values for n_components when re-instantiating the pca object and apply this to the training and test sets. The last line of code also states the number of columns needed to explain 75% of the variance in our data.
```ruby
pca = PCA(n_components = 0.75, random_state = 42)   # n_components is % covered by variance
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

pca.n_components_
```
---
Next I will build our classifier model. For simplicity I have chosen Random Forest Trees and no other classification techniques as the focus of this post is PCA.
```ruby
clf = RandomForestClassifier(random_state = 42)
clf.fit(X_train, y_train)
```
Finally I will assess the model accuracy using the accuracy score metric. In practice I would also look at precision, recall and F1-score but as stated above the focus for this post is on using PCA for dimension reductionality.
```ruby
y_pred_class = clf.predict(X_test)
accuracy_score(y_test, y_pred_class)
>>> 0.9166666666666666
```
The model has a high classification score and could be used to make predictions. However, interpretability can be difficult.
