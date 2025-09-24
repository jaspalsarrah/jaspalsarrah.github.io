---
layout: post
title: Predicting whether a Customer is likely to sign up to promotion following marketing campaign Using Classification techniques
image: "/posts/ofm-new-customer-campaign.jpg"
tags: [Python, Machine Learning, Decision Tree Classification, Random Forest Classification, Logistic Regression, K-Nearest-Neighbours (KNN) for Classification]
---

In this post I'm going to be using the machine learning algorithms Logistic Regression, Decision Tree Classification and Random Forest Classification and K-Nearest-Neighbours (KNN) for Classification to find an accurate model to predict customer signup rates following a marketing campaign. This campaign has run before but the business want to target mailers at customers likely to sign up rather than all customers to minimise cost. Therefore, we need to identify those customers that are likely to sign up based on the customer data available and previous sign up rates.

---

First I will import the required packages for the Logistic Regression model.

```ruby
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV
```
---

Next I will import the data, which is already saved as a pickle file.
```ruby
data_for_model = pd.read_pickle(open("data/abc_classification_modelling.p", "rb"))
```
After looking at the data, I can see the customer_id field will need to be dropped. I will also shuffle the data to avoid ensure any ordering that might be present does not affect the spread of data between training and test sets.
```ruby
data_for_model.drop("customer_id", axis=1, inplace=True)
```
After looking at the data, I can see the customer_id field will need to be dropped.
```ruby
data_for_model = shuffle(data_for_model)
```
I will look at the balance of the output variable signup_flag to assess whether there is an imbalance.
```ruby
data_for_model["signup_flag"].value_counts(normalize = True)
>>>      signup_flag
>>> 0    0.689535
>>> 1    0.310465
```
The data shows there is a significant difference between those that have signed up previoulsy and those that did not. This will need to be considered when assessing the model.

There are a low number of missing values therefore rather than impute values the rows with missing data will be dropped.
```ruby
data_for_model.isna().sum()
data_for_model.dropna(how="any", inplace=True)
```
The final step of the data preparation involves dealing with outliers. I will use the boxplot approach and remove them.
```ruby
outlier_investigation = data_for_model.describe()

outlier_columns = ["distance_from_store", "total_sales", "total_items"]

for column in outlier_columns:
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2  # Changed to 2 to stop eliminating too many
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    data_for_model.drop(outliers, inplace = True)
>>> 8 outliers detected in column distance_from_store
>>> 54 outliers detected in column total_sales
>>> 3 outliers detected in column total_items
```
Next I will split the input and output variables.
```ruby
X = data_for_model.drop(["signup_flag"], axis=1)
y = data_for_model["signup_flag"]
```
The data sets will now be split into training and test sets. I have used the stratify parameter to have a proportional split of customers that have previously signed up.
```ruby
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)
```
There one categorical variable. To deal with this I will be using One Hot Encoding. One of the encoded columns will be dropped to avoid the dummy variable trap. I would also like to highlight the encoding rules will be learned on the training data set and applied to all other data sets.
```ruby
categorical_vars = ["gender"]

one_hot_encoder = OneHotEncoder(sparse_output=False, drop = "first")

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])  
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)
X_train.drop(categorical_vars, axis=1, inplace=True) # Drops Categorical variable columns

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1)  
X_test.drop(categorical_vars, axis=1, inplace=True)
```
Feature Selection - I will use recursive feature elimination with cross validation to find the optimal number of features and drop any others. From the code below we can see the ideal number is 7 and one column, which is total_sales has been dropped.
```ruby
clf = LogisticRegression(max_iter = 1000)
feature_selector = RFECV(clf)

fit = feature_selector.fit(X_train,y_train)

optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features: {optimal_feature_count}")

X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()]

plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()
>>> Optimal number of features: 7
```
Next I will train the model
```ruby
clf = LogisticRegression(max_iter = 1000)
clf.fit(X_train, y_train)
```
Next I will train the model
```ruby
clf = LogisticRegression(max_iter = 1000)
clf.fit(X_train, y_train)
```
Finally we will assess the model. Using the trained model on the test data set we will get a set of predictions. We can also split the predictions giving the probability of each data point falling into one of the two classes (i.e. 0 or 1). I have kept the second column only which gives the probailities the customer signed up for delivery.
```ruby
y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:,1] 
```
I will use a confusion matrix to show the splits between true positives, false positives, false negatives and true negatives.
```ruby
conf_matrix = confusion_matrix(y_test, y_pred_class)

plt.style.use("seaborn-v0_8-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i, j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()
```
The first model assessment I will carry out is classification accuracy, which have a very good score. However, earlier I showed there was an uneven split between the output variable signup_flag. In such instances classification accuracy can be misleading when the data is heavily biased towards one class.
```ruby
accuracy_score(y_test, y_pred_class)
>>> 0.8662420382165605
```
Due to the uneven split I will calculate the F1-Score, which is the harmonic mean between precision and recall scores. This will provide a more balanced model score.
```ruby
precision_score(y_test, y_pred_class)
>>> 0.7837837837837838
recall_score(y_test, y_pred_class)
>>> 0.6904761904761905
f1_score(y_test, y_pred_class)
>>> 0.7341772151898734
```
Whilst lower than the classification accuracy, the F1-Score provide a more accurate model score.

Next I will apply the Decision Tree for Classification to the same data set. I will import the required packages.
```ruby
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
```
Next I will import the data, which is already saved as a pickle file.
```ruby
data_for_model = pd.read_pickle(open("data/abc_classification_modelling.p", "rb"))
```
After looking at the data, I can see the customer_id field will need to be dropped. I will also shuffle the data to avoid ensure any ordering that might be present does not affect the spread of data between training and test sets.
```ruby
data_for_model.drop("customer_id", axis=1, inplace=True)
```
After looking at the data, I can see the customer_id field will need to be dropped.
```ruby
data_for_model = shuffle(data_for_model)
```
I will look at the balance of the output variable signup_flag to assess whether there is an imbalance.
```ruby
data_for_model["signup_flag"].value_counts(normalize = True)
>>>      signup_flag
>>> 0    0.689535
>>> 1    0.310465
```
The data shows there is a significant difference between those that have signed up previoulsy and those that did not. This will need to be considered when assessing the model.

There are a low number of missing values therefore rather than impute values the rows with missing data will be dropped.
```ruby
data_for_model.isna().sum()
data_for_model.dropna(how="any", inplace=True)
```
This model is not affected by outliers therefore I will not remove any. The data will be split into input variables and output variable.
```ruby
X = data_for_model.drop(["signup_flag"], axis=1)
y = data_for_model["signup_flag"]
```
I will split data between training and test sets proportionally on the output variable using the stratify parameter.
```ruby
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
```
One hot encoding will be used to replace the only categorical variable with a binary output.
```ruby
categorical_vars = ["gender"]

one_hot_encoder = OneHotEncoder(sparse_output=False, drop = "first")

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])  
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)
X_train.drop(categorical_vars, axis=1, inplace=True) # Drops Categorical variable columns

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1)  
X_test.drop(categorical_vars, axis=1, inplace=True)
```
Before training the model I will find the optimal max_depth to use as the parameter for training the model.
```ruby
max_depth_list = list(range(1,15))
accuracy_scores = []

for depth in max_depth_list:
    clf = DecisionTreeClassifier(max_depth = depth, random_state = 42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test, y_pred) #Using f1 as data imbalance
    accuracy_scores.append(accuracy)

max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_depth = max_depth_list[max_accuracy_idx]

plt.plot(max_depth_list, accuracy_scores)
plt.scatter(optimal_depth, max_accuracy, marker = "x", color = "red")
plt.title(f"Accuracy (F1 Score) by Max Depth \n OptimalTree Depth: {optimal_depth} (Accuracy: {round(max_accuracy,4)})")
plt.xlabel("Max depth of decision tree")
plt.ylabel("Accuracy (F1 Score)")
plt.tight_layout()
plt.show()
>>> 9
```
The values of 9 will be used for the parameter max_depth when training the model.
```ruby
clf = DecisionTreeClassifier(max_depth = 9)
clf.fit(X_train, y_train)
```
Finally I will assess the model. Using the trained model on the test data set we will get a set of predictions. We can also split the predictions giving the probability of each data point falling into one of the two classes (i.e. 0 or 1). I have kept the second column only which gives the probailities the customer signed up for delivery.
```ruby
y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:,1] 
```
I will use a confusion matrix to show the splits between true positives, false positives, false negatives and true negatives.
```ruby
conf_matrix = confusion_matrix(y_test, y_pred_class)

plt.style.use("seaborn-v0_8-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i, j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()
```
The first model assessment I will carry out is classification accuracy, which have a very good score. However, earlier I showed there was an uneven split between the output variable signup_flag. In such instances classification accuracy can be misleading when the data is heavily biased towards one class.
```ruby
accuracy_score(y_test, y_pred_class)
>>> 0.9529411764705882
```
Due to the uneven split I will calculate the F1-Score, which is the harmonic mean between precision and recall scores. This will provide a more balanced model score.
```ruby
precision_score(y_test, y_pred_class)
>>> 0.9074074074074074
recall_score(y_test, y_pred_class)
>>> 0.9423076923076923
f1_score(y_test, y_pred_class)
>>> 0.9245283018867925
```
Whilst slightly lower than the classification accuracy, the F1-Score provide a more accurate model score, which is very good. It is also significantly higher than the logisitc regression model.

Next I will apply the Random Forest for Classification to the same data set. I will import the required packages.
```ruby
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
```
Next I will import the data, which is already saved as a pickle file.
```ruby
data_for_model = pd.read_pickle(open("data/abc_classification_modelling.p", "rb"))
```
After looking at the data, I can see the customer_id field will need to be dropped. I will also shuffle the data to avoid ensure any ordering that might be present does not affect the spread of data between training and test sets.
```ruby
data_for_model.drop("customer_id", axis=1, inplace=True)
```
After looking at the data, I can see the customer_id field will need to be dropped.
```ruby
data_for_model = shuffle(data_for_model)
```
I will look at the balance of the output variable signup_flag to assess whether there is an imbalance.
```ruby
data_for_model["signup_flag"].value_counts(normalize = True)
>>>      signup_flag
>>> 0    0.689535
>>> 1    0.310465
```
The data shows there is a significant difference between those that have signed up previously and those that did not. This will need to be considered when assessing the model.

There are a low number of missing values therefore rather than impute values the rows with missing data will be dropped.
```ruby
data_for_model.isna().sum()
data_for_model.dropna(how="any", inplace=True)
```
This model is not affected by outliers therefore I will not remove any. The data will be split into input variables and output variable.
```ruby
X = data_for_model.drop(["signup_flag"], axis=1)
y = data_for_model["signup_flag"]
```
I will split data between training and test sets proportionally on the output variable using the stratify parameter.
```ruby
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
```
One hot encoding will be used to replace the only categorical variable with a binary output.
```ruby
categorical_vars = ["gender"]

one_hot_encoder = OneHotEncoder(sparse_output=False, drop = "first")

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])  
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)
X_train.drop(categorical_vars, axis=1, inplace=True) # Drops Categorical variable columns

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1)  
X_test.drop(categorical_vars, axis=1, inplace=True)
```
Next I will train the model. I have set the parameters such that the number of trees is set to 500 and the number of random input variables is set to 5 as the default is all and could lead to overfitting.
```ruby
clf = RandomForestClassifier(n_estimators = 500, max_features = 5)
clf.fit(X_train, y_train)
```
Finally I will assess the model. Using the trained model on the test data set we will get a set of predictions. We can also split the predictions giving the probability of each data point falling into one of the two classes (i.e. 0 or 1). I have kept the second column only which gives the probailities the customer signed up for delivery.
```ruby
y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:,1] 
```
I will use a confusion matrix to show the splits between true positives, false positives, false negatives and true negatives.
```ruby
conf_matrix = confusion_matrix(y_test, y_pred_class)

plt.style.use("seaborn-v0_8-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i, j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()
```
The first model assessment I will carry out is classification accuracy, which have a very good score. However, earlier I showed there was an uneven split between the output variable signup_flag. In such instances classification accuracy can be misleading when the data is heavily biased towards one class.
```ruby
accuracy_score(y_test, y_pred_class)
>>> 0.9352941176470588
```
Due to the uneven split I will calculate the F1-Score, which is the harmonic mean between precision and recall scores. This will provide a more balanced model score.
```ruby
precision_score(y_test, y_pred_class)
>>> 0.8867924528301887
recall_score(y_test, y_pred_class)
>>> 0.9038461538461539
f1_score(y_test, y_pred_class)
>>> 0.8952380952380953
```
Whilst slightly lower than the decision tree for classification, the Random Forest also provides a very strong model to predict sign up rates.

I will also carried out permutation importance.
```ruby
result = permutation_importance(clf, X_test, y_test, n_repeats = 10, random_state = 42)
print(result)

permutation_importance = pd.DataFrame(result["importances_mean"])
feature_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_names,permutation_importance], axis = 1)
permutation_importance_summary.columns = ["input_variable", "permutation_importance"]
permutation_importance_summary.sort_values(by = "permutation_importance", inplace = True)

plt.barh(permutation_importance_summary["input_variable"],permutation_importance_summary["permutation_importance"])
plt.title("Permutation importance of Random Forest")
plt.xlabel("Permutation Importance")
plt.show()
```
There were some interesting results. The distance from store was the input variable that had the most significant impact.
However, there were some with a negative value implying the shuffled data performed better than the actual data. This would mean these 3 variables have no impact on thr output and could be removed from the data for modelling to save on computation.

The last algorithm for I will be be using for modelling is K-Nearest-Neighbours (KNN) for Classification. I will import the required packages.
```ruby
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import RFECV
```
Next I will import the data, which is already saved as a pickle file.
```ruby
data_for_model = pd.read_pickle(open("data/abc_classification_modelling.p", "rb"))
```
After looking at the data, I can see the customer_id field will need to be dropped. I will also shuffle the data to avoid ensure any ordering that might be present does not affect the spread of data between training and test sets.
```ruby
data_for_model.drop("customer_id", axis=1, inplace=True)
```
After looking at the data, I can see the customer_id field will need to be dropped.
```ruby
data_for_model = shuffle(data_for_model)
```
I will look at the balance of the output variable signup_flag to assess whether there is an imbalance.
```ruby
data_for_model["signup_flag"].value_counts(normalize = True)
>>>      signup_flag
>>> 0    0.689535
>>> 1    0.310465
```
The data shows there is a significant difference between those that have signed up previously and those that did not. This will need to be considered when assessing the model.

There are a low number of missing values therefore rather than impute values the rows with missing data will be dropped.
```ruby
data_for_model.isna().sum()
data_for_model.dropna(how="any", inplace=True)
```
The final step of the data preparation involves dealing with outliers. I will use the boxplot approach and remove them.
```ruby
outlier_investigation = data_for_model.describe()

outlier_columns = ["distance_from_store", "total_sales", "total_items"]

for column in outlier_columns:
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2  # Changed to 2 to stop eliminating too many
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    data_for_model.drop(outliers, inplace = True)
>>> 8 outliers detected in column distance_from_store
>>> 54 outliers detected in column total_sales
>>> 3 outliers detected in column total_items
```
The data will be split into input variables and output variable.
```ruby
X = data_for_model.drop(["signup_flag"], axis=1)
y = data_for_model["signup_flag"]
```
I will split data between training and test sets porportionally on the output variable using the stratify parameter.
```ruby
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
```
One hot encoding will be used to replace the only categorical variable with a binary output.
```ruby
categorical_vars = ["gender"]

one_hot_encoder = OneHotEncoder(sparse_output=False, drop = "first")

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])  
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)
X_train.drop(categorical_vars, axis=1, inplace=True) # Drops Categorical variable columns

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1)  
X_test.drop(categorical_vars, axis=1, inplace=True)
```
This algorithm focuses on the distance between points therefore I will be using feature scaling to scale values to the same scale using normalisation.
```ruby
scale_norm = MinMaxScaler()
X_train = pd.DataFrame(scale_norm.fit_transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(scale_norm.transform(X_test), columns = X_test.columns)
```
Using Random Forest I will find the optimal number of features and drop any others. Two columns have been dropped as they reduce the accuracy.
```ruby
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state = 42)
feature_selector = RFECV(clf)

fit = feature_selector.fit(X_train,y_train)

optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features: {optimal_feature_count}")

X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()]

plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()
```
Before training the model I will find the optimal n_neighbors parameter.
```ruby
k_list = list(range(2,25))
accuracy_scores = []

for k in k_list:
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test, y_pred) #Using f1 as data imbalance
    accuracy_scores.append(accuracy)

max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_k_value = k_list[max_accuracy_idx]


# Plot of max depths

plt.plot(k_list, accuracy_scores)
plt.scatter(optimal_k_value, max_accuracy, marker = "x", color = "red")
plt.title(f"Accuracy (F1 Score) by K \n Optimal Value for K: {optimal_k_value} (Accuracy: {round(max_accuracy,4)})")
plt.xlabel("K")
plt.ylabel("Accuracy (F1 Score)")
plt.tight_layout()
plt.show()
>>> 5
```
Next I will train the model. The optimal n_neighbors is 5 and the default in sci-lit learn is also 5. Therefore I do not need to explicitly set the parameter.
```ruby
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
```
Finally I will assess the model. Using the trained model on the test data set we will get a set of predictions. We can also split the predictions giving the probability of each data point falling into one of the two classes (i.e. 0 or 1). I have kept the second column only which gives the probailities the customer signed up for delivery.
```ruby
y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:,1] 
```
I will use a confusion matrix to show the splits between true positives, false positives, false negatives and true negatives.
```ruby
conf_matrix = confusion_matrix(y_test, y_pred_class)

plt.style.use("seaborn-v0_8-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i, j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()
```
The first model assessment I will carry out is classification accuracy, which have a very good score. However, earlier I showed there was an uneven split between the output variable signup_flag. In such instances classification accuracy can be misleading when the data is heavily biased towards one class.
```ruby
accuracy_score(y_test, y_pred_class)
>>> 0.9363057324840764
```
Due to the uneven split I will calculate the F1-Score, which is the harmonic mean between precision and recall scores. This will provide a more balanced model score.
```ruby
precision_score(y_test, y_pred_class)
>>> 1.0
recall_score(y_test, y_pred_class)
>>> 0.7619047619047619
f1_score(y_test, y_pred_class)
>>> 0.8648648648648649
```
Whilst slightly lower than the Decision Tree and Random Forest for classification, the KNN algrorithm also provides a very strong model to predict sign up rates. However, in this instance Decision Tree for Classification is optimal.

