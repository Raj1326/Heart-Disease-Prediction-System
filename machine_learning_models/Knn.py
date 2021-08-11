#!/usr/bin/env python
# coding: utf-8

# ## Heart Disease Prediction 

# In[128]:


#Importing basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import seaborn as sns # for data visualization
get_ipython().run_line_magic('matplotlib', 'inline')


# In[129]:


#Other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[130]:


#Machine Learning Algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[131]:


# Importing the dataset
dataset = pd.read_csv('dataset.csv')


# In[132]:


dataset.info()


# In[ ]:





# In[133]:


dataset.describe()


# ## Feature Selection

# In[134]:


corrmat = dataset.corr()
top_corr_features = corrmat.index
rcParams['figure.figsize'] = 20,20
#plot heat map
g = sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[135]:


#Visualization of Feature Distribution
dataset.hist()


# In[136]:


#Checking the Dataset is Balanced or not 
rcParams['figure.figsize'] = 8,6
plt.bar(dataset['target'].unique(),dataset['target'].value_counts(),color=['red','green'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')


# ## Data Processing

# In[137]:


#Feature Scaling 
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach','oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])


# In[138]:


dataset.head()


# In[139]:


# Splitting the dataset into the Training set and Test set
x = dataset.drop(['target'], axis = 1)
y = dataset['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


# ## KNN

# In[140]:


# Fitting K-Neighbors Classification to the Training set
knn_scores = []
for k in range(1,16):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(x_train, y_train)
    knn_scores.append(knn_classifier.score(x_test, y_test))


# In[141]:


plt.figure(figsize=(20,10))
plt.plot([k for k in range(1, 16)], knn_scores, color = 'red')
for i in range(1,16):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 16)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')


# In[142]:


# Fitting K-Neighbors Classification to the Training set with k=12
knn_classifier = KNeighborsClassifier(n_neighbors = 12)
knn_classifier.fit(x_train, y_train)


# In[143]:


# Predicting the Test set results
y_pred = knn_classifier.predict(x_test)
#print(y_pred)


# In[144]:


# Accuracy Score
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))


# In[145]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))


# In[146]:


#Interpretation:
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[147]:


#testing
arr = [65,1,0,145,233,1,2,150,1,2.3,3,0,7]
inputFeature = np.asarray(arr).reshape(1, -1)
print(inputFeature.shape)
prediction = knn_classifier.predict(inputFeature)
print(prediction)


# In[148]:


import pickle
with open('knn_model.pkl', 'wb') as file:
    pickle.dump(knn_classifier, file)

