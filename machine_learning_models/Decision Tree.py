#!/usr/bin/env python
# coding: utf-8

# ## Heart Disease Prediction 

# In[1]:


#Importing basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import seaborn as sns # for data visualization
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[3]:


#Machine Learning Algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[4]:


# Importing the dataset
dataset = pd.read_csv('dataset.csv')


# In[5]:


dataset.info()


# In[6]:


dataset.describe()


# ## Feature Selection

# In[7]:


corrmat = dataset.corr()
top_corr_features = corrmat.index
rcParams['figure.figsize'] = 20,20
#plot heat map
g = sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[8]:


#Visualization of Feature Distribution
dataset.hist()


# In[9]:


#Checking the Dataset is Balanced or not 
plt.figure(figsize=(8,6))
plt.bar(dataset['target'].unique(),dataset['target'].value_counts(),color=['red','green'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')


# ## Data Processing

# In[10]:


#Convert categorical variables into dummy variables
dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])


# In[11]:


#Feature Scaling 
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach','oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])


# In[12]:


dataset.head()


# In[13]:


# Splitting the dataset into the Training set and Test set
x = dataset.drop(['target'], axis = 1)
y = dataset['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


# ## Decision Tree Classifier

# In[14]:


# Fitting Decision Tree Classification to the Training set
dt_scores = []
for i in range(1, len(x.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)
    dt_classifier.fit(x_train, y_train)
    dt_scores.append(dt_classifier.score(x_test, y_test))


# In[15]:


plt.figure(figsize=(20,15))
plt.plot([i for i in range(1, len(x.columns) + 1)], dt_scores, color = 'green')
for i in range(1, len(x.columns) + 1):
    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))
plt.xticks([i for i in range(1, len(x.columns) + 1)])
plt.xlabel('Max features')
plt.ylabel('Scores')
plt.title('Decision Tree Classifier scores for different number of maximum features')


# In[16]:


# Fitting Decision Tree Classification to the Training set with max_features = 10
dt_classifier = DecisionTreeClassifier(max_features = 10, random_state = 0)
dt_classifier.fit(x_train, y_train)


# In[17]:


# Predicting the Test set results
y_pred = dt_classifier.predict(x_test)


# In[18]:


# Accuracy Score
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))


# In[19]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))


# In[20]:


#Interpretation:
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[23]:


rcParams['figure.figsize'] = 12,6
print("accuracy:",dt_classifier.score(x_test,y_test))
sns.heatmap(confusion_matrix(y_test,y_pred),annot = True)
plt.xlabel("Y_pred")
plt.ylabel("Y_test")
plt.show()


# In[22]:


import pickle
with open('dt_model.pkl', 'wb') as file:
    pickle.dump(dt_classifier, file)


# In[ ]:




