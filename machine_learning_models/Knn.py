#!/usr/bin/env python
# coding: utf-8

# ## Heart Disease Prediction 

# In[2]:


#Importing basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import seaborn as sns # for data visualization
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#Other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[4]:


#Machine Learning Algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[5]:


# Importing the dataset
dataset = pd.read_csv('dataset.csv')


# In[6]:


dataset.info()


# In[ ]:





# In[7]:


dataset.describe()


# ## Feature Selection

# In[8]:


corrmat = dataset.corr()
top_corr_features = corrmat.index
rcParams['figure.figsize'] = 20,20
#plot heat map
g = sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[9]:


#Visualization of Feature Distribution
dataset.hist()


# In[10]:


#Checking the Dataset is Balanced or not 
rcParams['figure.figsize'] = 8,6
plt.bar(dataset['target'].unique(),dataset['target'].value_counts(),color=['red','green'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')


# In[11]:



#VISULAIZATIONS----relationship between attributes
rcParams['figure.figsize'] = 12,6
pd.crosstab(dataset.thal,dataset.target).plot(kind='bar')
plt.title('bar chart for thal vs target')
plt.xlabel('thal')
plt.ylabel('target')


# In[12]:


pd.crosstab(dataset.trestbps,dataset.target).plot(kind='bar')
plt.title('bar chart for trestbps vs target')
plt.xlabel('trestbps')
plt.ylabel('target')


# In[13]:


pd.crosstab(dataset.cp,dataset.target).plot(kind='bar')
plt.title('bar chart for cp vs target')
plt.xlabel('cp')
plt.ylabel('target')


# In[14]:


pd.crosstab(dataset.chol,dataset.target).plot(kind='bar')
plt.title('bar chart for chol vs target')
plt.xlabel('chol')
plt.ylabel('target')


# In[15]:


pd.crosstab(dataset.restecg,dataset.target).plot(kind='bar')
plt.title('bar chart for restecg vs target')
plt.xlabel('restecg')
plt.ylabel('target')


# ## Data Processing

# In[16]:


#Convert categorical variables into dummy variables
dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])


# In[17]:


#Feature Scaling 
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach','oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])


# In[18]:


dataset.head()


# In[19]:


# Splitting the dataset into the Training set and Test set
x = dataset.drop(['target'], axis = 1)
y = dataset['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


# ## KNN

# In[20]:


# Fitting K-Neighbors Classification to the Training set
knn_scores = []
for k in range(1,16):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(x_train, y_train)
    knn_scores.append(knn_classifier.score(x_test, y_test))


# In[21]:


plt.figure(figsize=(20,10))
plt.plot([k for k in range(1, 16)], knn_scores, color = 'red')
for i in range(1,16):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 16)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')


# In[22]:


# Fitting K-Neighbors Classification to the Training set with k=12
knn_classifier = KNeighborsClassifier(n_neighbors = 12)
knn_classifier.fit(x_train, y_train)


# In[24]:


# Predicting the Test set results
y_pred = knn_classifier.predict(x_test)
#print(y_pred)


# In[ ]:


# Accuracy Score
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))


# In[27]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))


# In[28]:


#Interpretation:
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[29]:


print("accuracy:",knn_classifier.score(x_test,y_test))
sns.heatmap(confusion_matrix(y_test,y_pred),annot = True)
plt.xlabel("Y_pred")
plt.ylabel("Y_test")
plt.show()


# In[ ]:


import pickle
with open('knn_model.pkl', 'wb') as file:
    pickle.dump(knn_classifier, file)

