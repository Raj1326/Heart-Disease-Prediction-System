#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[3]:


df=pd.read_csv('heart.csv')


# In[4]:


df.info()


# In[5]:


df.describe()


# <b>Feature Selection</b> <br>
# To select the import feature we need to find relation <br>
# and select the feature

# In[6]:


import seaborn as sns
corrmatrix=df.corr()
top_corr_features=corrmatrix.index
plt.figure(figsize=(20,20))

#plot heat map
g= sns.heatmap(df[top_corr_features].corr(),annot=True,cmap='RdYlGn')


# In[7]:


df.hist()


# <b>one hot encodding</b><br>
# get_dummies are used for data manipulation of categorical data<br>
# It converts categorical data into dummy <br>
# It produces new column for each unique category <br>
# one hot encodding

# In[ ]:





# In[8]:


#converting data in one scale
standardScaler=StandardScaler()
coumn_to_scale=['age','trestbps','chol','thalach','oldpeak']
df[coumn_to_scale]=StandardScaler().fit_transform(df[coumn_to_scale])


# In[9]:


df.head()


# In[10]:


# splitting data
y=df['target'].values
x=df.drop(['target'],axis=1)


# In[11]:


# train test split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)


# <b>Logisitc Regression </b>

# In[12]:



lr = LogisticRegression(random_state = 42).fit(x_train,y_train)


# In[13]:


y_pred=lr.predict(x_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("accuracy:",lr.score(x_test,y_test))
sns.heatmap(confusion_matrix(y_test,y_pred),annot = True)
plt.xlabel("Y_pred")
plt.ylabel("Y_test")
plt.show()


# In[15]:


import pickle
filename='logist.pkl'
clf=pickle.dump(lr,open(filename,'wb'))


# In[ ]:




