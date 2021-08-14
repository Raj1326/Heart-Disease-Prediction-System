#!/usr/bin/env python
# coding: utf-8

# Importing basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

#Machine Learning Algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing the dataset
df=pd.read_csv('heart.csv')
df.info()
df.describe()

# Feature Selection
import seaborn as sns
corrmatrix=df.corr()
top_corr_features=corrmatrix.index
plt.figure(figsize=(20,20))

#plot heat map
g= sns.heatmap(df[top_corr_features].corr(),annot=True,cmap='RdYlGn')
df.hist()

#converting data in one scale
standardScaler=StandardScaler()
coumn_to_scale=['age','trestbps','chol','thalach','oldpeak']
df[coumn_to_scale]=StandardScaler().fit_transform(df[coumn_to_scale])
df.head()

# splitting data
y=df['target'].values
x=df.drop(['target'],axis=1)

# train test split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)

# Fitting Random Forest Classifier to the Training set
rf = RandomForestClassifier(random_state = 42).fit(x_train,y_train)

# Predicting the Test set results
y_pred=rf.predict(x_test)

# Accuracy Score
print("accuracy:",rf.score(x_test,y_test))

#Confusion Matrix
sns.heatmap(confusion_matrix(y_test,y_pred),annot = True)
plt.xlabel("Y_pred")
plt.ylabel("Y_test")
plt.show()

#Interpretation:
print(classification_report(y_test,y_pred))

#Converting to pickel file
import pickle
with open('randomforest.pkl','wb') as file:
    pickle.dump(rf,file)


# In[ ]:





# In[ ]:




