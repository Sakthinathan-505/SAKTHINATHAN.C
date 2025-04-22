#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()
print(iris)


# In[2]:


print("\ntype:\n",type(iris))


# In[3]:


print("\nkeys:\n",iris.keys())


# In[4]:


print("\ntype of data and target:\n",type(iris.data),type(iris.target))


# In[5]:


print("\ndata shape:\n",iris.data.shape)


# In[6]:


print("\ntarget names:\n",iris.target_names)


# In[13]:


X=iris.data
Y=iris.target
df=pd.DataFrame(X,columns=iris.feature_names)
print("\niris dataframe:\n",df.head())


# In[14]:


diabetes=datasets.load_diabetes()
print("\ndiabetes datasets;\n",diabetes);


# In[15]:


X=diabetes.data
y=diabetes.target
df=pd.DataFrame(X,columns=diabetes.feature_names)
print("\nDiabetes dataframe:\n",df.head())


# In[21]:


data=datasets.load_breast_cancer()
label_names=data['target_names']
labels=data['target']
feature_names=data['feature_names']
features=data['data']
print("Breast Cancer data:\n",data);
print("\nLabel names:\n",label_names)
print("\nLabels:\n",labels)
print("\nFeature names:\n",feature_names)
print("\nFeatures:\n",features)


# In[23]:


import matplotlib.pyplot as plt 
import pandas 
from sklearn import tree 
from sklearn.tree import Decision TreeClassifier 
import matplotlib.pyplot as plt 
df = pandas.read_csv("d:\decision_tree.csv") 
print(df) 


# In[ ]:




