#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression

# <ul>
#   <li>Type of Regression ALGORITHM</li>
# Regression models (both linear and non-linear) are used for predicting a real value
#   <li>It models the relationship between -> Dependent Variable & Single independent Variable.</li>
#   <li>Relationship is shown by straight Line</li>
# </ul>
# 
# 
# <img src = "https://github.com/Bhevendra/justrashdata/blob/main/1_GSAcN9G7stUJQbuOhu0HEg.png?raw=true"  alt="Simple Linear Regression" width="400" height="500" style="vertical-align:middle;margin:0px 50px" >

# <dl>
#   <dt>Dependent Variable <strong> (Y) </strong> </dt>
#   <dd>-> Continuos or real Value</dd>
#   <dt>Independent Variable <strong> (X) </strong> </dt>
#   <dd>-> Continuous or Categorical Values</dd>
# </dl>

# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("https://raw.githubusercontent.com/Bhevendra/Machine-Learning/main/Salary_Data.csv")


# In[3]:


df.head()


# In[4]:


plt.scatter(df["YearsExperience"], df["Salary"])
plt.xlabel("Years of Experience (X)")
plt.ylabel("Salary (Y)")
plt.title("Dataset for Linear Regression")


# In[5]:


df.iloc[:,:-1]


# In[6]:


df.iloc[:,-1]


# In[7]:


x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


# In[8]:


#Spliting the data into the Training and Test Set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=0)


# In[10]:


#Training the Simple Linear Regression Model on the Training set

from sklearn.linear_model import LinearRegression
lrs = LinearRegression()
lrs.fit(x_train,y_train)


# In[11]:


#Predicting the Test Set results

y_pred = lrs.predict(x_test)


# In[12]:


#Visualize the Training Set

plt.scatter(x_train,y_train,color='blue')
plt.plot(x_train,lrs.predict(x_train), color = 'pink')
plt.title('Salary Acc. to Experiance')
plt.xlabel('Experiance - years')
plt.ylabel('Salary')
plt.show()


# In[13]:


#Visualize the test set

plt.scatter(x_test, y_test, color='blue')
plt.plot(x_train,lrs.predict(x_train), color = 'pink')
plt.title('Salary Acc. to Experiance Test')
plt.xlabel('Experiance - years')
plt.ylabel('Salary')
plt.show()


# In[14]:


lrs.predict([[12]])


# In[15]:


lrs.coef_


# In[16]:


lrs.intercept_


# 
# Y = Salary
# 
# X = Experiance
# 
# B0 = Intercept
# 
# B1 = Coefficient
# 
# ### Final Equation
# 
# # Y = 9312.57 + 26780.09 * X + ε

#  <img src = "https://github.com/Bhevendra/justrashdata/blob/main/OLS-LR.png?raw=true"  alt="OLS" width="800" height="500" style="vertical-align:middle;margin:0px 50px"  >

# In[ ]:




