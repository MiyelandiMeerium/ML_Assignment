#!/usr/bin/env python
# coding: utf-8

# In[1]:


#predict the age of a passenger based on their gender, class of ticket, whether they survived or not, etc.
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns


# In[2]:


# Load the Titanic dataset
data = pd.read_csv('Titanic_Train_Dataset.csv')

# Drop irrelevant features
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)


# In[3]:


# Fill missing values in 'Age' column with the median
data['Age'].fillna(data['Age'].median(), inplace=True)

# Convert 'Sex' feature to numerical using label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])


# In[4]:


# Create feature and target arrays
X = data.drop('Age', axis=1)
y = data['Age']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


# Train the model using Linear Regression
reg = LinearRegression()
reg.fit(X_train, y_train)


# In[6]:


# Make predictions on the testing set
y_pred = reg.predict(X_test)


# In[7]:


# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)


# In[8]:


print("Mean Squared Error:", mse)


# In[15]:


# Create a boxplot of the 'Age' feature
sns.boxplot(x=data['Age'])


# In[11]:


# Visualize the correlation between features
dataplot = sns.heatmap(data.corr(), cmap="PiYG", annot=True)


# In[ ]:




