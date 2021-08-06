#!/usr/bin/env python
# coding: utf-8

# ## Share_market(Linear_Regression)
# #### Ramanathan N

# In[1]:


import numpy as np
import pandas as pd
import datetime
import quandl

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import seaborn as sns

plt.style.use('seaborn-darkgrid')
plt.rc('figure', figsize=(16,10))
plt.rc('lines', markersize=4)


# In[2]:


#start_date = datetime.date(2017,1,1)
#end_date = datetime.date(2018,12,28)
data = pd.read_csv("D:\google chrome\BAJAJ_AUTO.csv", low_memory = False, skiprows = 1, encoding = "ISO-8859-1")
#data = quandl.get("BSE/BOM532814", authtoken="43vNrp7GWQtzPqyguXPB")
print("The GTD dataset has {} samples with {} features.".format(*data.shape))


# In[3]:


data=data.iloc[0:2300]
data=data.iloc[::-1]


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.columns


# In[8]:


# Create a new DataFrame with only closing price and date
df = pd.DataFrame(data,columns=['Date','Close'])


# In[9]:


df.head()


# In[10]:


df.info()


# In[11]:


import matplotlib.dates as mdates


years = mdates.YearLocator()
months = mdates.MonthLocator()
yearsFmt = mdates.DateFormatter('%Y')  # add some space for the year label

fig,ax = plt.subplots()
ax.plot(df['Date'],df['Close'])

ax.xaxis.set_minor_locator(months)
plt.setp(ax.xaxis.get_minorticklabels(), rotation=90)
ax.xaxis.set_major_locator(years)

# Set figure title
plt.title('Close Stock Price History  [2011 - 2018]', fontsize=16)
# Set x label
plt.xlabel('Date', fontsize=14)
# Set y label
plt.ylabel('Closing Stock Price in Rs', fontsize=14)

# Rotate and align the x labels
fig.autofmt_xdate()

# Show plot
plt.show()


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


train, test = train_test_split(df,test_size=0.20)


# In[14]:


from sklearn.linear_model import LinearRegression


# In[15]:


X_train = np.array(train.index).reshape(-1, 1)
y_train = train['Close']


# In[16]:


model = LinearRegression()
# Fit linear model using the train data set
model.fit(X_train, y_train)


# In[17]:


print('Slope: ', np.asscalar(np.squeeze(model.coef_)))
# The Intercept
print('Intercept: ', model.intercept_)


# In[18]:


plt.figure(1, figsize=(16,10))
plt.title('Linear Regression | Price vs Time')
plt.scatter(X_train, y_train, edgecolor='w', label='Actual Price')
plt.plot(X_train, model.predict(X_train), color='r', label='Predicted Price')
plt.xlabel('Integer Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[19]:


X_test = np.array(test.index).reshape(-1, 1)
y_test = test['Close']


# In[20]:


y_pred = model.predict(X_test)
print(y_pred[0:25])
#df['Prediction'] = y_pred[:24]


# In[21]:


df.shape


# In[22]:


print(type(y_pred))
# Generate 25 random numbers
randints = np.random.randint(1500, size=20)

# Select row numbers == random numbers
df_sample = df[df.index.isin(randints)]
df_sample['Prediction']=y_pred[0:20]


# In[23]:


df_sample.head()


# In[24]:


# Create subplots to plot graph and control axes
fig, ax = plt.subplots()
df_sample.plot(x='Date', y=['Close', 'Prediction'], kind='bar', ax=ax)

# Set figure title
plt.title('Comparison Predicted vs Actual Price in Sample data selection', fontsize=16)

# 

# Set x label
plt.xlabel('Date', fontsize=14)

# Set y label
plt.ylabel('Stock Price in Rs', fontsize=14)

# Show plot
plt.show()


# In[25]:


from scipy.stats import norm

# Fit a normal distribution to the data:
mu, std = norm.fit(y_test - y_pred)

ax = sns.distplot((y_test - y_pred), label='Residual Histogram & Distribution')

# Calculate the pdf over a range of values         
x = np.linspace(min(y_test - y_pred), max(y_test - y_pred), 100)
p = norm.pdf(x, mu, std)

# And plot on the same axes that seaborn put the histogram
ax.plot(x, p, 'r', lw=2, label='Normal Distribution') 

plt.legend()
plt.show()


# In[26]:


df['Prediction'] = model.predict(np.array(df.index).reshape(-1, 1))


# In[27]:


df.head()


# In[28]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[29]:


print('R2: ', metrics.r2_score(y_test, y_pred))


# In[30]:


from sklearn.metrics import explained_variance_score
explained_variance_score(y_test, y_pred)

