#!/usr/bin/env python
# coding: utf-8

# ## Original Dataset
# ## 35064 rows * 18 columns for 12 different cities
# ## Extracting the first 10 dates of February 2014
# ## New dataset has 2880 rows * 18 columns

# # Importing Libraries

# In[1]:


import numpy as np
import seaborn as sns
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# # Importing Dataset

# In[2]:


df = pd.read_csv('/Users/nidhimahich/Sem2MLcoursework/Short_ChinaData.csv')
df


# # Calculating AQI

# In[3]:


# Create the 'NH3' column with NaN values
df['NH3'] = float('nan')


# In[4]:


df["PM10_24hr_avg"] = df.groupby("station")["PM10"].rolling(window = 24, min_periods = 16).mean().values
df["PM2.5_24hr_avg"] = df.groupby("station")["PM2.5"].rolling(window = 24, min_periods = 16).mean().values
df["SO2_24hr_avg"] = df.groupby("station")["SO2"].rolling(window = 24, min_periods = 16).mean().values
df["NO2_24hr_avg"] = df.groupby("station")["NO2"].rolling(window = 24, min_periods = 16).mean().values
df["NH3_24hr_avg"] = df.groupby("station")["NH3"].rolling(window = 24, min_periods = 16).mean().values
df["CO_8hr_max"] = df.groupby("station")["CO"].rolling(window = 8, min_periods = 1).max().values
df["O3_8hr_max"] = df.groupby("station")["O3"].rolling(window = 8, min_periods = 1).max().values


# In[5]:


def get_PM25_subindex(x):
    if x <= 30:
        return x * 50 / 30
    elif x <= 60:
        return 50 + (x - 30) * 50 / 30
    elif x <= 90:
        return 100 + (x - 60) * 100 / 30
    elif x <= 120:
        return 200 + (x - 90) * 100 / 30
    elif x <= 250:
        return 300 + (x - 120) * 100 / 130
    elif x > 250:
        return 400 + (x - 250) * 100 / 130
    else:
        return 0

df["PM2.5_SubIndex"] = df["PM2.5_24hr_avg"].apply(lambda x: get_PM25_subindex(x))


# In[6]:


## PM10 Sub-Index calculation
def get_PM10_subindex(x):
    if x <= 50:
        return x
    elif x <= 100:
        return x
    elif x <= 250:
        return 100 + (x - 100) * 100 / 150
    elif x <= 350:
        return 200 + (x - 250)
    elif x <= 430:
        return 300 + (x - 350) * 100 / 80
    elif x > 430:
        return 400 + (x - 430) * 100 / 80
    else:
        return 0

df["PM10_SubIndex"] = df["PM10_24hr_avg"].apply(lambda x: get_PM10_subindex(x))


# In[7]:


## SO2 Sub-Index calculation
def get_SO2_subindex(x):
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 380:
        return 100 + (x - 80) * 100 / 300
    elif x <= 800:
        return 200 + (x - 380) * 100 / 420
    elif x <= 1600:
        return 300 + (x - 800) * 100 / 800
    elif x > 1600:
        return 400 + (x - 1600) * 100 / 800
    else:
        return 0

df["SO2_SubIndex"] = df["SO2_24hr_avg"].apply(lambda x: get_SO2_subindex(x))


# In[8]:


## NOx Sub-Index calculation
def get_NOx_subindex(x):
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 180:
        return 100 + (x - 80) * 100 / 100
    elif x <= 280:
        return 200 + (x - 180) * 100 / 100
    elif x <= 400:
        return 300 + (x - 280) * 100 / 120
    elif x > 400:
        return 400 + (x - 400) * 100 / 120
    else:
        return 0

df["NO2_SubIndex"] = df["NO2_24hr_avg"].apply(lambda x: get_NOx_subindex(x))


# In[9]:


## NH3 Sub-Index calculation
def get_NH3_subindex(x):
    if x <= 200:
        return x * 50 / 200
    elif x <= 400:
        return 50 + (x - 200) * 50 / 200
    elif x <= 800:
        return 100 + (x - 400) * 100 / 400
    elif x <= 1200:
        return 200 + (x - 800) * 100 / 400
    elif x <= 1800:
        return 300 + (x - 1200) * 100 / 600
    elif x > 1800:
        return 400 + (x - 1800) * 100 / 600
    else:
        return 0

df["NH3_SubIndex"] = df["NH3_24hr_avg"].apply(lambda x: get_NH3_subindex(x))


# In[10]:


## CO Sub-Index calculation
def get_CO_subindex(x):
    if x <= 1:
        return x * 50 / 1
    elif x <= 2:
        return 50 + (x - 1) * 50 / 1
    elif x <= 10:
        return 100 + (x - 2) * 100 / 8
    elif x <= 17:
        return 200 + (x - 10) * 100 / 7
    elif x <= 34:
        return 300 + (x - 17) * 100 / 17
    elif x > 34:
        return 400 + (x - 34) * 100 / 17
    else:
        return 0

df["CO_SubIndex"] = df["CO_8hr_max"].apply(lambda x: get_CO_subindex(x))


# In[11]:


## O3 Sub-Index calculation
def get_O3_subindex(x):
    if x <= 50:
        return x * 50 / 50
    elif x <= 100:
        return 50 + (x - 50) * 50 / 50
    elif x <= 168:
        return 100 + (x - 100) * 100 / 68
    elif x <= 208:
        return 200 + (x - 168) * 100 / 40
    elif x <= 748:
        return 300 + (x - 208) * 100 / 539
    elif x > 748:
        return 400 + (x - 400) * 100 / 539
    else:
        return 0

df["O3_SubIndex"] = df["O3_8hr_max"].apply(lambda x: get_O3_subindex(x))


# In[12]:


## AQI bucketing
def get_AQI_bucket(x):
    if x <= 50:
        return "Good"
    elif x <= 100:
        return "Satisfactory"
    elif x <= 200:
        return "Moderate"
    elif x <= 300:
        return "Poor"
    elif x <= 400:
        return "Very Poor"
    elif x > 400:
        return "Severe"
    else:
        return np.NaN

df["Checks"] = (df["PM2.5_SubIndex"] > 0).astype(int) +                 (df["PM10_SubIndex"] > 0).astype(int) +                 (df["SO2_SubIndex"] > 0).astype(int) +                 (df["NO2_SubIndex"] > 0).astype(int) +                 (df["NH3_SubIndex"] > 0).astype(int) +                 (df["CO_SubIndex"] > 0).astype(int) +                 (df["O3_SubIndex"] > 0).astype(int)

df["AQI"] = round(df[["PM2.5_SubIndex", "PM10_SubIndex", "SO2_SubIndex", "NO2_SubIndex",
                                 "NH3_SubIndex", "CO_SubIndex", "O3_SubIndex"]].max(axis = 1))
df.loc[df["PM2.5_SubIndex"] + df["PM10_SubIndex"] <= 0, "AQI"] = np.NaN
df.loc[df.Checks < 3, "AQI"] = np.NaN

df["AQI_bucket"] = df["AQI"].apply(lambda x: get_AQI_bucket(x))
df[~df.AQI.isna()].head(13)


# In[13]:


df[~df.AQI.isna()].AQI_bucket.value_counts()


# # Outliers

# In[14]:


import numpy as np

numerical_columns = df.select_dtypes(include=np.number)

Q1 = numerical_columns.quantile(0.25)
Q3 = numerical_columns.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = ((numerical_columns < lower_bound) | (numerical_columns > upper_bound)).any(axis=1)

num_outliers = outliers.sum()
print("Number of outliers before removal:", num_outliers)

# Remove outliers from the dataset
df = df[~outliers]

# Re-check for missing values and outliers in each column
missing_values = df.isnull().sum()
outliers = ((df.select_dtypes(include=np.number) < lower_bound) | (df.select_dtypes(include=np.number) > upper_bound)).any(axis=1)

print("Missing Values:", missing_values.sum())
print("Number of outliers in updated dataset:", outliers.sum(), "\n")
print(df)


# # Treating Missing Values

# In[15]:


data2 = df.copy()


# In[16]:


data2 = data2.fillna(data2.median()) # Replace all null values with median


# # Daily AQI Level trends

# In[17]:


# Group the data by 'station' and 'year' and calculate the mean AQI
grouped_df = data2.groupby(['station', 'day'])['AQI'].mean().reset_index()

# Create a line plot using Plotly
fig = px.line(grouped_df, x='day', y='AQI', color='station',
              title='Daily AQI Levels by City', labels={'year': 'Year', 'AQI': 'AQI'})

# Display the graph
fig.show()


# # Mapping

# In[18]:


dist = (data2['station'])
distset = set(dist)
dd = list(distset)
dictOfWords = {dd[i]: i for i in range(0, len(dd))}
data2['station'] = data2['station'].map(dictOfWords)


# In[19]:


dist = (data2['AQI_bucket'])
distset = set(dist)
dd = list(distset)
dictOfWords = {dd[i]: i for i in range(0, len(dd))}
data2['AQI_bucket'] = data2['AQI_bucket'].map(dictOfWords)


# In[20]:


dist = (data2['wd'])
distset = set(dist)
dd = list(distset)
dictOfWords = {dd[i]: i for i in range(0, len(dd))}
data2['wd'] = data2['wd'].map(dictOfWords)


# In[21]:


data2['AQI_bucket'] = data2['AQI_bucket'].fillna(data2["AQI_bucket"].median())


# In[22]:


data2


# In[23]:


data2.isnull().sum()


# # Dropping redundant columns

# In[24]:


data2 = data2.drop('No', axis = 1)
data2 = data2.drop('PM2.5_24hr_avg', axis = 1)
data2 = data2.drop('PM10_24hr_avg', axis = 1)
data2 = data2.drop('SO2_24hr_avg', axis = 1)
data2 = data2.drop('NO2_24hr_avg', axis = 1)
data2 = data2.drop('NH3_24hr_avg', axis = 1)
data2 = data2.drop('CO_8hr_max', axis = 1)
data2 = data2.drop('O3_8hr_max', axis = 1)
data2 = data2.drop('PM2.5_SubIndex', axis = 1)
data2 = data2.drop('PM10_SubIndex', axis = 1)
data2 = data2.drop('SO2_SubIndex', axis = 1)
data2 = data2.drop('NO2_SubIndex', axis = 1)
data2 = data2.drop('NH3_SubIndex', axis = 1)
data2 = data2.drop('CO_SubIndex', axis = 1)
data2 = data2.drop('O3_SubIndex', axis = 1)
data2 = data2.drop('Checks', axis = 1)
data2 = data2.drop('year', axis =1)
data2 = data2.drop('month', axis =1)
data2 = data2.drop('day', axis =1)
data2 = data2.drop('hour', axis =1)


# In[25]:


data2.columns


# In[26]:


data2 = data2.drop('AQI_bucket', axis =1)


# # EDA (Analyse the data)

# In[27]:


# Plotting the bubble chart
fig = px.scatter(df, x= 'station', y='AQI')

# Showing the plot
fig.show()


# In[28]:


# Plotting the bubble chart
fig2 = px.scatter(df, x= 'PM2.5', y='AQI')

# Showing the plot
fig2.show()


# In[29]:


# Plotting the bubble chart
fig3 = px.scatter(df, x= 'PM10', y='AQI')

# Showing the plot
fig3.show()


# In[30]:


# Plotting the bubble chart
fig4 = px.scatter(df, x= 'SO2', y='AQI')

# Showing the plot
fig4.show()


# In[31]:


# Plotting the bubble chart
fig5 = px.scatter(df, x= 'NO2', y='AQI')

# Showing the plot
fig5.show()


# In[32]:


# Plotting the bubble chart
fig6 = px.scatter(df, x= 'CO', y='AQI')

# Showing the plot
fig6.show()


# In[33]:


# Plotting the bubble chart
fig7 = px.scatter(df, x= 'O3', y='AQI')

# Showing the plot
fig7.show()


# In[34]:


# Plotting the bubble chart
fig8 = px.scatter(df, x= 'TEMP', y='AQI')

# Showing the plot
fig8.show()


# In[35]:


# Plotting the bubble chart
fig9 = px.scatter(df, x= 'PRES', y='AQI')

# Showing the plot
fig9.show()


# In[36]:


# Plotting the bubble chart
fig10 = px.scatter(df, x= 'DEWP', y='AQI')

# Showing the plot
fig10.show()


# In[37]:


# Plotting the bubble chart
fig11 = px.scatter(df, x= 'RAIN', y='AQI')

# Showing the plot
fig11.show()


# In[38]:


# Plotting the bubble chart
fig12 = px.scatter(df, x= 'wd', y='AQI')

# Showing the plot
fig12.show()


# In[39]:


# Plotting the bubble chart
fig13 = px.scatter(df, x= 'WSPM', y='AQI')

# Showing the plot
fig13.show()


# # Feature Selection

# In[40]:


columns_of_interest = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM', 'wd']

# Calculate the correlation matrix
correlation_matrix = data2[columns_of_interest].corr()

# Create a copy of the correlation matrix and set the left half triangle values to NaN
correlation_matrix_tril = correlation_matrix.where(np.tril(np.ones(correlation_matrix.shape), k=-1).astype(bool))

# Plot the correlation matrix with the right diagonal values
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_tril, annot=True, fmt=".2f", square=True)

# Customize the plot
plt.title("Correlation Matrix")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()


# In[41]:


features = data2[['station','PM2.5','PM10', 'SO2', 'NO2', 'CO', 'O3','TEMP','PRES','DEWP','RAIN','wd','WSPM']]
labels = data2['AQI']


# # Training and Testing Data

# In[42]:


# Splitting into train and test data

Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,labels,test_size = 0.2, random_state = 2)


# # Random Forest Regressor

# In[43]:


# Create and train the Random Forest Regressor
regr = RandomForestRegressor(max_depth = 2, random_state = 0)
regr.fit(Xtrain, Ytrain)

y_pred = regr.predict(Xtest)

r2_RF = r2_score(Ytest, y_pred) # model accuracy
# Print the R2 score
print("R2 Score: ", r2_RF)


# In[44]:


# Predict the target variable using the test data
y_pred = regr.predict(Xtest[:100])

# Calculate the R-squared score (model accuracy)
accuracy = r2_score(Ytest[:100], y_pred)
print(f"Model Accuracy (R-squared): {accuracy}")

# Create a DataFrame with predicted and actual values
data = pd.DataFrame({'Predicted': y_pred, 'Actual': Ytest[:100].values})

# Plot the graph
fig = px.line(data, y=['Predicted', 'Actual'], title='Random Forest Regressor: Predicted vs Actual Values (100 Samples)')

# Set the x-axis label
fig.update_xaxes(title_text='Samples')

# Set the y-axis label
fig.update_yaxes(title_text='Values')

# Show the graph
fig.show()


# # Decision Tree

# In[45]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Create a decision tree regressor
model = DecisionTreeRegressor()

# Fit the model on the training data
model.fit(Xtrain, Ytrain)

# Make predictions on the test data
predictions = model.predict(Xtest)

# Calculate the R2 score
r2_DT = r2_score(Ytest, predictions)

# Print the R2 score
print("R2 Score: ", r2_DT)


# In[46]:


# Predict the target variable using the test data
y_pred = model.predict(Xtest[:100])

# Calculate the R-squared score (model accuracy)
accuracy = r2_score(Ytest[:100], y_pred)
print(f"Model Accuracy (R-squared): {accuracy}")

# Create a DataFrame with predicted and actual values
data = pd.DataFrame({'Predicted': y_pred, 'Actual': Ytest[:100].values})

# Plot the graph
fig = px.line(data, y=['Predicted', 'Actual'], title='Decision Tree: Predicted vs Actual Values (100 Samples)')

# Set the x-axis label
fig.update_xaxes(title_text='Samples')

# Set the y-axis label
fig.update_yaxes(title_text='Values')

# Show the graph
fig.show()


# # Logistic Regression

# In[47]:


# Create a logistic regression classifier
lr_model = LogisticRegression(max_iter=1000)

# Fit the model on the training data
lr_model.fit(Xtrain, Ytrain)

# Make predictions on the test data
predictions = lr_model.predict(Xtest)

# Calculate the accuracy score
accuracy_LR = accuracy_score(Ytest, predictions)

# Print the accuracy score
print("Accuracy: ", accuracy_LR)


# In[48]:


# Predict the target variable using the test data
y_pred = lr_model.predict(Xtest[:100])

# Calculate the R-squared score (model accuracy)
accuracy = r2_score(Ytest[:100], y_pred)
print(f"Model Accuracy (R-squared): {accuracy}")

# Create a DataFrame with predicted and actual values
data = pd.DataFrame({'Predicted': y_pred, 'Actual': Ytest[:100].values})

# Plot the graph
fig = px.line(data, y=['Predicted', 'Actual'], title='Predicted vs Actual Values (100 Samples)')

# Set the x-axis label
fig.update_xaxes(title_text='Samples')

# Set the y-axis label
fig.update_yaxes(title_text='Values')

# Show the graph
fig.show()


# # ADA Boost

# In[49]:



# Create an AdaBoost classifier with a decision tree base estimator
ada_model = AdaBoostClassifier()

# Fit the model on the training data
ada_model.fit(Xtrain, Ytrain)

# Make predictions on the test data
predictions = ada_model.predict(Xtest)

# Calculate the accuracy score
accuracy = accuracy_score(Ytest, predictions)

# Print the accuracy score
print("Accuracy: ", accuracy)


# In[50]:


# Predict the target variable using the test data
y_pred = ada_model.predict(Xtest[:100])

# Calculate the R-squared score (model accuracy)
accuracy = r2_score(Ytest[:100], y_pred)
print(f"Model Accuracy (R-squared): {accuracy}")

# Create a DataFrame with predicted and actual values
data = pd.DataFrame({'Predicted': y_pred, 'Actual': Ytest[:100].values})

# Plot the graph
fig = px.line(data, y=['Predicted', 'Actual'], title='Predicted vs Actual Values (100 Samples)')

# Set the x-axis label
fig.update_xaxes(title_text='Samples')

# Set the y-axis label
fig.update_yaxes(title_text='Values')

# Show the graph
fig.show()


# # Support Vector Machine

# In[51]:


# Create an SVM classifier
svc_model = SVC()

# Fit the model on the training data
svc_model.fit(Xtrain, Ytrain)

# Make predictions on the test data
predictions = svc_model.predict(Xtest)

# Calculate the accuracy score
accuracy_SVC = accuracy_score(Ytest, predictions)

# Print the accuracy score
print("Accuracy: ", accuracy_SVC)


# In[52]:


# Predict the target variable using the test data
y_pred = svc_model.predict(Xtest[:100])

# Calculate the R-squared score (model accuracy)
accuracy = r2_score(Ytest[:100], y_pred)
print(f"Model Accuracy (R-squared): {accuracy}")

# Create a DataFrame with predicted and actual values
data = pd.DataFrame({'Predicted': y_pred, 'Actual': Ytest[:100].values})

# Plot the graph
fig = px.line(data, y=['Predicted', 'Actual'], title='Predicted vs Actual Values (100 Samples)')

# Set the x-axis label
fig.update_xaxes(title_text='Samples')

# Set the y-axis label
fig.update_yaxes(title_text='Values')

# Show the graph
fig.show()


# # Accuracy of different models

# In[53]:


import matplotlib.pyplot as plt

# Define the models and their corresponding accuracy scores
models = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'SVM']
accuracy_scores = [r2_RF, r2_DT, accuracy_LR, accuracy_SVC]

# Set up the figure and axes
fig, ax = plt.subplots()
ax.bar(models, accuracy_scores, color='#4287f5', edgecolor='black')

# Set the plot title and axes labels
ax.set_title('Accuracy Comparison of Different Models', fontsize=16)
ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Accuracy (R2 Score)', fontsize=12)

# Set the y-axis limits between 0 and 1
ax.set_ylim([0, 1])

# Add grid lines
ax.grid(True, linestyle='--', alpha=0.5)

# Add data labels to the bars
for i, v in enumerate(accuracy_scores):
    ax.text(i, v + 0.01, str(round(v, 2)), ha='center')

# Rotate x-axis labels if necessary
plt.xticks(rotation=45, ha='right')

# Adjust the plot layout
plt.tight_layout()

# Display the plot
plt.show()

