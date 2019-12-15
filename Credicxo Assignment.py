#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


musk = pd.read_csv('musk.csv')


# In[3]:


musk.head()


# In[4]:


musk['class'].value_counts()


# In[5]:


musk.info()


# In[6]:


musk.describe()


# ## Missing Value Treatment

# In[7]:


(musk.isnull().sum()).sum()


# ### No missing value found

# ## Outier Treatment

# In[8]:


musk.columns


# In[11]:


bp = PdfPages('BoxPlots with Attrition Split.pdf')
for variable in musk.columns.difference(['ID', 'molecule_name', 'conformation_name']):
    fig,axes = plt.subplots(figsize=(10,4))
    sns.boxplot( y=variable, data = musk)
    plt.title(str('Box Plot of ') + str(variable))
    
bp.close()


# ### No outier found using Box Plot

# ### 2.1 Data Exploratory Analysis
#     - Variable reduction using T-test

# In[14]:


tstats_df = pd.DataFrame()
for variable in musk.columns.difference(['ID', 'molecule_name', 'conformation_name', 'class']):
    tstats = stats.ttest_ind(musk[musk['class']==1][variable],musk[musk['class']==0][variable])
    temp = pd.DataFrame([variable, tstats[0], tstats[1]]).T
    temp.columns = ['Variable Name', 'T-Statistic', 'P-Value']
    tstats_df = pd.concat([tstats_df, temp], axis=0, ignore_index=True)


# In[15]:


print(tstats_df)


# In[26]:


var_list=list(tstats_df.sort_values(by='P-Value', ascending=True).head(30)['Variable Name'])


# #### Top 30 variable is selected with minimum P-value from T-test.

# ### Data Exploratory Analysis
#     - Variable Transformation: (i) Bucketing

# In[30]:


bp = PdfPages('Transformation Plots.pdf')

for variable in var_list:
    binned = pd.cut(musk[variable], bins=10, labels=list(range(1,11)))
    binned = binned.dropna()
    ser = musk.groupby(binned)['class'].sum() / (musk.groupby(binned)['class'].count()-musk.groupby(binned)['class'].sum())
    ser = np.log(ser)
    fig,axes = plt.subplots(figsize=(10,4))
    sns.barplot(x=ser.index,y=ser)
    plt.ylabel('Log Odds Ratio')
    plt.title(str('Logit Plot for identifying if the bucketing is required or not for variable ') + str(variable))
    bp.savefig(fig)

bp.close()


# #### No Variable needs Bucketing

# ###  Data Exploratory Analysis
#     - Multicollinearity

# In[34]:


b = musk[var_list]
b


# In[41]:


b = musk[var_list]

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(b.values, i) for i in range(b.shape[1])]
vif["features"] = b.columns

print(vif)


# In[43]:


var_list2=list(vif[vif['VIF Factor']<=20]['features'])


# In[44]:


var_list2


# #### Selected the variables having VIF<= 20

# ###  Model Build and Diagnostics
#  - Train and Test split

# In[49]:


X = musk[var_list2]
y= musk['class']
train_features = X.columns
train_X, test_X,train_y,test_y = train_test_split(X,y, test_size=0.2, random_state=42)


# ### 3.2 Model Build and Diagnostics
#     - Model build on the train_X sample using logistic regression model

# In[52]:


logreg = LogisticRegression()
logreg.fit( train_X, train_y )


# In[53]:


logreg.coef_[0]


# In[55]:


list( zip( train_features, logreg.coef_[0] ) )


# In[60]:


#Predicting the test cases
musk_test_pred = pd.DataFrame( { 'actual':  test_y,
                            'predicted': logreg.predict( test_X ) } )


# In[65]:


musk_test_pred['predicted'].value_counts()


# In[62]:


# Creating a confusion matrix

from sklearn import metrics

cm = metrics.confusion_matrix( musk_test_pred.actual,
                            musk_test_pred.predicted, [1,0] )
cm


# In[64]:


sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = [1, 0] , yticklabels = [1,0] )
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[66]:


#Accuracy
score = metrics.accuracy_score( hr_test_pred.actual, hr_test_pred.predicted )
round( float(score), 2 )


# In[69]:


#Precision
68/(68+44)


# In[70]:


#Recall
1067/(1067+141)


# In[72]:


#F1 Score
2*(0.61*0.88)/(0.61+0.88)


# In[ ]:




