#!/usr/bin/env python
# coding: utf-8

# ## Fetal Health - Exploratory Data Analysis
# 
# This project's data comes from kaggle. I will leave a link here to the source page: https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification?datasetId=916586&sortBy=voteCount&sort=votes
# 
# With this project, I will attempt to create a model that classifies records/information from cardiotocogram exams into three different statuses, originally labelled by obstetritians. 
# 
# This notebook will focus on EDA and any transformations to use towards model building.

# In[2]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv("Documents/FetalHealthClassification/fetal_health.csv")


# In[44]:


#Target -> fetal_health
#1 -> Normal
#2 -> Suspect
#3 -> Pathological (pathological meaning potential involvement of disease)

df.head()


# In[192]:


df.tail()


# In[45]:


df.shape


# In[46]:


#Checking for any null values, for which we don't have any with this data.

df.isnull().sum()


# In[47]:


#There doesn't seem to be any abnormalities from glancing the common statistics here. 
#The min and max baseline heart rate is 106 and 160 respectively.

df.describe()


# We can see we don't have any object or string datatype columns, and our target is already set to float. 

# In[48]:


df.info()


# We'll also have to keep in mind there are many more normal records than there are suspect or pathological.

# In[54]:


df['fetal_health'].value_counts()


# To start our EDA, we can make some boxplots of the different attributes we have split by the fetal health classification target.

# In[49]:


sns.set(font_scale=1.5)
plt.figure(figsize=(20,12))
g = sns.boxenplot(x='fetal_health', y='baseline value', data=df,
             saturation=1.5)
g.set_xlabel('Fetal Health')
g.set_ylabel('Baseline Fetal Heartrate Value')
g.set_title('Baseline Heartrate per Fetal Health Classification', fontsize=20)
plt.show()


# 2.0 marks records that are suspect for poor condition. Those that are normal or pathological seem to be somewhat similar.
# 
# Next we'll try fetal movements and see if there are any further trends there.

# In[50]:


sns.set(font_scale=1.5)
plt.figure(figsize=(20,12))
g = sns.boxenplot(x='fetal_health', y='fetal_movement', data=df,
             saturation=1.5)
g.set_xlabel('Fetal Health')
g.set_ylabel('Fetal Movement')
g.set_title('Fetal Movements/second per Health Classification', fontsize=20)
plt.show()


# There seems to more commonly be movements among reocrds classified as pathological.
# 
# Now we'll look at uterine contractions

# In[4]:


sns.set(font_scale=1.5)
plt.figure(figsize=(20,12))
g = sns.boxenplot(x='fetal_health', y='uterine_contractions', data=df,
             saturation=1.5)
g.set_xlabel('Fetal Health')
g.set_ylabel('Uterine Contractions')
g.set_title('Uterine Contractions per Health Classification', fontsize=20)
plt.show()


# There doesn't seem to be a lot of variability among uterine contractions. There are more higher values for normal records.
# 
# Now, we'll look at each of the deceleration attributes.

# In[52]:


dec_cols = ['light_decelerations', 'severe_decelerations', 'prolongued_decelerations'] #there may be a typo in the last column here

for col in dec_cols:
    sns.set(font_scale=1.5)
    plt.figure(figsize=(16,8))
    g = sns.boxenplot(x='fetal_health', y=col, data=df,
             saturation=1.5)
    g.set_xlabel('Fetal Health')
    g.set_ylabel(col)
    g.set_title('{} & Health Classification'.format(col), fontsize=20)
    plt.show()


# We can see there is much more deceleration among pathological records than normal or suspect records. I've also noticed some traits among normal and pathological records seem to be similar, such as light decelerations and baseline heart rate.
# 
# Now I'll create a pair plot to look at distributions across these different attributes and among the 3 different classes provided.

# In[179]:


pair_cols = ['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 'light_decelerations', 'severe_decelerations', 'prolongued_decelerations', 'fetal_health']

sns.pairplot(df[pair_cols], hue="fetal_health", height=3, aspect=1, palette='magma')
g.fig.subplots_adjust(top=0.95)
g.fig.suptitle('Pairplot for Various Attributes', fontsize=26)
plt.show()


# We can see there are primarily pathological records recording higher values within severe decelerations as well as prolonged decelerations in relation to baseline heartrate and a few other attributes (at the bottom). So we can note that these pathological records have a trace for higher values in these relations. This is keeping in mind the lower amount of pathological records.

# In[55]:


var_cols = ['abnormal_short_term_variability', 'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability']

for col in var_cols:
    sns.set(font_scale=1.5)
    plt.figure(figsize=(16,8))
    g = sns.boxenplot(x='fetal_health', y=col, data=df,
             saturation=1.5)
    g.set_xlabel('Fetal Health')
    g.set_ylabel(col)
    g.set_title('{} & Health Classification'.format(col), fontsize=20)
    plt.show()


# Suspect and pathological records seem to have higher abonormal short term variability as well as generally higher percentage of time with abnormal long term variability.

# In[157]:


var_cols.append('fetal_health')


# In[178]:


sns.set(font_scale=1.5)
g = sns.pairplot(df[var_cols], hue="fetal_health", height=5, aspect=1.5, palette='magma')
g.fig.subplots_adjust(top=0.95)
g.fig.suptitle('Pairplot for Abnormal Variability', fontsize=26)
plt.show()


# We can see that pathological records have higher values for these attributes, namely percentage of time with long term variability and abnormal short term variability.

# In[72]:


g = sns.relplot(
    data=df,
    x="prolongued_decelerations", y="uterine_contractions",
    hue="fetal_health", height=8, aspect=1.5, palette='plasma')
g.set(title='Uterine Contractions vs. Prolonged Decelerations')
plt.show()


# Here is a re-created plot from the pairplot above with most possible plots we can make with the data, where with lower uterine contractions, there are higher prolonged decelerations among pathological records. This is an example within the data of how the classes are separated by the data. I believe using LDA would greatly aid this analysis. 

# After reading a little on Cardiotocography, I learned a little more on what is illustrated on the graphs produced. Here is a link to the information: https://en.wikipedia.org/wiki/Cardiotocography
# 
# This link also provides some insight into which pregnancies fall under the 3 classes we want to distinguish here. We can use that information to help us.
# 
# I then could discern that the histogram prefix of these features refer to graphs generated (fetal heartrate) from the pregnant women.
# 
# Below, I'll attempt to get a sense of how many records of various attributes fall under certain ranges of graph features. 

# In[94]:


hist_cols = df.columns.tolist()[11:21]

df[hist_cols].hist(bins=15, figsize=(18, 30), layout=(5, 2));
plt.suptitle('Distributions of Cardiotocography Values')


# In[181]:


hist_cols.append('fetal_health')


# In[186]:


new_hist_cols = hist_cols[-6:]


# In[190]:


sns.set(font_scale=1.5)
g = sns.pairplot(df[new_hist_cols], hue="fetal_health", height=3, aspect=1.0, palette='magma')
g.fig.subplots_adjust(top=0.95)
g.fig.suptitle('Pairplot for Cardiotocography Graphs', fontsize=26)
plt.show()


# We can see that pathological records have lower metrics for mode, median, and mean for cardiotocography exams among their graph tendencies. This also comes with higher variance.
# 
# The normal and suspect columns have higher values for the same metrics.

# In[105]:


g = sns.catplot(x="baseline value", y="prolongued_decelerations", hue="fetal_health",
                palette="plasma", height=6, aspect=2.0,
                data=df)
g.set_xticklabels(rotation=90)
g.set(title='Prolonged Decelerations and Heartrate Baseline')
g.despine(left=True)


# We can see that within a range of heartrate values, pathlogical records have higher prolonged decelerations.

# In[111]:


base_comp_cols = pair_cols[1:-1]
base_comp_cols


# Below, I will compare two of the deceleration attributes again to decipher any commonalities. Namely, <b>is there a correlation between light and prolonged decelerations?</b>

# In[154]:


f, ax = plt.subplots(figsize=(16,10))
sns.despine(bottom=True, left=True)

# Show each observation with a scatterplot
sns.stripplot(x="prolongued_decelerations", y="light_decelerations", hue="fetal_health",
              data=df, dodge=True)
ax.set_title('Light vs Prolonged Decelerations')
plt.show()


# We can see that generally the pathological records have more prolonged decelerations than light decelerations.

# Lastly, let's plot a heatmap to pick up any correlations.

# In[195]:


sns.set(font_scale=1.0)
plt.figure(figsize=(20, 12)) 
plt.title('Fetal Health Classification Heatmap', fontsize=16)
heatmap = sns.heatmap(df.corr(), annot=True)


# We can notice the cardiotocography exam graph statistic metrics are correlated with each other. We'll keep note of this as we move into model building and track other aspects during the process.

# ## Closing Remarks:
# 
# * Pathological records have more severe and prolonged decelerations. We could see this from the box plots created in the beginning.
# 
# 
# * Among pathological reocrds, there are decreases in uterine contractions, fetal movements, and range of baseline heartrate values that correlate with higher pprolonged deceleration rates.
# 
# 
# * Among normal and some suspect records, we can see that there are increased accelerations with not as much deceleration. It seems that <b>the suspect and pathological records have more decelerations in general.</b>
# 
# 
# * Pathological records have higher values of percentage of time with long term variability and abnormal short term variability.
# 
# 
# * Pathological records have lower metrics for mode, median, and mean for cardiotocography exams among their graph tendencies, while normal records have higher values for these metrics. This correlates with their cardiotocography graphs.
# 
# Within model building, I believe LDA will be a great step in determining what best separates these classes.

# In[ ]:




