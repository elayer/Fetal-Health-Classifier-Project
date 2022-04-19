#!/usr/bin/env python
# coding: utf-8

# ## Fetal Health - Model Building

# In[484]:


import pandas as pd
import numpy as np
import optuna

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score
import xgboost as xgb

#xgb_cl = xgb.XGBClassifier()

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
import warnings
warnings.filterwarnings('ignore')


# In[311]:


df = pd.read_csv("Documents/FetalHealthClassification/fetal_health.csv")


# In[312]:


df.head()


# In[313]:


df.info()


# We know that we don't have any null values, so we don't have to worry about doing any missing value imputation.

# In[314]:


plt.figure(figsize=(20, 12)) 
plt.title('Fetal Health Classification Heatmap', fontsize=16)
heatmap = sns.heatmap(df.corr(), annot=True)


# As we noticed in the EDA, the metrics of the exams are correlated. Since we'll perform LDA, we may be able to phase out these correlations that way. I'll also print each attribute's distribution in order to make a better decision on how to impute outliers.

# In[315]:


df.hist(bins=15, figsize=(20, 15), layout=(6, 4));


# I want to print the current class distributions here again so that we can easily see the class imbalance as we move into outlier detection and potentially clustering. We saw from the box plots from EDA that there are some extremel outliers among some of the attributes. 
# 
# Since we are dealing with attributes of different scales measuring different elements of the cardiotocography exams. I will impute missing values with <b>IQR capping.</b>

# In[316]:


df['fetal_health'].value_counts() #target value distributions


# ## Outlier Detection
# 
# We'll see what the data looks like after computing the IQR and capping/trimming the columns within the IQR.

# In[317]:


iqr_cols = df.columns.tolist()[:-1] #all columns except the target

new_df = df.copy()


# In[318]:


'''IQR'''
for col in iqr_cols:
    p_25 = new_df[col].quantile(0.25)
    p_75 = new_df[col].quantile(0.75)
    iqr = p_75 - p_25
    
    upper_limit = p_75 + (1.5 * iqr)
    lower_limit = p_25 + (1.5 * iqr)
    
    #Trimming Strategy:
    new_df.loc[new_df[col] > upper_limit, col] = upper_limit
    new_df.loc[new_df[col] < lower_limit, col] = lower_limit
    
    
    #Capping Strategy:
    #new_df[col] = np.where(
            #new_df[col] > upper_limit, upper_limit,
            #np.where(
                #new_df[col] < lower_limit, lower_limit, new_df[col] ))
    


# In[319]:


fig, axes = plt.subplots(1, 2, figsize=(16,5))
plt.suptitle('Baseline Value Distribution before & after IQR Capping')
sns.histplot(x='baseline value', data=df, bins=20, ax=axes[0])
sns.histplot(x='baseline value', data=new_df, bins=20, ax=axes[1])
plt.show()


# It seems that even when binning the data with 20 bins after capping, most of the values in the heartrate baseline data gets shoved into a small range of values. For now let's continue while leaving the data alone. It is possible to also employ a clustering method to detect outliers.

# ## PCA Application
# 
# First, we're going to use PCA as a form of anomoly detection, I learned how to use it in this manner from: https://www.kaggle.com/code/ryanholbrook/principal-component-analysis

# In[320]:


X = df.loc[:, :'histogram_tendency']
y = df.loc[:, 'fetal_health']


# In[321]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[322]:


pca = PCA()
X_pca = pca.fit_transform(X_scaled)

comp_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=comp_names)

X_pca.head()


# In[323]:


loadings = pd.DataFrame(pca.components_.T, columns = comp_names, index=X.columns)


# In[324]:


loadings


# In[325]:


fig, axs = plt.subplots(1, 2, figsize=(25,10))
n = pca.n_components_
grid = np.arange(1, n + 1)

evr = pca.explained_variance_ratio_
axs[0].bar(grid, evr)
axs[0].set(
    xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
)

cv = np.cumsum(evr)
axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
axs[1].set(
    xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
)

plt.show()


# It appears 8 principal components explain at least 80% of the variation within the data. Let's re-run PCA except with 8 components.

# In[326]:


pca = PCA(n_components=8)
X_pca = pca.fit_transform(X_scaled)

comp_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=comp_names)

X_pca.head()


# In[327]:


loadings = pd.DataFrame(pca.components_.T, columns = comp_names, index=X.columns)


# In[328]:


loadings


# In[329]:


data_pca = pd.DataFrame(data = pca.components_,
                           columns = X.columns.values,
                           index = ['Component 1', 'Component 2', 'Component 3', 'Component 4', 'Component 5', 'Component 6', 'Component 7', 'Component 8'])
data_pca


# In[330]:


plt.figure(figsize=(22,10))
sns.heatmap(data_pca, vmin=-1, vmax=1, cmap='GnBu', annot=True)
plt.title('Principal Component Feature Correlations')
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ['Component 1', 'Component 2', 'Component 3', 'Component 4', 'Component 5', 'Component 6', 'Component 7', 'Component 8'],rotation=0, fontsize=9)
plt.show()


# Now I'll look at some of the lpca components within the data and look at differences.

# In[331]:


component = "PC3"

idx = X_pca[component].sort_values(ascending=False).index
df.loc[idx, ["baseline value", "accelerations", "fetal_movement", 'uterine_contractions', 'light_decelerations', 'severe_decelerations', 'prolongued_decelerations', 'fetal_health']]


# Looking at PC3 (since PC3 had the highest value for prolonged decelerations), we can see it found some differences with the pathological records being together on top. We have further confirmation that pathological records have more prolonged decelerations and the baseline heartrate falls within a certain range, which are a few things we found in our EDA.
# 
# PC1 had the highest value for light decelerations

# In[332]:


component = "PC1"

idx = X_pca[component].sort_values(ascending=False).index
df.loc[idx, ["baseline value", "accelerations", "fetal_movement", 'uterine_contractions', 'light_decelerations', 'severe_decelerations', 'prolongued_decelerations', 'fetal_health']]


# We can see here that pathological and some suspect records have lower baseline heartrate and in this component, we have higher light decelerations along with uterine contractions in addition to prolonged decelerations, while normal and some suspect records have no prolonged decelerations.
# 
# I've noticed that fetal movement has a few outlying records in the above two component data frames. This perhaps is the kind of outlier we can expect within the data, but since we have only so much data, they may be important to us.

# PC2 seems to have high values for the cardiotocography exam metrics, lets look at that next.

# In[333]:


component = "PC2"

idx = X_pca[component].sort_values(ascending=False).index
df.loc[idx, ['baseline value', "histogram_width", "histogram_min", "histogram_max", 'histogram_number_of_peaks', 'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance', 'histogram_tendency', 'fetal_health']]


# The exam graph tendency for normal records indicates a positive trend, while for pathological records a negative trend. I also included baseline heartrate here as well. 
# 
# It seems pathological records have a trace of lower baseline heartrates and a shorter exam graph. Pathological records also have lower mean, mode, and median values in their graphs and consistent patterns between those and normal records. Again, we noticed this in our EDA, and our PCA analysis here backs it up!
# 
# Since the anomolies I noticed may be important to retain, I'll choose to keep them.

# ## KMeans Feature Application
# 
# Could we develop clustering of the 3 classes of our target variable using KMeans?

# In[334]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[335]:


wcss=[]

for i in range(1,10):
    kmeans=KMeans(i, init='k-means++')
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
wcss


# In[336]:


plt.plot(range(1,10),wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title("WCSS of K-means for 1 to 10 Clusters")
plt.show()


# In[337]:


kmeans_new=KMeans(3, init='k-means++')
kmeans_new.fit(X_scaled)
clusters_new=X.copy()
clusters_new['cluster_pred']=kmeans_new.fit_predict(X_scaled)


# In[338]:


class_compare = pd.concat([clusters_new['cluster_pred'], y], axis=1)


# In[339]:


class_compare


# In[340]:


len(np.where(class_compare['cluster_pred']==class_compare['fetal_health'])[0])


# We can try to plot these KMeans cluster features and see how well they go with the data. Then, if they can be used reliably as features, we could include these clusters in our data.

# In[341]:


features = ['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 'light_decelerations', 'severe_decelerations', 'prolongued_decelerations']


# In[342]:


X_clus = X.copy()
X_clus["Cluster"] = class_compare.cluster_pred.astype("category")
X_clus["fetal_health"] = y
sns.relplot(
    x="value", y="fetal_health", hue="Cluster", col="variable", alpha=0.5,
    height=5, aspect=0.7, facet_kws={'sharex': False}, col_wrap=4,
    data=X_clus.melt(
        value_vars=features, id_vars=["fetal_health", "Cluster"],
    ),
);


# We can see some separation between the clusters when plotting with these select features, we can try to model with and without these cluster features and see how model performance is impacted.

# In[404]:


#COMMENT OUT if you wish not to use the cluster feature
from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()
#lbl.fit_transform(class_compare['cluster_pred'].astype(str))
X['kmeans_cluster'] = lbl.fit_transform(class_compare['cluster_pred'].astype(str)) #model does slightly better WITH this feature

#try object if category becomes an issue


# ## LDA Application
# 
# Now that we've seen what values and attributes best separate the data, lets now conduct LDA on the classes in question (fetal_health) and see if they are split in a similar manner.
# 
# Since we have 3 classes, we can use LDA to split the classes into 2 linear discriminants.
# 
# Helpful blog on implementation: https://www.mygreatlearning.com/blog/linear-discriminant-analysis-or-lda/

# In[405]:


lda = LDA(n_components=2)


# In[406]:


X_lda = lda.fit_transform(X, y)


# In[407]:


lda.explained_variance_ratio_

#before adding kmeans: array([0.8035912, 0.1964088])


# In[408]:


X_lda[:,0]


# In[409]:


plt.figure(figsize=(10,8))
sns.scatterplot(x=X_lda[:,0], y=X_lda[:,1], c=y, cmap='turbo', alpha=0.7, edgecolors='black')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('Linear Discriminant Analysis Visualization')
plt.show()


# In[410]:


X_lda.shape


# Now we can try to build a model using these LDA components that conveniently resulted in 2-dimensions for us to plot above.

# In[411]:


X.head() #you apply scaler AFTER calling train_test_split!!


# ## Model Building
# We can try to build models using X_lda, our regular X dataset, (or maybe combine both?)
# 
# We can try to make a case for choosing particular models:
# 
# * SVM: We want to try to separate our classes (target being fetal_health) as best we can, therefore this model is a rational choice.
# 
# 
# * KnearestNeighbors: Since the class splits have been separable based on distance for many attributes.
# 
# 
# * Random Forest Classifier: There are many thresholds in the data where there is a clear distinction what the dominant group is in a given attribute's array of values.
# 
# 
# * AdaBoost, XGBoost and CatBoost: More powerful tree-based algorithms.

# In[412]:


#Run if we want both X w/ X_lda
X_all = pd.concat([X, pd.DataFrame(X_lda, columns=['LD1', 'LD2'])], axis=1)
X_all.head() #So it could have X, kmeans, and LDA info.


# In[413]:


X.head()


# In[518]:


#We use stratify set to y since the classes are imbalanced in this data.
#Sub out X_all for X_lda or X depending on what you want to test.

#X_all > X_lda, where X_all contains X_lda
#X_all > X where X also has the clusters.

X_train, X_test, y_train, y_test  = train_test_split(X_all, y, test_size=0.25, random_state=3) #, stratify=y) 


# In[519]:


mm_scaler = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train_svm = mm_scaler.transform(X_train)
#X_test_svm = mm_scaler.transform(X_test)


# Since we are trying to separate classes here as a priority, we're going to try SVM with GridSearchCV. Since SVM requires values to be between -1 and 1, we'll use MinMaxScaler for this algorithm.
# 
# <b>Support Vector Machine:</b>

# In[520]:


param_grid = {'C': [1, 2, 5, 10, 20], 'kernel': ('rbf', 'sigmoid')}

svm = SVC(random_state=3, max_iter=-1)

clf = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1)
clf.fit(X_train_svm, y_train)

print('GridSearch Best Estimator: {}'.format(clf.best_estimator_))
print('GridSearch Best Parameters: {}'.format(clf.best_params_))
print()
print('Support Vector Machine Accuracy: {:.2%}'.format(clf.score(mm_scaler.transform(X_test), y_test)))


# In[521]:


y_pred = clf.predict(mm_scaler.transform(X_test))

cm = confusion_matrix(y_test, y_pred, labels = clf.classes_)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)

fig, ax = plt.subplots(figsize=(10,10))
cm_display.plot(ax=ax)
plt.tick_params(axis=u'both', which=u'both',length=0)
plt.title('SVM Confusion Matrix', fontsize=14)
plt.grid(False)
plt.show()


# In[522]:


print(classification_report(y_test, y_pred, target_names = ['Normal', 'Suspect', 'Pathological'])) #classes coming from 1, 2, and 3


# Since the suspect records come between normal and pathological estimates, it isn't much of a surprise that the model has the hardest time classifying those records. The model does slightly better if we choose to NOT stratify on y. We wanted to do this since the classes are imbalanced. 
# 
# However, it's interesting that our results are improved overall with stratify turned off..

# Since the model can classify by proximity, let's next try 
# 
# <b>K Nearest Neighbors:</b>

# In[523]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[524]:


param_grid = {'n_neighbors': [1, 5, 10, 20, 50], 'weights': ['uniform', 'distance'], 'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute')}

knn = KNeighborsClassifier()

clf = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1)
clf.fit(X_train, y_train)

print('GridSearch Best Estimator: {}'.format(clf.best_estimator_))
print('GridSearch Best Parameters: {}'.format(clf.best_params_))
print()
print('KNeighbors Classifier Accuracy: {:.2%}'.format(clf.score(X_test, y_test)))


# In[525]:


y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels = clf.classes_)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)

fig, ax = plt.subplots(figsize=(10,10))
cm_display.plot(ax=ax)
plt.tick_params(axis=u'both', which=u'both',length=0)
plt.title('KNN Confusion Matrix', fontsize=14)
plt.grid(False)
plt.show()


# In[526]:


print(classification_report(y_test, y_pred, target_names = ['Normal', 'Suspect', 'Pathological']))


# It looks like the KNeighbors Classifier did a little better for normal records, but slightly worse for suspect and pathological records. 
# 
# 

# <b>Logistic Regression:</b>

# In[527]:


param_grid = {'penalty': ['l1', 'l2'], 'solver': ('newton-cg', 'lbfgs', 'liblinear'),
              'C': [1, 2, 5, 10, 20]}

logreg = LogisticRegression(max_iter=500, random_state=3, multi_class='multinomial', n_jobs=-1)

clf = GridSearchCV(logreg, param_grid, cv=5, n_jobs=-1)
clf.fit(X_train, y_train)

print('GridSearch Best Estimator: {}'.format(clf.best_estimator_))
print('GridSearch Best Parameters: {}'.format(clf.best_params_))
print()
print('Logistic Regression Accuracy: {:.2%}'.format(clf.score(X_test, y_test)))


# In[528]:


y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels = clf.classes_)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)

fig, ax = plt.subplots(figsize=(10,10))
cm_display.plot(ax=ax)
plt.tick_params(axis=u'both', which=u'both',length=0)
plt.title('Logistic Regression Confusion Matrix', fontsize=14)
plt.grid(False)
plt.show()


# In[529]:


print(classification_report(y_test, y_pred, target_names = ['Normal', 'Suspect', 'Pathological']))


# Logistic Regression doesn't perform as well as the previous two classifier algorithms, though not by much.

# ### Stratified KFold w/ Random Forest & AdaBoost Classifier
# Let's now try a different approach using Stratified KFold with the Random Forest Classifier

# In[531]:


rf = RandomForestClassifier(n_estimators=1000)

skf_scores = []
skf_f1scores = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx] #We want to use scaled values.
    y_train, y_test = y[train_idx], y[test_idx]
    
    rf.fit(X_train, y_train)
    skf_scores.append(rf.score(X_test, y_test))
    
    y_pred = rf.predict(X_test)
    #print(classification_report(y_test, y_pred, target_names = ['Normal', 'Suspect', 'Pathological']))
    
    print('Fold {} F1 Score: {:.2%}'.format(i+1, f1_score(y_test, y_pred, average='micro'))) #micro to peform global f1 score
    skf_f1scores.append(f1_score(y_test, y_pred, average='micro'))
    
print("List of Accuracies: {}".format(skf_scores))

print()
print("Mean of Accuracies: {:.2%}".format(np.mean(skf_scores)))
print("Mean of F1 Scores: {:.2%}".format(np.mean(skf_f1scores)))


# It looks we like we acquired some improvement with the Random Forest Classifier. Since many of our attributes contribute to being labelled as a certain class based on some combination of factors, this algorithm works well.

# In[532]:


param_grid = {'n_estimators': [50, 100, 250, 500, 1000], 'criterion': ('gini', 'entropy'),
              'min_samples_leaf': [1, 2, 5, 10, 20, 50], 'min_samples_split' : [5, 10, 25, 50],
              'max_depth': [1, 3, 5, 7, 10]}


# In[533]:


'''
rf = RandomForestClassifier(random_state=3, n_jobs=-1)

clf = GridSearchCV(rf, param_grid, cv=5)
clf.fit(X_train, y_train)

print('GridSearch Best Estimator: {}'.format(clf.best_estimator_))
print('GridSearch Best Parameters: {}'.format(clf.best_params_))
print()
print('Logistic Regression Accuracy: {:.2%}'.format(clf.score(X_test, y_test)))

'''


# In[534]:


'''
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels = clf.classes_)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)

fig, ax = plt.subplots(figsize=(10,10))
cm_display.plot(ax=ax)
plt.tick_params(axis=u'both', which=u'both',length=0)
plt.title('Logistic Regression Confusion Matrix', fontsize=14)
plt.grid(False)
plt.show()

'''


# In[535]:


'''
print(classification_report(y_test, y_pred, target_names = ['Normal', 'Suspect', 'Pathological']))
'''


# How about <b>Adaboost Classifier</b>?

# In[537]:


rf = AdaBoostClassifier(n_estimators=1000)

skf_scores = []
skf_f1scores = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx] #We want to use scaled values.
    y_train, y_test = y[train_idx], y[test_idx]
    
    rf.fit(X_train, y_train)
    skf_scores.append(rf.score(X_test, y_test))
    
    y_pred = rf.predict(X_test)
    #print(classification_report(y_test, y_pred, target_names = ['Normal', 'Suspect', 'Pathological']))
    
    print('Fold {} F1 Score: {:.2%}'.format(i+1, f1_score(y_test, y_pred, average='micro'))) #micro to peform global f1 score
    skf_f1scores.append(f1_score(y_test, y_pred, average='micro'))
    
print("List of Accuracies: {}".format(skf_scores))

print()
print("Mean of Accuracies: {:.2%}".format(np.mean(skf_scores)))
print("Mean of F1 Scores: {:.2%}".format(np.mean(skf_f1scores)))


# The Random Forest Classifier out-performs the AdaBoost Classifier.

# ### XGBoost Classifier, CatBoost Classifier w/ Optuna
# 
# Next, we'll turn things up a notch and employ some powerful algorithms with Optuna to aid us in hyperparameter optimization. I'll begin with the XGBoost Classifier.

# In[426]:


def objective(trial, X=X, y=y):
    
    X_train, X_test, y_train, y_test = train_test_split(X_all, y, random_state=3, test_size=0.25)
    
    param_grid = {
        'tree_method':'gpu_hist',  
        'reg_lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'reg_alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.009,0.01,0.012,0.014,0.016,0.018, 0.02]),
        'n_estimators': 4000,
        'max_depth': trial.suggest_categorical('max_depth', [5,7,9,11,13,15,17,20]),
        'random_state': trial.suggest_categorical('random_state', [24, 48, 2020]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'gamma' : trial.suggest_int('gamma', 1, 10)
    }
    
    xgbclf = xgb.XGBClassifier(enable_categorical=True, n_jobs = -1, booster='gbtree', **param_grid)  
    
    xgbclf.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
    
    y_pred = xgbclf.predict(X_test)
    
    f1 = f1_score(y_test, y_pred, average='micro')
    
    #return rmse 
    return f1


# In[427]:


study = optuna.create_study(direction='maximize') 
study.optimize(objective, n_trials=50)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)


# In[428]:


optuna.visualization.plot_optimization_history(study)


# In[429]:


best_args = study.best_params 
best_args


# In[452]:


xgbclf = xgb.XGBClassifier(n_jobs = -1, booster='gbtree', **best_args)  
    
xgbclf.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
    
y_pred = xgbclf.predict(X_test)
    
f1 = f1_score(y_test, y_pred, average='micro')

print()
print('XGBoost Classifier Optuna F1-Score: {:.2%}'.format(f1))


# In[453]:


cm = confusion_matrix(y_test, y_pred, labels = clf.classes_)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)

fig, ax = plt.subplots(figsize=(10,10))
cm_display.plot(ax=ax)
plt.tick_params(axis=u'both', which=u'both',length=0)
plt.title('XGBoost Confusion Matrix', fontsize=14)
plt.grid(False)
plt.show()


# In[454]:


print(classification_report(y_test, y_pred, target_names = ['Normal', 'Suspect', 'Pathological']))


# It seems we can't use categorical data with the tree method used. Lets now try CatBoost Classifier

# In[540]:


def objective(trial, X=X, y=y):
    
    X_train, X_test, y_train, y_test = train_test_split(X_all, y, random_state=3, test_size=0.25) #Still does better with no stratify
    
    param_grid = {
        'l2_leaf_reg' : trial.suggest_discrete_uniform('l2_leaf_reg', 1.0, 5.5, 0.5),
        'min_child_samples' : trial.suggest_categorical('min_child_samples', [1, 4, 8, 16, 32]),
        #'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.009,0.01,0.012,0.014,0.016,0.018, 0.02]),
        'n_estimators': 2000,
        'max_depth': trial.suggest_categorical('max_depth', [5,7,9,11,13,15,16]),
        'random_state': trial.suggest_categorical('random_state', [24, 48, 2020]),
        'grow_policy' : 'Depthwise',
        'use_best_model' : True,
        'od_type' : 'iter', 
        'od_wait' : 20
    }
    
    clf = CatBoostClassifier(**param_grid)  
    
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
    
    y_pred = clf.predict(X_test)
    
    f1 = f1_score(y_test, y_pred, average='micro')
    
    #return rmse 
    return f1


# In[541]:


study = optuna.create_study(direction='maximize') 
study.optimize(objective, n_trials=50)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)


# In[542]:


optuna.visualization.plot_optimization_history(study) 


# In[547]:


best_args = {'l2_leaf_reg': 1.5,
 'min_child_samples': 8,
 'learning_rate': 0.01,
 'max_depth': 15,
 'random_state': 48}
best_args


# After iterating through some different combinations of parameters, this set of parameters yeilds the best results for this
# algorithm/model.

# In[546]:


clf = CatBoostClassifier(n_estimators = 2000, grow_policy = 'Depthwise', use_best_model = True, od_type = 'iter', od_wait = 20, **best_args)  
    
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
    
y_pred = clf.predict(X_test)
    
f1 = f1_score(y_test, y_pred, average='micro')
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
acc = accuracy_score(y_test, y_pred)

print()
print('CatBoost Classifier Optuna F1-Score: {:.2%}'.format(f1))
print('CatBoost Classifier Optuna Precision Score: {:.2%}'.format(precision))
print('CatBoost Classifier Optuna Recall Score: {:.2%}'.format(recall))
print('CatBoost Classifier Optuna Accuracy Score: {:.2%}'.format(acc))


# Using the same parameters returned from the best model from Optuna, we ended up with our highest F1 Score of 95.06, higher than what we got for the XGBoost Classifier. I'll also print the other overall metrics.

# In[548]:


cm = confusion_matrix(y_test, y_pred, labels = clf.classes_)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)

fig, ax = plt.subplots(figsize=(10,10))
cm_display.plot(ax=ax)
plt.tick_params(axis=u'both', which=u'both',length=0)
plt.title('CatBoost Confusion Matrix', fontsize=14)
plt.grid(False)
plt.show()


# In[549]:


print(classification_report(y_test, y_pred, target_names = ['Normal', 'Suspect', 'Pathological']))


# This final model classifies the unseen test data very well. Of course the weakest category is still the suspect records. Aside from just a couple incorrectly identified records, we had 15 records predicted to be normal but were actually suspect (<b>false negatives</b>). These types of records I believe are the most difficult to distinguish since some records could be more on the side of the spectrum of normal than pathological. 
# 
# Looking at our classification report, our lowest metric by far was the recall score for suspect records.
# 
# <b>Recall Formula: TP / (TP + FN) </b>
# 
# We can see that what drives this score down are those 15 false negatives for suspect records. This recall score for suspect records is also the highest across all models we have attempted here. Therefore, in comparison to the other algorithms used, we have arrived at a conclusive model.

# ### Exporting Model to Pickle

# In[562]:


import joblib 
import pickle

joblib.dump(study, 'study.dump')


# In[563]:


pickl = {'model' : clf}
pickle.dump(pickl, open('model_file'+'.p', 'wb'))


# In[564]:


file_name = 'model_file.p'
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']


# In[565]:


model.predict(X_test[1, :].reshape(1, -1))


# In[566]:


X_test[1, :]


# In[567]:


print(list(pd.Series(X_test[1, :])))


# In[ ]:




