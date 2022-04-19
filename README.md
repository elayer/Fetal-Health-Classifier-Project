## Fetal Health Condition Classifier - Overview:

* Created a model to classify for babies in fetal development, cardiotocography exams on whether the fetus has normal health conditions, is suspect of having some pathology, or has some pathological condition.  

* Engineered new features utilizing Linear Discriminant Analysis and KMeans Clustering. I also performed PCA to orchestrate further and visualize further class separability, but chose not to include the components as features due to the number of components and potential overfitting (<i>could attempt to build new models with them in the future however, if I come back to this project</i>).

* Began model building with Support Vector Machine, K-Nearest Neighbors, Logistic Regression, Random Forest Classifier, and AdaBoost Classifier. I then used optuna for hyperparameter optmimization with XGBoost Classifier and CatBoost Classifier.

* Created an API for potential clients using Flask (picture of an example input of data included).


## Code and Resources Used:

**Python Version:** 3.8.5

**Packages:** numpy, pandas, scipy, matplotlib, seaborn, sklearn, xgboost, catboost, sklearn packages that include but are not limited to: (pca, lda, kmeans, random forest classifier, svm, logisitc regression, various preprocessing and model selection packages), optuna

**Web Framework Requirements Command:** ```pip install -r requirements.txt```

## References:

* Various project structure and process elements were learned from Ken Jee's YouTube series: 
https://www.youtube.com/watch?v=MpF9HENQjDo&list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t

* Wikipedia article concerning cardiotocography exams and information about them: 
https://en.wikipedia.org/wiki/Cardiotocography#:~:text=Cardiotocography%20(CTG)%20is%20a%20technique,monitoring%20is%20called%20a%20cardiotocograph

* <b>Disclaimer regarding the data:</b> I did NOT scrape or collect this data myself. The data used for this project was obtained from Kaggle (source in the following link):
https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification

## Data Cleaning

Since the data was already thoroughly cleaned upon obtaining the dataset, there is very minimal cleaning tasks carried out in this particular project. I did perform some outlier detection, but chose to include any existing outliers as they could be important towards the analysis given the size of the dataset and the integrity of how the data was collected.

## EDA
Some key findings including an example of a test datapoint being classified in using Flask is included below. I noticed that there was a consistent heartrate baseline for pathological records having higher prolonged decelerations. You can then see for higher values of prolonged decelerations (far right column) in the pairplot, there are lower values for accelerations and uterine contractions in general than normal records. 

I also include the separation identified by PCA and how these distinctions in the cardiotocography exam metrics can be used to identify whether a datapoint is normal or pathological. In addition, there is also the application of LDA which results in a 2D visual where the three class distinctions can be seen between the linear discriminants. 

Lastly, I include a picture from the flask endpoint constructed, where an appropriate set of data corresponding to each attribute used in the model can be used to make a prediction on whether a record is normal, suspect, or pathological.

![alt text](https://github.com/elayer/Fetal-Health-Classifier-Project/blob/main/prolonged-to-baseline.png "Baseline Heartrate to Prolonged Decelerations")
![alt text](https://github.com/elayer/Fetal-Health-Classifier-Project/blob/main/pairplot-pathological-pattern.png "Pairplot for Pathological Patterns")
![alt text](https://github.com/elayer/Fetal-Health-Classifier-Project/blob/main/pca-patterns.png "PCA Patterns")
![alt text](https://github.com/elayer/Fetal-Health-Classifier-Project/blob/main/lda-visual.png "LDA Visual")
![alt text](https://github.com/elayer/Fetal-Health-Classifier-Project/blob/main/data-prediction-example.png "Test Data Prediction Example")

## Model Building
Before building any models, I transformed the categorical variables into appropriate numeric types. I transformed brand into dummy variables since some of the more expensive computers were similar in distribution of price, and the same goes for less expensive computers. I then ordinally encoded the processor types since each type seemed to have a steady increase in price as you improved the quality of the processor.

I first tried a few different linear models and some variations of them:

* starting with Linear regression, and then trying Lasso, Ridge, and ElasticNet to see if the results would change since we have many binary columns. 

* This then led me to try Random Forest, XGBoost, and CatBoost regression because of the sparse binary/categorical nature of most of the attributes in the data. 

## Model Performance
The Random Forest, XGBoost, and CatBoost regression models respectively had imrpoved performances. These models considerably outperformed the linear regression models I tried previously. Below are the R2 score values for the models:

* Linear Regression: 72.28 (the best of the linear models)

* Random Forest Regression: 83.76

* XGBoost Regression: 85.38

* CatBoost Regression: 85.87

I used Optuna with XGBoost and CatBoost to build an optimized model especially with the various attributes that these algorithms have.

## Productionization
I lasted created a Flask API hosted on a local webserver. For this step I primarily followed the productionization step from the YouTube tutorial series found in the refernces above. This endpoint could be used to take in certain aspects of a computer, make appropriate transformations to the variables, and return a predicted price for a computer.  
