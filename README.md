## Fetal Health Condition Classifier - Overview:

* Created a model to classify for babies in fetal development, cardiotocography exams on whether the fetus has normal health conditions, is suspect of having some pathology, or has some pathological condition.  

* Engineered new features utilizing Linear Discriminant Analysis and KMeans Clustering. I also performed PCA to orchestrate further and visualize further class separability, but chose not to include the components as features due to the number of components and potential overfitting (<i>could attempt to build new models with them in the future however, if I come back to this project</i>).

* Began model building with Support Vector Machine, K-Nearest Neighbors, Logistic Regression, Random Forest Classifier, and AdaBoost Classifier. I then used optuna for hyperparameter optmimization with XGBoost Classifier and CatBoost Classifier.

* Created an API for potential clients using Flask (picture of an example input of data included).

* <b>UPDATE:</b> I recently uploaded and included results when implementing artifical data using SMOTE. Using this algorithm greatly enchanced evaluation metrics, but in a practical setting, we have to keep in mind this is artificial data and may not be completely representtative of real-world data. The current flask still uses the CatBoost model without SMOTE.

The pickled model containing the SMOTE balanced data of the CatBoost Classifier is too large to upload to GitHub (44MB). Therefore, that model will have to remain local for the time being. If you would like access to the model file, feel free to contact me.


## Code and Resources Used:

**Python Version:** 3.8.5

**Packages:** numpy, pandas, scipy, matplotlib, seaborn, sklearn, xgboost, catboost, sklearn packages that include but are not limited to: (pca, lda, kmeans, random forest classifier, svm, logisitc regression, various preprocessing and model selection packages), optuna

**Web Framework Requirements Command:** ```pip install -r requirements.txt```

## References:

* Various project structure and process elements were learned from Ken Jee's YouTube series: 
https://www.youtube.com/watch?v=MpF9HENQjDo&list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t

* Wikipedia article concerning cardiotocography exams and information about them: 
https://en.wikipedia.org/wiki/Cardiotocography#:~:text=Cardiotocography%20(CTG)%20is%20a%20technique,monitoring%20is%20called%20a%20cardiotocograph

* <b>Disclaimer regarding the data:</b> I did NOT scrape or collect this data myself originally. The data used for this project was obtained from Kaggle (source in the following link):
https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification

## Data Cleaning

Since the data was already thoroughly cleaned upon obtaining the dataset, there is very minimal cleaning tasks carried out in this particular project. I did perform some outlier detection, but chose to include any existing outliers as they could be important towards the analysis given the size of the dataset and the integrity of how the data was collected.

## EDA
Some key findings including an example of a test datapoint being classified in using Flask is included below. I noticed that there was a consistent heartrate baseline for pathological records having higher prolonged decelerations. You can then see for higher values of prolonged decelerations (far right column) in the pairplot, there are lower values for accelerations and uterine contractions in general than normal records. 

I also include the separation identified by PCA and how these distinctions in the cardiotocography exam metrics can be used to identify whether a datapoint is normal or pathological. In addition, there is also the application of LDA which results in a 2D visual where the three class distinctions can be seen between the linear discriminants. 

<b>Update:</b> I include a graph of the ROC AUC Curve scores for the CatBoost Classifier using the SMOTE algorithm data included.

Target Labels can be translated as follows (according to the data source, each record was labelled by three Obstetritians):

<b>1.0 -> Normal</b>

<b>2.0 -> Suspect</b>

<b>3.0 -> Pathological</b>

![alt text](https://github.com/elayer/Fetal-Health-Classifier-Project/blob/main/prolonged-to-baseline.png "Baseline Heartrate to Prolonged Decelerations")
![alt text](https://github.com/elayer/Fetal-Health-Classifier-Project/blob/main/pairplot-pathological-pattern.png "Pairplot for Pathological Patterns")
![alt text](https://github.com/elayer/Fetal-Health-Classifier-Project/blob/main/pca-patterns.png "PCA Patterns")
![alt text](https://github.com/elayer/Fetal-Health-Classifier-Project/blob/main/lda-visual.png "LDA Visual")
![alt text](https://github.com/elayer/Fetal-Health-Classifier-Project/blob/main/catboost-roc.png "CatBoost ROC AUC Score")

## Model Building
Before building any models, I included the linear discriminants from my LDA application as well as clusters created from applying KMeans Clustering to the dataset as new features. I then scaled the data using MinMaxScaler for the Support Vector Machine implementation, and StandardScaler for all other models attempted. 

* I began model testing with the Support Vector Machine, since we are interested in creating an optimal class seperability, and then attempted K-Nearest Neighbors and Logistic Regression to compare it with these different algorithms. Oddly enough, the models performed better without stratification, but in practice, it may be more beneficial and conducive to practicality to stratify the the testing data since there were more normal records than pathological records. 

* This then led me to try Random Forest and AdaBoost Classifier in tandem with StratifiedKFold to ensure balanced classes while training. The Random Forest Classifier performed better on all folds.

* I then concluded using optuna with XGBoost and CatBoost Classifiers. As expected, the CatBoost Classifier yielded the best results out of all the models attempted. 

(<i>As a potential point to try in the future, I wonder how well the models would perform if including some newly sampled data to balance the records on the class targets. SMOTE is a potential method to perform this</i>).

## Model Performance
The Random Forest and CatBoost classifier models had the best two performances, with CatBoost being the top model. These models performed a little better over the previous models of Support Vector Machine, K-Nearest Neighbors, and Logistic Regression Models attempted. Below are the recorded <b>Weighted F1 Scores and Accuracies</b> for each of the models performed:

* Support Vector Machine F1 Score: 91%, Accuracy: 91.54% --- <i>with SMOTE</i> F1 Score: 96%, Accuracy: 96.46%

* K-Nearest Neighbors F1 Score: 92%, Accuracy: 92.11% --- <i>with SMOTE</i> F1 Score: 98%, Accuracy: 97.91%

* Logistic Regression F1 Score: 91%, Accuracy: 90.60% --- <i>with SMOTE</i> F1 Score: 87%, Accuracy: 86.71%

* Random Forest Classifier F1 Score: 94.40%, Accuracy: 94.40% --- <i>with SMOTE</i> F1 Score: 97.82%, Accuracy: 97.82%

* AdaBoost Classifier F1 Score: 84.24%, Accuracy: 84.24% --- <i>with SMOTE</i> F1 Score: 85.80%, Accuracy: 85.80%

* XGBoost Classifier F1 Score: 92.24%, Accuracy: 92% --- <i>with SMOTE</i> F1 Score: 97.48%, Accuracy: 97%

* CatBoost Classifier F1 Score: 95.06%, Accuracy: 95.06% --- <i>with SMOTE</i> F1 Score: 98.99%, Accuracy: 98.99%

I used Optuna with XGBoost and CatBoost to build an optimized model since these algorthms include a myriad of attributes to test when trying to optimize them (hyperparameters).

## Productionization
I lasted created a Flask API hosted on a local webserver. For this step I primarily followed the productionization step from the YouTube tutorial series found in the refernces above. This endpoint could be used to access, given cardiotocography exam results, a prediction for whether that pregnancy has normal health conditions or a pathological concern.

<b>UPDATE:</b> Below is an example prediction using the Flask API:

![alt text](https://github.com/elayer/Fetal-Health-Classifier-Project/blob/main/fetal_homepage.png "Website Homepage")
![alt text](https://github.com/elayer/Fetal-Health-Classifier-Project/blob/main/fetal_prediction.png "Website Prediction Page")

## Future Improvements
In the case we do not wish to use artificial data, we can simply remove the SMOTE implementation and stratify the dependent variable when splitting the data into train and test sets. From here, we can run the same models knowing the data is genuine.

5/27/2022: Applying LDA using scaled data instead of unscaled data.
