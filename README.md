## Fetal Health Condition Classifier - Overview:

* Created a model to classify for babies in fetal development, cardiotocography exams on whether the fetus has normal health conditions, is suspect of having some pathology, or has some pathological condition.  

* Engineered new features utilizing Linear Discriminant Analysis 

* Began model building with Linear, Lasso, Ridge, and ElasticNet linear models, as well Random Forest regression. Then, built optimized models using Optuna with XGBoost and CatBoost regression 

* Created an API for potential clients using Flask. 


## Code and Resources Used:

**Python Version:** 3.8.5

**Packages:** numpy, pandas, scipy, matplotlib, seaborn, sklearn, xgboost, catboost, sklearn (pca, lda, kmeans, random forest classifier, preprocessing packages)

**Web Framework Requirements Command:** ```pip install -r requirements.txt```

## References:

* Various project structure and process elements were learned from Ken Jee's YouTube series: 
https://www.youtube.com/watch?v=MpF9HENQjDo&list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t

* Wikipedia article concerning cardiotocography exams and information about them: 
https://en.wikipedia.org/wiki/Cardiotocography#:~:text=Cardiotocography%20(CTG)%20is%20a%20technique,monitoring%20is%20called%20a%20cardiotocograph

* <b>Disclaimer regarding the data:</b> I did NOT scrape or collect this data myself. The data used for this project was obtained from Kaggle (source in the following link):
https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification

## Web Scraping:

Created a web scraper using Requests and Beauitful Soup. From each product listing page from Amazon, the following information was obtained:
*   Brand
*   Avg. Ratings
*   Number of Ratings
*   Processor Type
*   RAM
*   Disk Size
*   Processor Speed
*   Bluetooth
*   Liquid Cooled
*   Price

## Data Cleaning

After collecting data, I performed several steps to clean and refine the data to prepare for further processing and model building. I went through the following steps to clean and prepare the data:

* Parsed the brand name out of the general product information collected from the listings

* Created Liquid Cooled and Bluetooth attributes by parsing the product information for if it contained the capability for these features

* Coalesced the processor types written differently to be uniform across the data and removed any outliers 

* Reformatted the price, number of ratings, and avg. ratings columns to be appropriate numeric values. Dropped rows that had no price target variable

* Reformatted and rescaled Processor speed, RAM, and disk size to numeric values and scaled each attribute to GB

*   Removed outliers from the data that had very extreme value using <b>Z-Score</b>

* Ordinally encoded processor tpyes and dropped the records that were extremely underrepresented

* Created dummy variables for the brands of the computers that were not extremely underrepresented

## EDA
Some noteable findings from performing exploratory data analysis can be seen below. When going from a low to more high-end processor, the price of a computer does indeed increase. The same applies to RAM. In addition, I noticed some brands were priced higher even with similar or lower amounts of disk space. I eventually found that just as big of a driver in price was the brand of a computer, and not only the specs.

![alt text](https://github.com/elayer/Amazon-Computer-Project/blob/main/price-by-processor-type.png "Processor Type Boxplots")
![alt text](https://github.com/elayer/Amazon-Computer-Project/blob/main/price-histogram.png "Price Distribution")
![alt text](https://github.com/elayer/Amazon-Computer-Project/blob/main/price-by-RAM-boxplots.png "RAM Boxplots")
![alt text](https://github.com/elayer/Amazon-Computer-Project/blob/main/price-to-brand-lmplots.png "RAM vs. Price per Brand")

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
