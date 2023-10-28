## Prediction of Heart Disease

The heart is an essential part of the human body. Heart related diseases such as coronary heart disease (CHD) and myocardial infarction can lead to death. Predicting such diseases based on various symptoms and habits of human beings in advance may help to cure them in early stages. However, prediction and diagnosis of heart disease requires more perfection and precision since a little carelessness or small mistake can lead to several other diseases or death of a person. Machine learning, a branch of Artificial Intelligence (AI), can be used in predicting these heart diseases. In this project, we will build various machine learning algorithms such as logistic regression and decision trees using R programming and calculate the prediction accuracy and area under the curve for each model to find the best model for predicting heart disease. We used RStudio programming to implement these algorithms. 

### Dataset

The dataset was taken from Kaggle. This data set originally came from a CDC survey in the year 2020 which was created to determine the health status of U.S. residents. The original data set contains 401,958 samples and 279 attributes, the description of which can be found [here](https://www.cdc.gov/brfss/annual_data/2020/pdf/codebook20_llcp-v2-508.pdf). Since some of the information is not significant in predicting heart disease, the data set was further refined and cleaned by a user on Kaggle, which is the data set used for this project. 

The sample size of the cleaned data set is 319,795 observations (survey respondents) across 18 variables. There are no missing values within the data set.

### Methodology

<img width="550" alt="image" src="https://github.com/UmaBetageri/Heart-Disease-Prediction/assets/134670470/cc955ab9-7cac-4a60-9805-5905585fe1ed">

### Results and Conclusions


<img width="430" alt="image" src="https://github.com/UmaBetageri/Heart-Disease-Prediction/assets/134670470/33b48517-16b3-4d02-97ab-12bd6f737606">

<img width="520" alt="image" src="https://github.com/UmaBetageri/Heart-Disease-Prediction/assets/134670470/c276791d-4492-4d50-8fc1-25fa8e6a3190">

Table: Comparison of Various Machine Learning Models

All of the models were able to predict heart disease with greater than 80% accuracy, and 6 of the 8 models were more than 90% accurate. If we only consider the accuracy of the predictions, the Logistic Regression (91.56%), Random Forest (91.48%), and Decision Tree (91.39%) models are the best. However, since most of the individuals in the data set do not have heart disease, the ability of the model to successfully predict both the presence and absence of heart disease should be considered when evaluating the success of each model. For instance, the Decision Tree was 91.39% accurate overall, but was completely unsuccessful in predicting heart disease in people that had the disease. By plotting the ROC curve and evaluating the AUC to determine the proportion of true positive results, we can better understand the best models to predict heart disease, instead of relying on accuracy alone. Based on AUC, the best model for predicting heart disease is Logistic Regression (0.843).



