## BA - 749 - R Code for Final Project ##

## Title: Prediction of Heart Disease

## Group - 5E: Umadevi, Duc, Alexandra and Jeffrey

## Project Title: Prediction of Heart Disease

rm(list=ls())

# Import Data set 
heart = read.csv("heart_2020_cleaned.csv", header=T, na.strings="?")
head(heart)

# Sample Size of Data set
dim(heart)

## Data Cleaning 
# Remove missing values in data set
heart = na.omit(heart) 

# Data Processing 
# Conversion of Characters to Numerical Values - Age is Quantitative Variable
heart$AgeGroup = 13
heart$AgeGroup[heart$AgeCategory=="18-24"]= 1
heart$AgeGroup[heart$AgeCategory=="25-29"]= 2
heart$AgeGroup[heart$AgeCategory=="30-34"]= 3
heart$AgeGroup[heart$AgeCategory=="35-39"]= 4
heart$AgeGroup[heart$AgeCategory=="40-44"]= 5
heart$AgeGroup[heart$AgeCategory=="45-49"]= 6
heart$AgeGroup[heart$AgeCategory=="50-54"]= 7
heart$AgeGroup[heart$AgeCategory=="55-59"]= 8
heart$AgeGroup[heart$AgeCategory=="60-64"]= 9
heart$AgeGroup[heart$AgeCategory=="65-69"]= 10
heart$AgeGroup[heart$AgeCategory=="70-74"]= 11
heart$AgeGroup[heart$AgeCategory=="75-79"]= 12


# Conversion of string value to factors with various levels
heart$HeartDisease <-as.factor(heart$HeartDisease)
heart$Smoking <-as.factor(heart$Smoking)
heart$AlcoholDrinking <-as.factor(heart$AlcoholDrinking)
heart$Stroke <-as.factor(heart$Stroke)
heart$DiffWalking <-as.factor(heart$DiffWalking)
heart$Sex <-as.factor(heart$Sex)
heart$Race <-as.factor(heart$Race)
heart$Diabetic <-as.factor(heart$Diabetic)
heart$PhysicalActivity <-as.factor(heart$PhysicalActivity)
heart$GenHealth <-as.factor(heart$GenHealth)
heart$Asthma <-as.factor(heart$Asthma)
heart$KidneyDisease <-as.factor(heart$KidneyDisease)
heart$SkinCancer <-as.factor(heart$SkinCancer)
heart$AgeCategory <-as.factor(heart$AgeCategory)

# Removing Age category from the data set
heart$AgeCategory = NULL
dim(heart)

# Summary of data set
summary(heart)

# Structure of the data set
str(heart)

# Standard Deviation of Quantitative variables.
sd(heart$BMI)
sd(heart$PhysicalHealth)
sd(heart$MentalHealth)
sd(heart$SleepTime)
sd(heart$AgeGroup)

# Correlation matrix for quantitative variables
install.packages("corrplot")
library(corrplot)
cor = cor(heart[, c('BMI','PhysicalHealth','MentalHealth', 'SleepTime', 'AgeGroup')])
corrplot(cor, order = 'AOE', addCoef.col = 'black', tl.pos = 'd')

# Count of each class of response variable
barplot(table(heart$HeartDisease), col=c("darkgreen","red"), main="Count of each heart disease class")


# Exploratory Data Analysis

# Histograms of quantitative Variables
library(ggplot2)

# Plot for Age Group
ggplot(heart,aes(x=AgeGroup,fill=HeartDisease,color=HeartDisease)) + geom_histogram(binwidth = 1,color="black") + labs(x = "AgeGroup", y = "Frequency", title = "Heart Disease w.r.t. Age") + scale_x_continuous(breaks=0:15)

# Plot for BMI
ggplot(heart,aes(x=BMI,fill=HeartDisease,color=HeartDisease)) + geom_histogram(binwidth = 3,color="black") + labs(x = "Body Mass Index", y = "Frequency", title = "Heart Disease w.r.t. BMI") 

# Plot for Sleep Time
ggplot(heart,aes(x=SleepTime,fill=HeartDisease,color=HeartDisease)) + geom_histogram(binwidth = 2,color="black") + labs(x = "Sleep Time", y = "Frequency", title = "Heart Disease w.r.t. Sleep Time")

# Plot for Physical Health
ggplot(heart,aes(x=PhysicalHealth,fill=HeartDisease,color=HeartDisease)) + geom_histogram(binwidth = 2,color="black") + labs(x = "Physical Health", y = "Frequency", title = "Heart Disease w.r.t. Physical Health")

# Plot for Mental Health
ggplot(heart,aes(x=MentalHealth,fill=HeartDisease,color=HeartDisease)) + geom_histogram(binwidth = 2,color="black") + labs(x = "Mental Health", y = "Frequency", title = "Heart Disease w.r.t. Mental Health") 


# Pie Chart for General Health Conditions
mytable <- table(heart$GenHealth)
pct<-round(mytable/sum(mytable)*100)
lbls1<-paste(names(mytable),pct)
lbls<-paste(lbls1, "%", sep="")
pie(mytable, labels = lbls,col = rainbow(length(lbls)),main="Health Condition",radius = 0.9)

# Box Plots for quantitative variables
boxplot(SleepTime~HeartDisease, data=heart, ylab="Sleep Time" , xlab="Heart Disease")
boxplot(AgeGroup~HeartDisease, data=heart, ylab="Age Group" , xlab="Heart Disease")
boxplot(PhysicalHealth~HeartDisease, data=heart, ylab="Physical Health" , xlab="Heart Disease")
boxplot(MentalHealth~HeartDisease, data=heart, ylab="Mental Health" , xlab="Heart Disease")
boxplot(BMI~HeartDisease, data=heart, ylab="BMI" , xlab="General Health")
boxplot(BMI~GenHealth, data=heart, ylab="BMI" , xlab="General Health")
boxplot(BMI~PhysicalActivity, data=heart, ylab="BMI" , xlab="General Health")

# Bar Plots
counts <- table(heart$HeartDisease, heart$Race)
barplot(counts, main="Impact of Races on Heart Disease", xlab="Race", col=c("darkblue","red"), legend = rownames(counts))

counts <- table(heart$HeartDisease, heart$Smoking)
barplot(counts, main="Impact of Smoking on Heart Disease", xlab="Smoking", col=c("darkblue","red"), legend = rownames(counts))

counts <- table(heart$HeartDisease, heart$Asthma)
barplot(counts, main="Impact of Asthma on Heart Disease", xlab="Asthma", col=c("darkblue","red"), legend = rownames(counts))

counts <- table(heart$HeartDisease, heart$Sex)
barplot(counts, main="Impact of Sex on Heart Disease", xlab="Sex", col=c("darkblue","red"), legend = rownames(counts))

counts <- table(heart$HeartDisease, heart$PhysicalActivity)
barplot(counts, main="Impact of PhysicalActivity on Heart Disease", xlab="PhysicalActivity", col=c("darkblue","red"), legend = rownames(counts))


## Data partition - Cross Validation

# 70% training and 30% testing data
n = nrow(heart)
set.seed(100)
sample = sample(1:n, replace = FALSE) 
train.index = sample[1:(0.7*n)]
train = heart[train.index, ]
test = heart[-train.index, ]

# length of splits 
dim(train)
dim(test)

### Machine Learning Models to predict Heart Disease

## Model-1: Logistic Regression ##

# Building the model on training data set
glm = glm(HeartDisease~., data=train, family = "binomial")
summary(glm)

# Predictions of testing data
glm_pred = predict(glm, test, type = "response")

# Setting threshold value of 0.5
glm_class = rep("No", nrow(test))
glm_class[glm_pred > .5] = "Yes"

# Computing confusion matrix for glm
library(caret)
glm_cm = confusionMatrix(data= as.factor(glm_class), reference = test$HeartDisease, positive = "Yes")
glm_cm

## Plotting ROC for glm
library(pROC)
glm_roc = roc(test$HeartDisease~ glm_pred, plot = TRUE, print.auc = TRUE)
auc(glm_roc)


## Model-2 : Linear Discriminant Analysis ##

# Building the LDA model on training data set
library(MASS)
lda = lda(HeartDisease~., data=train)
summary(lda)

# Predictions on testing data
lda_pred = predict(lda, test)
lda_class = lda_pred$class

# Computing confusion matrix for LDA
lda_cm= confusionMatrix(data= as.factor(lda_class), reference=test$HeartDisease, positive = "Yes")
lda_cm

# Plotting ROC for LDA
lda_roc = roc(response= test$HeartDisease, predictor=lda_pred$posterior[,1], plot = TRUE, print.auc = TRUE)  #ROC curve
auc(lda_roc)


## Model-3: Quadratic Discriminant Analysis ##

# Building the QDA model on training data set
library(MASS)
qda = qda(HeartDisease~., data=train)
summary(qda)

# Predictions on testing data
qda_pred = predict(qda, test)
qda_class = qda_pred$class

# Computing confusion matrix for QDA
qda_cm= confusionMatrix(data= as.factor(qda_class), reference=test$HeartDisease, positive = "Yes")
qda_cm

# Plotting ROC for QDA
qda_roc = roc(response= test$HeartDisease, predictor=qda_pred$posterior[,1], plot = TRUE, print.auc = TRUE)  #ROC curve
auc(qda_roc)


## Model-4: Naive Bayes ##

# Building the NB model on training data set
library(e1071)
nb = naiveBayes(HeartDisease~., data=train)
nb

# Predictions on testing data of NB
nb_class = predict(nb, test)
nb_pred = predict(nb, test, type = "raw")

# Computing confusion matrix of NB
nb_cm= confusionMatrix(data= as.factor(nb_class), reference=test$HeartDisease, positive = "Yes")
nb_cm

# Plotting ROC for NB
nb_roc = roc(response= test$HeartDisease, predictor=nb_pred[,1], plot = TRUE, print.auc = TRUE)  #ROC curve
auc(nb_roc)


# Model-5: Decision Tree ##

library(tree)
tree = tree(HeartDisease~., data=train)
tree
plot(tree)
text(tree, pretty = 0)
summary(tree)

# Prediction of the response on the test data
tree_pred = predict(tree, newdata = test, type = 'class')
summary(tree_pred)
tree_prob = predict(tree, newdata = test)

# confusion matrix of Decision Tree
tree_cm = confusionMatrix(data = tree_pred, reference = test$HeartDisease, positive = "Yes")
tree_cm

# ROC Curve for Decision Tree
tree_roc = roc(response= test$HeartDisease, predictor=tree_prob[,2], plot = TRUE, print.auc = TRUE)
auc(tree_roc)


## Pruned Tree ##

set.seed(100)

# cv to determine the optimal tree size
cv = cv.tree(tree, FUN= prune.misclass)
cv

# Best size
best = cv$size[which.min(cv$dev)]
best

# Plot of cv
plot(cv$size, cv$dev, type='b')

# pruned tree corresponding to the optimal tree size
prune = prune.misclass(tree, best=5)
plot(prune)
text(prune, pretty=0)
summary(prune)

# Prediction of the response on the test data
prun_prob = predict(prune, newdata = test)
summary(prun_prob)

prun_pred = predict(prune, test, type = 'class')
summary(prun_pred)

# Confusion matrix for prune
prun_cm = confusionMatrix(data = prun_pred, reference = test$HeartDisease, positive = "Yes")
prun_cm

# ROC curve for prune
prune_roc = roc(response= test$HeartDisease, predictor=prun_prob[,2], plot = TRUE, print.auc = TRUE)
auc(prune_roc)

## Results are same for both decision tree and pruned tree


## Model-6: Bagging ##
library(randomForest)
set.seed(100)

# Building the bag model on training data set
bag = randomForest(HeartDisease~., data = train, mtry = 17, importance = TRUE)
bag

# Predictions on testing data
bag_pred = predict(bag, newdata = test, type = 'class')
summary(bag_pred)

# Probabilities
bag_prob = predict(bag, test, index=2, type="prob")

# Confusion matrix for bag
bag_cm = confusionMatrix(data = bag_pred, reference = test$HeartDisease, positive = "Yes")
bag_cm

# Plotting ROC Curve for bag
bag_roc = roc(response = test$HeartDisease, predictor= bag_prob[,2], plot = TRUE, print.auc = TRUE)
auc(bag_roc)

# variables which are most important
importance(bag)
varImpPlot(bag)


## Model-7: Random Forest ##

## RF model with m = 2

set.seed(100)

# Building the RF model on training data set
rf_1 = randomForest(HeartDisease~., data = train, mtry = 2, importance = TRUE)
rf_1

# Predictions
rf_1_pred = predict(rf_1, newdata = test, type = 'class')
summary(rf_1_pred)

# Finding Probabilities
rf_1_prob = predict(rf_1, test, index=2, type="prob")

# Confusion matrix for RF
rf_1_cm = confusionMatrix(data = rf_1_pred, reference = test$HeartDisease, positive = "Yes")
rf_1_cm

rf_1_roc = roc(response = test$HeartDisease, predictor= rf_1_prob[,2], plot = TRUE, print.auc = TRUE)
auc(rf_1_roc)

# variables which are most important
importance(rf_1)
varImpPlot(rf_1)


## Random Forest model for m = 8

set.seed(100)

# Building the RF model on training data set
rf_2 = randomForest(HeartDisease~., data = train, mtry = 8, importance = TRUE)
rf_2

# Predictions on testing data
rf_2_pred = predict(rf_2, newdata = test, type='class')
summary(rf_2_pred)

# Probabilities for RF_2
rf_2_prob = predict(rf_2, test, index=2, type="prob")

# Confusion matrix for random forest with m=8
rf_2_cm = confusionMatrix(data = rf_2_pred, reference = test$HeartDisease, positive = "Yes")
rf_2_cm

# ROC Curve and AUC
rf_2_roc = roc(response= test$HeartDisease, predictor=rf_2_prob[,2], plot = TRUE, print.auc = TRUE)
auc(rf_2_roc)

# variables which are most important
importance(rf_2)
varImpPlot(rf_2)


# Comparison of AUC for all the Models
auc(glm_roc) # Highest.
auc(lda_roc)  
auc(qda_roc)
auc(nb_roc)
auc(tree_roc)
auc(bag_roc)
auc(rf_1_roc)
auc(rf_2_roc)

# Comparison of ROC Curves for all the models
ggroc(list(glm=glm_roc, lda=lda_roc, qda=qda_roc, nb=nb_roc, tree=tree_roc, bag=bag_roc, rf.m2=rf_1_roc, rf.m8=rf_2_roc))

### End of the coding ###

