#Brent Young
#MSDS 454
#Summer 2018
#Midterm Project

#Load Libraries

library(Hmisc)
library(Rcpp) #Missingness Map
library(pROC) #ROC Curve
library(ROCR) #ROC Curve, AUC
library(AUC)
library(Deducer) #ROC Curve
library(InformationValue) #ROC Curve
library(corrplot) #Data Visualization
library(ggcorrplot) #Data Visualization
library (ggplot2) #Data Visualization
library(GGally) #Data Visualization
library(lattice)#Data Visualization
library(gridExtra)
library(gmodels)#Cross-tabs
library(dplyr)
library(reshape2)
library(readr)
library(glm2)
library(aod)
library(rcompanion) 
library(leaps) #Best subsets
library(MASS) #Linear Discriminant Analysis
library(car)

library(boot) #bootstrap
library(leaps) #Best subset selection; stepwise
library(glmnet) #Ridge Regression & the Lasso
library(pls) #Principal components Regression
library(splines) #regression splines
library(gam) #Generalized Additive Models
library(akima)
library(caret) #Machine Learning
library(yardstick) #Machine Learning
library(class) #KNN
library(nnet) # Neural Network
library(tree) # Decision Trees 
library(randomForest) # Bagging and Random Forest
library(e1071) #Support Vector Machines, Naive Bayes Classifier
library(gbm) # Gradient Boosting Machines
library(adabag) #Boosting
library(xgboost) # Gradient Boosting Machines

#Load the data

setwd("~/R/MSDS 454")

raw_train <- read.csv("bd_train.csv")
raw_test <- read.csv("bd_test.csv")
raw_test["Made.Donation.in.March.2007"] <-NA
str(raw_test)
total <- rbind(raw_train,raw_test)

#total <- read.csv("bd_combined.csv") # Alternatively, load the "bd_combined.cv" file, manually combined train and test

names(total) <- make.names(names(total))  # Change variable names to make it easier in R
colnames(total)[1] <- "ID"  
colnames(total)[2] <- "MSLD"              # rename Months.since.Last.Donation
colnames(total)[3] <- "NUM"               # rename Number.of.Donations (total number of donations)
colnames(total)[4] <- "VOLUME"            # rename Total.Volume.Donated..c.c..
colnames(total)[5] <- "MSFD"              # rename Months.since.First.Donation
colnames(total)[6] <- "TARGET_FLAG"       # rename Target

#Descriptive Statistics
str(total)
summary(total) # No N/A's, except 200 N/As in test set as expected
describe(total)
dim(total)

#Feature Creation
total <- total %>% mutate(DPM = MSFD/NUM) # Donations Per Month
total <- total %>% mutate(TENRAT = MSLD/MSFD) #Ratio of Months since Last Donation to First Donation
total <- total %>% mutate(DF = (MSFD-MSLD/NUM)) #Donation Frequency

#MSLD_bin
total$MSLD_bin[total$MSLD >= 0 & total$MSLD <= 4] <- "0 to 4"
total$MSLD_bin[total$MSLD >= 5 & total$MSLD <= 8] <- "5 to 8"
total$MSLD_bin[total$MSLD >= 9 & total$MSLD <= 12] <- "9 to 12"
total$MSLD_bin[total$MSLD >= 13] <- "13+"
total$MSLD_bin <- factor(total$MSLD_bin)
total$MSLD_bin <- factor(total$MSLD_bin, levels=c("0 to 4","5 to 8","9 to 12","13+"))

#REPEAT
total$REPEAT <- as.factor(ifelse(total$NUM > 1, 1, 0 )) #Repeat Customer, Yes or No

#Create SQRT and LOG Transformations on Numeric Variables

total$SQRT_MSLD <- sqrt(total$MSLD)
total$SQRT_NUM <- sqrt(total$NUM)
total$SQRT_VOLUME <- sqrt(total$VOLUME)
total$SQRT_MSFD <- sqrt(total$MSFD)
total$SQRT_TENRAT <- sqrt(total$TENRAT)
total$SQRT_DF <- sqrt(total$DF)

total$LOG_NUM <- log(total$NUM)
total$LOG_VOLUME <- log(total$VOLUME)
total$LOG_MSFD <- log(total$MSFD)

################################# Quick High Level EDA for Overall Dataset #################################

#Descriptive Statistics
str(total)
summary(total) # No N/A's, except 200 N/As in test set as expected
describe(total)
dim(total)

#Check for Missingness
sapply(total, function(x) sum(is.na(x)))
sum(is.na(total))

####################################### Split Datasets ######################################
#Split Data into Train/Validation/Test 

train_set = total[1:576,]

#Recode the class labels to Yes/No (required when using class probs)
#train_set$TARGET_FLAG<-factor(train_set$TARGET_FLAG,levels = c(0,1),labels=c("No", "Yes"))
train_set$TARGET_FLAG <- as.factor(train_set$TARGET_FLAG)

str(train_set)

set.seed(123)
validationIndex <- createDataPartition(train_set$TARGET_FLAG, p=0.652, list=FALSE)

train <- train_set[validationIndex,]
validation <- train_set[-validationIndex,]
test = total[577:776,]

str(train)
summary(train)
str(validation)
str(test)
describe(test)

############################## EDA for Classification Models - Training Data ##############################

######EDA for Numeric Variables#####

par(mfrow=c(3,3))
hist(train$MSLD, col = "#A71930", xlab = "MSLD", main = "Histogram of MSLD")
hist(train$NUM, col = "#09ADAD", xlab = "NUM", main = "Histogram of NUM")
hist(train$VOLUME, col = "#DBCEAC", xlab = "VOLUME", main = "Histogram of VOLUME")
boxplot(train$MSLD, col = "#A71930", main = "Boxplot of MSLD")
boxplot(train$NUM, col = "#09ADAD", main = "Boxplot of NUM")
boxplot(train$VOLUME, col = "#DBCEAC", main = "Boxplot of VOLUME")

par(mfrow=c(1,1))

par(mfrow=c(2,2))
hist(train$MSFD, col = "#A71930", xlab = "MSFD", main = "Histogram of MSFD")
hist(train$DPM, col = "#09ADAD", xlab = "DPM", main = "Histogram of DPM")
boxplot(train$ MSFD, col = "#A71930", main = "Boxplot of MSFD")
boxplot(train$DPM, col = "#09ADAD", main = "Boxplot of DPM")
par(mfrow=c(1,1))

par(mfrow=c(2,2))
hist(train$TENRAT, col = "#A71930", xlab = "TENRAT", main = "Histogram of TENRAT")
hist(train$DF, col = "#09ADAD", xlab = "DF", main = "Histogram of DF")
boxplot(train$TENRAT, col = "#A71930", main = "Boxplot of TENRAT")
boxplot(train$DF, col = "#09ADAD", main = "Boxplot of DF")
par(mfrow=c(1,1))

#Outlier Analysis
quantile(train$MSLD, c(.01, .05, .95, .99))
quantile(train$NUM, c(.01, .05, .95, .99))
quantile(train$VOLUME, c(.01, .05, .95, .99))
quantile(train$MSFD, c(.01, .05, .95, .99))
quantile(train$DPM, c(.01, .05, .95, .99))
quantile(train$TENRAT, c(.01, .05, .95, .99))
quantile(train$DF, c(.01, .05, .95, .99))

#Boxplots for Numeric Variables
ggplot(train, aes(x=TARGET_FLAG, y= MSLD)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of MSLD") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(train, aes(x=TARGET_FLAG, y= NUM)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of NUM") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(train, aes(x=TARGET_FLAG, y= VOLUME)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of VOLUME") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(train, aes(x=TARGET_FLAG, y= MSFD)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of MSFD") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(train, aes(x=TARGET_FLAG, y= DPM)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of DPM") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(train, aes(x=TARGET_FLAG, y= TENRAT)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of TENRAT") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(train, aes(x=TARGET_FLAG, y= DF)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of DF") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#Correlation Matrix
subdatnumcor <- subset(train, select=c("MSLD","NUM","VOLUME","MSFD","DPM","TENRAT","DF"))

par(mfrow=c(1,1))  
corr <- round(cor(subdatnumcor),2)
ggcorrplot(corr, outline.col = "white", ggtheme = ggplot2::theme_gray, 
           colors = c("#E46726", "white", "#6D9EC1"),lab = TRUE)
par(mfrow=c(1,1))  

#VOLUME and NUM are the same, so we can remove VOLUME
#DF and MSFD are highly correlated

#Scatterplot Matrix 
require(lattice)
pairs(subdatnumcor, pch = 21)

######EDA for Qualitative Variables#####

library(ggplot2)
#TARGET_FLAG
require(ggplot2)
ggplot(train) +
  geom_bar( aes(TARGET_FLAG) ) +
  ggtitle("TARGET_FLAG") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#MSLD_bin
require(ggplot2)
ggplot(train) +
  geom_bar( aes(MSLD_bin) ) +
  ggtitle("MSLD_bin") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

#REPEAT
require(ggplot2)
ggplot(train) +
  geom_bar( aes(REPEAT) ) +
  ggtitle("REPEAT") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

###Crosstabs
attach(train)
library(gmodels)

#MSLD_bin
CrossTable(TARGET_FLAG, MSLD_bin, prop.r=FALSE, prop.c=TRUE, prop.t=TRUE, prop.chisq=FALSE)
l <- ggplot(train, aes(MSLD_bin,fill = TARGET_FLAG))
l <- l + geom_histogram(stat="count")
tapply(as.numeric(train$TARGET_FLAG) - 1 , train$ MSLD_bin,mean)

A <- ggplot(train, aes(x = MSLD_bin, fill = TARGET_FLAG)) + geom_bar(position = 'fill')+
  theme_minimal() + coord_flip()
grid.arrange(l,A, nrow = 2)

#REPEAT
CrossTable(TARGET_FLAG, REPEAT, prop.r=FALSE, prop.c=TRUE, prop.t=TRUE, prop.chisq=FALSE)
l <- ggplot(train, aes(REPEAT,fill = TARGET_FLAG))
l <- l + geom_histogram(stat="count")
tapply(as.numeric(train$TARGET_FLAG) - 1 , train$ REPEAT,mean)

A <- ggplot(train, aes(x = REPEAT, fill = TARGET_FLAG)) + geom_bar(position = 'fill')+
  theme_minimal() + coord_flip()
grid.arrange(l,A, nrow = 2)

####################################### Set up Data for Analysis ######################################

#Standardize
#Training Dataset
preObj <- preProcess(train[,-c(1,6)], method=c("center", "scale"))
x.train <- predict(preObj, train)
str(x.train)
summary(x.train)

x.train$ID <- NULL  #Remove id column
x.train$VOLUME <- NULL  #Remove VOLUME

#Validation Dataset
preObj <- preProcess(validation[,-c(1,6)], method=c("center", "scale"))
x.validation <- predict(preObj, validation)
str(x.validation)
summary(x.validation)

x.validation$ID <- NULL  #Remove id column
x.validation$VOLUME <- NULL  #Remove VOLUME

#Test Dataset
preObj <- preProcess(test[,-c(1,6)], method=c("center", "scale"))
x.test <- predict(preObj, test)
str(x.test)
summary(x.test) 

####################################### Build Models ######################################

####Logistic Regression Model###
#Full Model for Variable Selection & Baseline
model.logfull <- glm(TARGET_FLAG ~ ., x.train, family=binomial("logit"))
varImp(model.logfull)

#Stepwise Regression for Variable Selection
model.lower = glm(TARGET_FLAG ~ 1, x.train, family = binomial(link="logit"))
model.logfull <- glm(TARGET_FLAG ~ ., x.train, family=binomial("logit"))
stepAIC(model.lower, scope = list(upper=model.logfull), direction="both", test="Chisq", data=x.train)

#model.log2 <- glm(TARGET_FLAG ~ MSLD + NUM + MSFD, 
#x.train, family=binomial("logit")) #With Outlier Management

#model.log2 <- glm(TARGET_FLAG ~ MSLD + NUM + MSFD +TENRAT, 
#x.train, family=binomial("logit")) #Without Outlier Management #BEST

model.log2 <- glm(TARGET_FLAG ~ MSLD + LOG_NUM + DF, 
x.train, family=binomial("logit")) #Without Outlier Management #NEW

summary(model.log2) #As MSLD, MSFD, TENRAT decreases, more likely to donate. As NUM increases, more likely to donate. 
Anova(model.log2, type="II", test="Wald")
varImp(model.log2)
nagelkerke(model.log2)
vif(model.log2)

#Performance Metrics
AIC(model.log2) 
BIC(model.log2) 

##Performance on Validation Set
glm.pred <- predict(model.log2, x.validation, type="response") 

#Confusion Matrix
glm.pred  <- ifelse(glm.pred  > 0.5,1,0)
xtab.log1=table(glm.pred, x.validation$TARGET_FLAG)
confusionMatrix(xtab.log1, positive = "1") #0.77

#ROC Curve
library(ROCR)
prob <- predict(model.log2, newdata=x.validation, type="response")
pred <- prediction(prob, x.validation$TARGET_FLAG)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf,lwd=2, col="blue", main="ROC Curve", colorize=TRUE)
abline(a=0, b=1)

#AUC
perf_auc <- performance(pred, measure = "auc")
perf_auc <- perf_auc@y.values[[1]]
perf_auc #0.785636

#LogLoss
set.seed(1)
x.validation$P_TARGET_FLAG<- predict(model.log2, newdata = x.validation, type = "response")

actual <- x.validation[c("TARGET_FLAG")]
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)
actual$TARGET_FLAG[actual$TARGET_FLAG=="1"] <- "0"
actual$TARGET_FLAG[actual$TARGET_FLAG=="2"] <- "1"
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)

str(actual)
predicted <- x.validation[c("P_TARGET_FLAG")]
str(predicted)

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

LogLoss<-LogLoss(actual, predicted) 
(LogLoss/100)/2 #0.4608874

###LDA###
model.lda2 <- lda(TARGET_FLAG ~ MSLD + LOG_NUM + DF, 
                  x.train, family=binomial("logit"))

##Performance on Validation Set
lda.pred <- predict(model.lda2, x.validation)$posterior[,2] 

#Confusion Matrix
lda.pred  <- ifelse(lda.pred  > 0.5,1,0)
xtab.lda1=table(lda.pred, x.validation$TARGET_FLAG)
confusionMatrix(xtab.lda1, positive = "1") #0.77

#ROC Curve
library(ROCR)

test <-  predict(model.lda2, x.validation)$posterior
pred <- prediction(test[,2], x.validation$TARGET_FLAG)
perf <- performance(pred, "tpr", "fpr")
plot(perf,lwd=2, col="blue", main="ROC Curve", colorize=TRUE)
abline(a=0, b=1)

#AUC
perf_auc <- performance(pred, measure = "auc")
perf_auc <- perf_auc@y.values[[1]]
perf_auc #0.785636

#LogLoss
set.seed(1)
x.validation$P_TARGET_FLAG<- predict(model.lda2, x.validation)$posterior[,2] 

actual <- x.validation[c("TARGET_FLAG")]
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)
actual$TARGET_FLAG[actual$TARGET_FLAG=="1"] <- "0"
actual$TARGET_FLAG[actual$TARGET_FLAG=="2"] <- "1"
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)

str(actual)
predicted <- x.validation[c("P_TARGET_FLAG")]
str(predicted)

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

LogLoss(actual,predicted) 
LogLoss<-LogLoss(actual, predicted) 
(LogLoss/100)/2 #0.4601341

###QDA###
model.qda2 <- qda(TARGET_FLAG ~ MSLD + LOG_NUM + DF, 
                  x.train, family=binomial("logit"))

##Performance on Validation Set
qda.pred <- predict(model.qda2, x.validation)$posterior[,2] 

#Confusion Matrix
qda.pred  <- ifelse(qda.pred  > 0.5,1,0)
xtab.qda1=table(qda.pred, x.validation$TARGET_FLAG)
confusionMatrix(xtab.qda1, positive = "1") #0.795

#ROC Curve
library(ROCR)

test <-  predict(model.qda2, x.validation)$posterior
pred <- prediction(test[,2], x.validation$TARGET_FLAG)
perf <- performance(pred, "tpr", "fpr")
plot(perf,lwd=2, col="blue", main="ROC Curve", colorize=TRUE)
abline(a=0, b=1)

#AUC
perf_auc <- performance(pred, measure = "auc")
perf_auc <- perf_auc@y.values[[1]]
perf_auc #0.7783717

#LogLoss
set.seed(1)
x.validation$P_TARGET_FLAG<- predict(model.qda2, x.validation)$posterior[,2] 

actual <- x.validation[c("TARGET_FLAG")]
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)
actual$TARGET_FLAG[actual$TARGET_FLAG=="1"] <- "0"
actual$TARGET_FLAG[actual$TARGET_FLAG=="2"] <- "1"
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)

str(actual)

predicted <- x.validation[c("P_TARGET_FLAG")]
predicted$P_TARGET_FLAG<-round(predicted$P_TARGET_FLAG, 9)
str(predicted)

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

LogLoss<-LogLoss(actual, predicted) 
(LogLoss/100)/2 #NA

##### Decision Tree #####

#Fit Full Decision Tree
model.treefull<- tree(TARGET_FLAG ~ ., x.train)
summary(model.treefull)
model.treefull

#Plot Full Tree
plot(model.treefull)
text(model.treefull,pretty=0)

# Evaluate Full Tree on Validation Set
post.valid.treefull <- predict(model.treefull, x.validation, type="class") 

##Performance for Full Tree on Validation Set
table(post.valid.treefull, x.validation$TARGET_FLAG) # classification table
mean(post.valid.treefull== x.validation$TARGET_FLAG) #Accuracy 0.765
treeError <- mean(post.valid.treefull!= x.validation$TARGET_FLAG) #Error 0.235
treeError
xtab.treefull=table(post.valid.treefull, x.validation$TARGET_FLAG)
confusionMatrix(xtab.treefull, positive = "1")

### Use Cross-Validation to Prune Tree ###
set.seed(3)
cv.treeprune=cv.tree(model.treefull,FUN=prune.misclass)
names(cv.treeprune)
cv.treeprune

#We plot the error rate as a function of both size and k.
par(mfrow=c(1,2))
plot(cv.treeprune$size, cv.treeprune$dev,type="b")
plot(cv.treeprune$k, cv.treeprune$dev,type="b")
par(mfrow=c(1,1))

#Apply Pruned Tree
model.treeprune=prune.misclass(model.treefull,best=5)
plot(model.treeprune)
text(model.treeprune,pretty=0)

##Evaluate Pruned Tree on Validation Set
post.valid.treeprune <- predict(model.treeprune, x.validation, type="class") 

#Performance for Pruned Tree on Validation Set
table(post.valid.treeprune, x.validation$TARGET_FLAG) # classification table
mean(post.valid.treeprune== x.validation$TARGET_FLAG) #Accuracy 0.76
treeError <- mean(post.valid.treeprune!= x.validation$TARGET_FLAG) #Error 0.24
treeError
xtab.treeprune=table(post.valid.treeprune, x.validation$TARGET_FLAG)
confusionMatrix(xtab.treeprune, positive = "1")

#AUC
library(ModelMetrics)
prob <- predict(model.treeprune, newdata=x.validation, type='vector')
auc<-auc(x.validation$TARGET_FLAG,prob[,2])
auc #0.6761239

#LogLoss
set.seed(1)
x.validation$P_TARGET_FLAG<- predict(model.treeprune, newdata = x.validation, type = "vector")[,2]

actual <- x.validation[c("TARGET_FLAG")]
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)
actual$TARGET_FLAG[actual$TARGET_FLAG=="1"] <- "0"
actual$TARGET_FLAG[actual$TARGET_FLAG=="2"] <- "1"
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)

str(actual)
predicted <- x.validation[c("P_TARGET_FLAG")]
str(predicted)

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

LogLoss<-LogLoss(actual, predicted) 
(LogLoss/100)/2 #0.5129595

###Bagging###
detach(package:ModelMetrics)
library(randomForest)
set.seed(1)
bag.model=randomForest(TARGET_FLAG~.,data= x.train, mtry=17, importance=TRUE,ntree=100)
bag.model
##Performance on Validation Set
prediction_bag<- predict(bag.model, x.validation, type="class")

#Confusion Matrix
library(caret)
xtab.bag=table(prediction_bag, x.validation$TARGET_FLAG)
xtab.bag #0.77
confusionMatrix(prediction_bag,x.validation$TARGET_FLAG, positive = "1") #0.755

#AUC
library(ModelMetrics)
prob <- predict(bag.model, newdata=x.validation, type='prob')
auc<-auc(x.validation$TARGET_FLAG,prob[,2])
auc #0.6057429

#LogLoss
set.seed(1)
x.validation$P_TARGET_FLAG<- predict(bag.model, newdata = x.validation, type = "prob")[,2]

actual <- x.validation[c("TARGET_FLAG")]
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)
actual$TARGET_FLAG[actual$TARGET_FLAG=="1"] <- "0"
actual$TARGET_FLAG[actual$TARGET_FLAG=="2"] <- "1"
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)

str(actual)
predicted <- x.validation[c("P_TARGET_FLAG")]
str(predicted)

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

LogLoss<-LogLoss(actual, predicted) 
(LogLoss/100)/2 #N/A

###Random Forest Model###
detach(package:ModelMetrics)
library(caret)
set.seed(1)
control=trainControl((method="repeatedcv"),number=10,repeats=5)
model=train(TARGET_FLAG~.,data=x.train,method="rpart",preProcess="scale",trControl=control)
importance=varImp(model,scale=FALSE)
importance

library(randomForest)
set.seed(1)
rf.model=randomForest(TARGET_FLAG~.,data= x.train,importance=TRUE,ntree=400)
rf.model
importance(rf.model)
varImpPlot(rf.model)

##Performance on Validation Set
prediction_rf<- predict(rf.model, x.validation, type="class")

#Confusion Matrix
library(caret)
xtab.rf=table(prediction_rf, x.validation$TARGET_FLAG)
xtab.rf #0.78
confusionMatrix(prediction_rf,x.validation$TARGET_FLAG, positive = "1") #0.78

#AUC
library(ModelMetrics)
prob <- predict(rf.model, newdata=x.validation, type='prob')
auc<-auc(x.validation$TARGET_FLAG,prob[,2])
auc #0.683114

#LogLoss
set.seed(1)
x.validation$P_TARGET_FLAG<- predict(rf.model, newdata = x.validation, type = "prob")[,2]

actual <- x.validation[c("TARGET_FLAG")]
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)
actual$TARGET_FLAG[actual$TARGET_FLAG=="1"] <- "0"
actual$TARGET_FLAG[actual$TARGET_FLAG=="2"] <- "1"
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)

str(actual)
predicted <- x.validation[c("P_TARGET_FLAG")]
str(predicted)

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

LogLoss<-LogLoss(actual, predicted) 
(LogLoss/100)/2 #N/A

### Gradient Boosting Machines ###
detach(package:ModelMetrics)

x.train.boost= x.train
x.train.boost$TARGET_FLAG=ifelse(x.train.boost$TARGET_FLAG == "1",1,0)

library(gbm)

#Find ideal parameter using CV using automatic grid search
library(caret)
fitControl <- trainControl(method="repeatedcv", number=5, repeats=5)
set.seed(1)
gbmFit <- train(TARGET_FLAG ~ ., data = x.train.boost,
                method = "gbm", trControl = fitControl, verbose = FALSE, tuneLength=5)
gbmFit

#The final values used for the model were n.trees = 50, interaction.depth = 1, shrinkage = 0.1 and n.minobsinnode = 10.

#Fit Boost
set.seed(1)
boost.model<-gbm(TARGET_FLAG~.,data = x.train.boost,distribution="bernoulli",n.trees=50,shrinkage = 0.1,interaction.depth=1, n.minobsinnode = 10)

summary(boost.model)

#Evaluate on Validation Set
boost.pred<- predict(boost.model, x.validation, type="response", n.trees = 50) # n.valid 

#Confusion Matrix
boost.pred  <- ifelse(boost.pred  > 0.30,1,0)
xtab.boost1=table(boost.pred, x.validation$TARGET_FLAG)
confusionMatrix(xtab.boost1, positive = "1") #0.765

#AUC
library(ROCR)
library(cvAUC)
labels <- x.validation[,"TARGET_FLAG"]
AUC(predictions = boost.pred, labels = labels) #0.6387061

#LogLoss
set.seed(1)
x.validation$P_TARGET_FLAG<- predict(boost.model, newdata = x.validation, type = "response", n.trees = 50)

actual <- x.validation[c("TARGET_FLAG")]
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)
actual$TARGET_FLAG[actual$TARGET_FLAG=="1"] <- "0"
actual$TARGET_FLAG[actual$TARGET_FLAG=="2"] <- "1"
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)

str(actual)
predicted <- x.validation[c("P_TARGET_FLAG")]
str(predicted)

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

LogLoss<-LogLoss(actual, predicted) 
(LogLoss/100)/2 #0.4992688

###Neural Network###
library(caret)
#Use CV to find ideal parameters
fitControl <- trainControl(method="cv", number=5)
set.seed(1)
#nnetFit <- train(TARGET_FLAG ~MSLD + LOG_NUM + MSFD+TENRAT, x.train,
                #method = "nnet", trControl = fitControl, verbose = FALSE) #OLD

nnetFit <- train(TARGET_FLAG ~MSLD + LOG_NUM + DF, x.train,
method = "nnet", trControl = fitControl, verbose = FALSE)
nnetFit
plot(nnetFit)

#The final values used for the model were size = 1 and decay = 1e-04.

set.seed(800)
model.nnet1 <- nnet(TARGET_FLAG ~MSLD + LOG_NUM + DF, x.train, size = 1, decay=1e-04, maxit=1000)
model.nnet1 

#Performance on Validation Set
nnet.pred <- predict(model.nnet1, newdata=x.validation,type="class")

#Confusion Matrix
#table(nnet.pred,x.validation$TARGET_FLAG)
#(144+10)/200 #0.77

nn.table=table(nnet.pred,x.validation$TARGET_FLAG)
confusionMatrix(nn.table,positive = "1") #0.77

#ROC Curve
library(ROCR)

prob <- predict(model.nnet1, newdata=x.validation, type="raw")
library(ROCR)
pred = prediction(prob, x.validation$TARGET_FLAG)
perf = performance(pred, "tpr", "fpr")
plot(perf,lwd=2, col="blue", main="ROC Curve", colorize=TRUE)
abline(a=0, b=1)

#AUC
perf_auc <- performance(pred, measure = "auc")
perf_auc <- perf_auc@y.values[[1]]
perf_auc #0.7871436

#LogLoss
set.seed(1)
x.validation$P_TARGET_FLAG<- predict(model.nnet1, newdata = x.validation, type = "raw")

actual <- x.validation[c("TARGET_FLAG")]
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)
actual$TARGET_FLAG[actual$TARGET_FLAG=="1"] <- "0"
actual$TARGET_FLAG[actual$TARGET_FLAG=="2"] <- "1"
actual$TARGET_FLAG <- as.numeric(actual$TARGET_FLAG)

str(actual)
predicted <- x.validation[c("P_TARGET_FLAG")]
str(predicted)

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

LogLoss<-LogLoss(actual, predicted) 
(LogLoss/100)/2 #0.4579446

#########################################Predictions for Test Dataset & Final Submission########################################
set.seed(1)
#x.test$P_TARGET_FLAG<- predict(model.nnet1, newdata = x.test, type = "raw")#Nnet
#x.test$P_TARGET_FLAG<- predict(model.nnet1, newdata = x.test, type = "prob")[,2] #RF
#x.test$P_TARGET_FLAG<- predict(naiveBayes.model, newdata = x.test, type = "raw")[,2] #NaiveBayes
#x.test$P_TARGET_FLAG<- predict(model.lda2, x.test)$posterior[,2] #LDA
x.test$P_TARGET_FLAG<- predict(model.qda2, x.test)$posterior[,2] #QDA
#x.test$P_TARGET_FLAG<- predict(model.log2, newdata = x.test, type = "response") #Logistic

#subset of data set for the deliverable "Scored data file"
scores <- x.test[c("ID","P_TARGET_FLAG")]

#Submission format file
submission_format <- read.csv("BloodDonationSubmissionFormat.csv", check.names=FALSE)
colnames(scores) <- colnames(submission_format)
write.csv(scores, file="submission_QDA3.csv", row.names=FALSE )
