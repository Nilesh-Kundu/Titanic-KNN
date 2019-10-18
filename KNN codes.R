library(rpart)# tree models 
library(caret) # feature selection
library(rpart.plot) # plot dtree
library(ROCR) # model evaluation
library(e1071) # tuning model
library(RColorBrewer)
library(rattle)# optional, if you can't install it, it's okay
library(tree)
library(ISLR)

setwd("C:\\Users\\ADMIN\\Desktop\\R Models\\Decision Tree")
Carseats <- read.csv("Titanic.csv")
head(Carseats)
tail(Carseats)
str(Carseats)
summary(Carseats)

Carseats <- Carseats[ -c(1,4,9,11) ]
## Let's also change the labels under the "status" from (0,1) to (normal, abnormal)   
Carseats$Sex <- as.numeric(Carseats$Sex)
Carseats$Pclass <- as.numeric(Carseats$Pclass)
Carseats$SibSp <- as.numeric(Carseats$SibSp)
Carseats$Parch <- as.numeric(Carseats$Parch) 
Carseats$Embarked <- as.numeric(Carseats$Embarked)

## Check the missing value (if any)
sapply(Carseats, function(x) sum(is.na(x)))

Carseats <- na.omit(Carseats)

#Normalization
normalize <- function(x) {
return ((x - min(x)) / (max(x) - min(x))) }

Carseats.n <- as.data.frame(lapply(Carseats[,2:8], normalize))
head(Carseats.n)

#Data splitting
set.seed(123)
dat.d <- sample(1:nrow(Carseats.n),size=nrow(Carseats.n)*0.7,replace = FALSE) #random selection of 70% data.
 
train <- Carseats[dat.d,] # 70% training data
test <- Carseats[-dat.d,] # remaining 30% test data

#Creating seperate dataframe for 'Survived' feature which is our target.
train_labels <- Carseats[dat.d,1]
test_labels <-Carseats[-dat.d,1]

##Building a K-NN model
library(class)

#Find the number of observation
NROW(train_labels) 
sqrt(731)

##The square root of 700 is around 27.03, therefore we’ll create two models. 
##One with ‘K’ value as 26 and the other model with a ‘K’ value as 27.
knn.27 <- knn(train=train, test=test, cl=train_labels, k=27)
knn.28 <- knn(train=train, test=test, cl=train_labels, k=28)

#Calculate the proportion of correct classification for k = 27, 28
ACC.27 <- 100 * sum(test_labels == knn.27)/NROW(test_labels)
ACC.28 <- 100 * sum(test_labels == knn.28)/NROW(test_labels)

# Check prediction against actual value in tabular form for k=27
table(knn.27 ,test_labels)

# Check prediction against actual value in tabular form for k=28
table(knn.28 ,test_labels)

confusionMatrix(table(knn.27 ,test_labels))
confusionMatrix(table(knn.28 ,test_labels))

##Optimization
i=1                          # declaration to initiate for loop
k.optm=1                     # declaration to initiate for loop
for (i in 1:28){ 
    knn.mod <-  knn(train=train, test=test, cl=train_labels, k=i)
    k.optm[i] <- 100 * sum(test_labels == knn.mod)/NROW(test_labels)
    k=i  
    cat(k,'=',k.optm[i],'\n')       # to print % accuracy 
}

plot(k.optm, type="b", xlab="K- Value",ylab="Accuracy level") 

knn.1 <- knn(train=train, test=test, cl=train_labels, k=1) 
ACC.1 <- 100 * sum(test_labels == knn.1)/NROW(test_labels)
table(knn.1 ,test_labels)
confusionMatrix(table(knn.1 ,test_labels))