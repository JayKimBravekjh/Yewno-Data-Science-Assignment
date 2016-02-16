library(xgboost)
library(rpart)
library(Ckmeans.1d.dp)
library(ggplot2)

set.seed(420)

## import data 

train <- read.csv("train.csv")
test <- read.csv('test.csv')

#str(train)

## combines all features in one file. 
feature <- function(train_df, test_df) { 
  test_df$Survived <- NA
  combined <- rbind(train_df, test_df)
  
  combined$Name <- as.character(combined$Name)
  
  combined$Title <- sapply(combined$Name, FUN=function(x) {strsplit(x, split= '[,.]')[[1]][2]})
  combined$Title <- sub(' ', '', combined$Title)
  
  combined$Title[combined$Title %in% c('Mme', 'Mille')] <- 'Mlle'
  combined$Title[combined$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
  combined$Title[combined$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
  combined$Title <- factor(combined$Title)
  
  combined$FamilySize <- combined$SibSp + combined$Parch + 1
  combined$Surname <- sapply(combined$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
  combined$FamilyID <- paste(as.character(combined$FamilySize), combined$Surname, sep="")
  combined$FamilyID[combined$FamilySize <= 2] <- 'Small'
  combined$FamilyID <- factor(combined$FamilyID)
  
  
  Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize, data=combined[!is.na(combined$Age),], method="anova")
  combined$Age[is.na(combined$Age)] <- predict(Agefit, combined[is.na(combined$Age),])
  
  combined$Embarked[c(62,830)] = "S"
  combined$Embarked <- factor(combined$Embarked)
  combined$Fare[1044] <- median(combined$Fare, na.rm = TRUE)
  
  combined$FamilyID2 <- combined$FamilyID
  combined$FamilyID2 <- as.character(combined$FamilyID2)
  combined$FamilyID2[combined$FamilySize <= 3] <- 'Small'
  combined$FamilyID2 <- factor(combined$FamilyID2)
  return(combined)
}

data <- feature(train, test)

## some neccessary feature combining 
combined2 <- data[, -c(1, 4, 9, 11, 15, 17)]
combined2$Pclass <- as.numeric(combined2$Pclass)-1
combined2$Sex <- as.numeric(combined2$Sex) -1 
combined2$Embarked <- as.numeric(combined2$Embarked) - 1
combined2$Title <- as.numeric(combined2$Title) - 1
combined2$FamilySize <- as.numeric(combined2$FamilySize) - 1
combined2$FamilyID <- as.numeric(combined2$FamilyID) - 1

combined2 <- as.matrix(combined2)

train <- combined2[1:300,]
test <- combined2[301:1309,]


param <- list("objective" = "binary:logistic")

cv.nround <- 15
cv.nfold <- 3

xgboost_cv = xgb.cv(param=param, data = train[, -c(1)], label = train[, c(1)], nfold = cv.nfold, nrounds = cv.nround)

nround = 15
fit_xgboost <- xgboost(param=param, data = train[, -c(1)], label = train[, c(1)], nrounds = nround)

names <- dimnames(train)[[2]]

# importance matrix using xgboost. 
importance_matrix <- xgb.importance(names, model = fit_xgboost)
xgb.plot.importance(importance_matrix)

pred_xgboost_test <- predict(fit_xgboost, test[, -c(1)])
pred_xgboost_train <- predict(fit_xgboost, train[, -c(1)])

proportion <- sapply(seq(.3, .7, .01), function(step) c(step, sum(ifelse(pred_xgboost_train<step, 0, 1)!=train[, c(1)])))
dim(proportion)

predict_xgboost_train <- ifelse(pred_xgboost_train<proportion[,which.min(proportion[2,])][1],0, 1)
head(predict_xgboost_train)
score <- sum(train[, c(1)] == predict_xgboost_train)/nrow(train)
score

predict_xgboost_test <- ifelse(pred_xgboost_test<proportion[, which.min(proportion[2,])][1],0, 1)
# test <- as.data.frame(test) 



  
