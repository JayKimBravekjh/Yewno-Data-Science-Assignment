## random forests test

library(sandwich, verbose = FALSE, warn.conflicts = FALSE)
library(lattice, verbose = FALSE, warn.conflicts = FALSE)
library(ggplot2, verbose = FALSE, warn.conflicts = FALSE)
library(caret, verbose = FALSE, warn.conflicts = FALSE)
library(foreach, verbose = FALSE, warn.conflicts = FALSE)
library(randomForest, verbose = FALSE, warn.conflicts = FALSE)

to.dendrogram <- function(dfrep, rownum = 1, height.increment = 0.1){
  
  if(dfrep[rownum, 'status'] == -1){
    rval <- list()
     
    attr(rval, "members") <- 1
    attr(rval, "height") <- 0.0
    attr(rval, "label") <- dfrep[rownum, 'prediction']
    attr(rval, "leaf") <- TRUE
    
  }else{
    left <- to.dendrogram(dfrep, dfrep[rownum, 'left daughter'], height.increment)
    right <- to.dendrogram(dfrep, dfrep[rownum, 'right daughter'], height.increment)
    rval <- list(left, right)
    
    attr(rval, "members") <- attr(left, "members") + attr(right, "members")
    attr(rval, "height") <- max(attr(left, "height"), attr(right, "height"))+
      height.increment
    attr(rval, "leaf") <- FALSE
    attr(rval, "edgetext") <- dfrep[rownum, 'split var']
  }
  
  class(rval) <- "dendrogram"
  
  return(rval)
}

err.rate <- function(x) {
  num <- x$confusion[1] + x$confusion[4]
  den <- x$confusion[1] + x$confusion[2] + x$confusion[3] + x$confusion[4]
  ac <- 1 - (num / den)
  ac <- ac * 100
  return(ac)
}


train <- read.csv("train.csv", head = TRUE, sep = ",")
test <- read.csv("test.csv", head = TRUE, sep = ",")

str(train)
str(test)

summary(train)
summary(test)

dftrain <- train[,c(2, 3, 5, 6, 7, 8, 10, 12)]
dftest <- test[, c(2, 3, 5, 6, 7, 9, 11)]

dftrain$Family <- dftrain$SibSp + dftrain$Parch
dftest$Family <- dftest$SibSp + dftest$Parch

dftrain$Survived <- factor(dftrain$Survived, levels = 0:1, labels = c("Not survived", "Survived"))
dftrain$Pclass <- as.factor(dftrain$Pclass)
dftest$Pclass <- as.factor(dftest$Pclass)

dftrain$Embarked <- as.character(dftrain$Embarked)

dftrain$Age <- ifelse(is.na(dftrain$Age), mean(dftrain$Age, na.rm = TRUE), dftrain$Age)
dftest$Age <- ifelse(is.na(dftest$Age), mean(dftest$Age, na.rm = TRUE), dftest$Age)

which(dftrain$Embarked == "")
dftrain$Embarked[c(62, 830)] = "S"
dftrain$Embarked <- factor(dftrain$Embarked)

dftrain$Embarked <- factor(dftrain$Embarked)

dftest$Fare <- ifelse(is.na(dftest$Fare), mean(dftest$Fare, na.rm = TRUE), dftest$Fare)

summary(dftrain)
summary(dftest)

# testing variables based on survival rate . 

f1 <- Survived ~ Fare
f2 <- Survived ~ Fare + Age
f3 <- Survived ~ Fare + Age + Pclass
f4 <- Survived ~ Fare + Age + Pclass + SibSp
f5 <- Survived ~ Fare + Age + Pclass + SibSp + Parch
f6 <- Survived ~ Fare + Age + Pclass + SibSp + Parch + Embarked




set.seed(1234) #set seed

# apply randomForests 

md1 <- randomForest(f1, data = dftrain, ntree = 300, proximity = TRUE, importance = TRUE)
md2 <- randomForest(f2, data = dftrain, ntree = 300, proximity = TRUE, importance = TRUE)
md3 <- randomForest(f3, data = dftrain, ntree = 300, proximity = TRUE, importance = TRUE)
md4 <- randomForest(f4, data = dftrain, ntree = 300, proximity = TRUE, importance = TRUE)
md5 <- randomForest(f5, data = dftrain, ntree = 300, proximity = TRUE, importance = TRUE)
md6 <- randomForest(f6, data = dftrain, ntree = 300, proximity = TRUE, importance = TRUE)


ModelsErrRate <- round(c(err.rate(md1),err.rate(md2),err.rate(md3),err.rate(md4),err.rate(md5),err.rate(md6)),3)
print(ModelsErrRate)


plot(md5, main = "Error rate over trees")
print(md5)


tree <- getTree(md5, 1, labelVar = TRUE)
d <- to.dendrogram(tree)
str(d)
plot(d,center=TRUE, leaflab = "none", edgePar = list(t.cex = 1, p.col = NA, p.lty = 0))
MDSplot(md1, dftrain$Survived, k=2, palette = 1:3)

#Importance of vars
importance(md5)
varImpPlot(md5, main = "Average Importance plots")


predictions <- predict(md5, dftest)
results <- data.frame(test$PassengerId, predictions)
