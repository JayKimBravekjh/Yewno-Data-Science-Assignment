library(randomForest)
train <- read.csv("train.csv")
test <- read.csv("test.csv")

summary(train)

train$Survived <- factor(train$Survived, levels = c(1, 0))
levels(train$Survived) <- c("Survived", "Died")
train$Pclass <- as.factor(train$Pclass)
levels(train$Pclass) <- c("1st Class", "2nd Class", "3rd Class")

png("1_survival_by_class.png", width=800, height=600)
mosaicplot(train$Pclass ~ train$Survived, main="Passenger Survival by Class", 
           color = c("#8dd3c7", "#fb8072"), shade=FALSE, xlab="", ylab="", 
           off = c(0), cex.axis=1.4)

dev.off()
