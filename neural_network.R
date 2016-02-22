train<-read.csv("train.csv")
train <- train[1:300,]
test<-read.csv("test.csv")
y <- train[,2]
library(dplyr)
library(tidyr)
library(neuralnet)

# drop two entries with no Embarked values
train<-train[train$Embarked!="",]

# set NA ages to mean for the class and sex
newage<-function(age,class,sex){if(is.na(age)){
  mean(train[train$Sex==sex &
               train$Pclass==class,]$Age,na.rm=TRUE)}
  else age}

train$newage<- mapply(newage, train$Age, train$Pclass, train$Sex)

# make name length a feature
tr1<-train %>%
  mutate(lname=sapply(as.character(Name),nchar)) %>%
  mutate(ageknown=is.na(Age)) %>%
  dplyr::select(-Ticket,-Cabin,-Name,-Age)

bins<-tr1 %>% # piped functions follow
  
  # make it narrow, don't touch numeric variables and IDs
  gather(catnames,catvalues,-PassengerId,-Survived,-newage,-Fare,-SibSp,
         -Parch,-lname,-ageknown) %>%
  
  # make single column out of them
  unite(newfactor,catnames,catvalues,sep=".") %>%
  
  # add a new column - it's "1" for every record
  mutate( is = 1) %>%
  
  # create a column from each factor, and where there's no record, add "0"
  spread(newfactor, is, fill = 0)

seed<-2
##prepare list for neuralnet() call

bins.nn<-function(df,rep=1,hidden=c(1),threshold=0.1) {
  set.seed(seed)
  nn.obj<-neuralnet(Survived ~ SibSp+ Parch+ Fare+ newage+ lname+ ageknown+ Embarked.C+ Embarked.Q+ Embarked.S+ Pclass.1+ Pclass.2+ Pclass.3+ Sex.female+ Sex.male,
                    data=df,
                    hidden=hidden,
                    lifesign="full",
                    lifesign.step=2000,
                    threshold=threshold,
                    rep=rep)
  return(nn.obj)}

# clean up results from NAs and 2s
cleanup<-function(vect){
  sapply(vect,function(x){
    if(is.na(x)) 0
    else if(x>0) 1
    else 0})}

qualify<-function(real,guess){
  check<-table(real,guess)
  good.ones<-check[1,1]+check[2,2]
  bad.ones<-check[1,2]+check[2,1]
  paste0(as.character(round(100*good.ones/(good.ones+bad.ones))),'%')
}



#n.full <- bins.nn(bins, rep=5, hidden=c(4), threshold = 0.25)

##### test data

test$newage<- mapply(newage, test$Age, test$Pclass, test$Sex)

ts1<-test %>%
  mutate(lname=sapply(as.character(Name),nchar)) %>%
  mutate(ageknown=is.na(Age)) %>%
  dplyr::select(-Ticket,-Cabin,-Name,-Age)

bins.test<-ts1 %>% # piped functions follow
  
  # make it narrow, don't touch numeric variables and IDs
  gather(catnames,catvalues,-PassengerId,-newage,-Fare,-SibSp,
         -Parch,-lname,-ageknown) %>%
  
  # make single column out of them
  unite(newfactor,catnames,catvalues,sep=".") %>%
  
  # add a new column - it's "1" for every record
  mutate( is = 1) %>%
  
  # create a column from each factor, and where there's no record, add "0"
  spread(newfactor, is, fill = 0)

nfeat.test<-dim(bins.test)[2] 


trainers<-bins

nfeat<-dim(bins)[2] 


mult<-list()
eff<-list()
tries=200
for(i in 1:tries){
  cat("Iteration #",i,"/",tries,"\n")
  set.seed(i)
  r<- 3
  h<- 5
  t<- 0.25
  nr1<-dim(trainers)[1]
  ttrainset<-sample.int(nr1,round(0.9*nr1))
  ttrainers<-trainers[ttrainset,]
  ttesters<-trainers[-ttrainset,]
  mult[[i]]<-bins.nn(ttrainers,rep=r,hidden=h,threshold=t)
  
  res<-neuralnet::compute(mult[[i]],ttesters[,3:nfeat])
  eff[[i]]<-qualify(cleanup(round(res$net.result)),
                    ttesters$Survived)
  print(eff[[i]])
}

pult<-matrix(NA, nrow=dim(bins.test)[1])
alltries<-1:tries
mineff<-90
goodtries<-alltries[unlist(eff)>mineff]
for(i  in goodtries){
  res<-neuralnet::compute(mult[[i]],bins.test[,2:nfeat.test])
  pult<-cbind(pult,cleanup(round(res$net.result)))                           
}
pult<-dplyr::select(as.data.frame(pult),-V1) # drop NA column
predi<-rowSums(pult)
cu<-mean(predi[predi!=0]) 

ppredi<-sapply(predi,function(x)if(x>cu) 1 else 0)
print(paste('training set accuracy:',mean(ppredi == (y))))

upload<-ppredi
names(upload)<-c("Survived")
upload1<-data.frame(cbind(bins.test$PassengerId,upload))
names(upload1)<-c("PassengerId","Survived")
write.csv(upload1,file="res.csv",row.names=FALSE,quote=FALSE)
