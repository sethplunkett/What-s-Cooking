library(jsonlite)
library(tm)
library(SnowballC)
library(randomForest)

# get test and train data from JSON files
train <- fromJSON("train.json", flatten = TRUE)
test <- fromJSON("test.json", flatten = TRUE)

# preprocess
treatment<-function(fname){
fname$ingredients <- lapply(fname$ingredients, tolower) 
fname$ingredients <- lapply(fname$ingredients, function(x) gsub("-", "_", x)) 
fname$ingredients <- lapply(fname$ingredients, function(x) gsub("[^a-z0-9_ ]", "", x))
}
treatment(train)
treatment(test)

# stemming
MyCorpus <- Corpus(VectorSource(train$ingredients))
MyCorpus2 <- Corpus(VectorSource(test$ingredients))
MyCorpus <- tm_map(MyCorpus, stemDocument) 
MyCorpus2 <- tm_map(MyCorpus2, stemDocument)

ingredientsDTM <- DocumentTermMatrix(MyCorpus) 
ingredientsDTM2 <- DocumentTermMatrix(MyCorpus2)

sparse <- removeSparseTerms(ingredientsDTM, 0.98) 
sparse2 <- removeSparseTerms(ingredientsDTM2, 0.98) 

ingredientsDTM <- as.data.frame(as.matrix(sparse)) 
ingredientsDTM2 <- as.data.frame(as.matrix(sparse2))

trainColumns<-names(ingredientsDTM) 
testColumns<-names(ingredientsDTM2)

intersect<-intersect(trainColumns,testColumns)
ingredientsDTM<- ingredientsDTM[,c(intersect)] 
ingredientsDTM2<- ingredientsDTM2[,c(intersect)] 
ingredientsDTM$cuisine <- as.factor(train$cuisine)
names(ingredientsDTM) <- gsub("-", "", names(ingredientsDTM)) 
names(ingredientsDTM2) <- gsub("-", "", names(ingredientsDTM2))


library(randomForest)
forestmodel <- randomForest(cuisine ~., data=ingredientsDTM, importance=TRUE, ntree=20) 
forestPredict<-predict(forestmodel, newdata = ingredientsDTM2, type = "class")
submission <- data.frame(id = test$id, cuisine = forestPredict)
write.csv(submission, "submission.csv", quote = FALSE, row.names = FALSE)