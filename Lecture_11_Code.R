library(fscaret)

data(funcRegPred)
print(funcRegPred)

data(funcClassPred)
print(funcClassPred)

titanicDF <- read.csv('http://math.ucdenver.edu/RTutorial/titanic.txt',sep='\t')
titanicDF$Title <- ifelse(grepl('Mr',titanicDF$Name),
                          'Mr',ifelse(grepl('Mrs',titanicDF$Name),
                                      'Mrs',ifelse(grepl('Miss',titanicDF$Name),
                                                   'Miss','Nothing')))
titanicDF$Age[is.na(titanicDF$Age)] <- median(titanicDF$Age, na.rm=T)

# miso format
titanicDF <- titanicDF[c('PClass', 'Age',  'Sex',  'Title', 'Survived')]

titanicDF$Title <- as.factor(titanicDF$Title)
titanicDummy <- dummyVars("~.",data=titanicDF, fullRank=F)
titanicDF <- as.data.frame(predict(titanicDummy,titanicDF))
print(names(titanicDF))

# to enable probabilities we need to force outcome to factor
titanicDF$Survived <- as.factor(ifelse(titanicDF$Survived==1, 'yes', 'no'))

set.seed(1234)
splitIndex <- sample(nrow(titanicDF), floor(0.8*nrow(titanicDF)))
traindf <- titanicDF[ splitIndex,]
testdf  <- titanicDF[-splitIndex,]

myFS.class <-fscaret(traindf, testdf, myTimeLimit = 20,
                     preprocessData=TRUE, with.labels=TRUE,
                     classPred=TRUE,
                     regPred=FALSE,
                     Used.funcClassPred=c("gbm","rpart","pls"),
                     supress.output=FALSE, no.cores=NULL,
                     saveModel=FALSE)

results <-  myFS.class$VarImp$matrixVarImp.MeasureError

results$Input_no <- as.numeric(results$Input_no)
results <- results[,setdiff(names(results), c('SUM%','ImpGrad'))]
myFS.class$PPlabels$Input_no <- as.numeric(rownames(myFS.class$PPlabels))
results <- merge(x=results, y=myFS.class$PPlabels, by="Input_no", all.x=T)
results <- results[order(-results$SUM),]
print(head(results))

traindf_truncated <- traindf[, c(head(as.character(results$Labels),5), 'Survived')]
dim(traindf_truncated)

objControl <- trainControl(method='cv', number=3, returnResamp='none',
                           summaryFunction = twoClassSummary, classProbs = TRUE)
# pls model
set.seed(1234)
pls_model <- train(Survived~., data=traindf_truncated, method="pls",
                   metric='roc', trControl=objControl)
pls_predictions <- predict(object=pls_model,
                           testdf[,setdiff(names(traindf_truncated), 'Survived')], type='prob')
library(pROC)
print(auc(predictor=pls_predictions[[2]],response=ifelse(testdf$Survived=='yes',1,0)))

method_names <- c("C5.0", "gbm", "rf")
for (method_name in method_names) {
  print(method_name)
  set.seed(1234)
  model <- train(Survived~., data=traindf_truncated,
                 method=funcClassPred[funcClassPred==method_name],
                 metric='roc', trControl=objControl)
  predictions <- predict(object=model,
                         testdf[,setdiff(names(traindf_truncated), 'Survived')], type='prob')
  print(auc(predictor=predictions[[2]],response=
              ifelse(testdf$Survived=='yes',1,0)))
}
