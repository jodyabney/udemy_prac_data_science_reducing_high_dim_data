
library(RCurl) # download https data
urlfile <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.data'
x <- getURL(urlfile, ssl.verifypeer = FALSE)
gisetteRaw <- read.table(textConnection(x), sep = '', header = FALSE, stringsAsFactors = FALSE)

urlfile <- "https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.labels"
x <- getURL(urlfile, ssl.verifypeer = FALSE)
g_labels <- read.table(textConnection(x), sep = '', header = FALSE, stringsAsFactors = FALSE)
print(dim(gisetteRaw))


# SMALLER DATA SET:
# truncate data set if you're having trouble running prcomp but note the scores won't be the same as in the walkthrough, a few percentage points lower:
# gisetteRaw <-gisetteRaw[1:2000,]
# g_labels <-data.frame('V1'=g_labels[1:2000,] )

library(caret)
nzv <- nearZeroVar(gisetteRaw, saveMetrics = TRUE)
print(paste('Range:',range(nzv$percentUnique)))

print(head(nzv))

print(paste('Column count before cutoff:',ncol(gisetteRaw)))

dim(nzv[nzv$percentUnique > 0.1,])

gisette_nzv <- gisetteRaw[c(rownames(nzv[nzv$percentUnique > 0.1,])) ]
print(paste('Column count after cutoff:',ncol(gisette_nzv)))

gisette_df <- cbind(as.data.frame(sapply(gisette_nzv, as.numeric)),
                    cluster=g_labels$V1)

# split data set into training and testing
set.seed(1234)
split <- sample(nrow(gisette_df), floor(0.5*nrow(gisette_df)))
traindf <- gisette_df[split,]
testdf<- gisette_df[-split,]

traindf$cluster <- as.factor(traindf$cluster )
fitControl <- trainControl(method="none")
model <- train(cluster~., data=traindf,
               tuneGrid = expand.grid(n.trees = 50, interaction.depth = 3, shrinkage = 0.1, n.minobsinnode=10),
               trControl=fitControl, method="gbm",
               metric='roc')

testdf$cluster <- as.factor(testdf$cluster ) 
predictions <- predict(object=model, testdf[,setdiff(names(testdf), 'cluster')], type='raw')

head(predictions)

print(postResample(pred=predictions, obs=testdf$cluster))

# if your machine can't handle this, try using the smaller data set supplied above (search for SMALLER DATA SET)
pmatrix <- scale(gisette_nzv)
princ <- prcomp(pmatrix)

n.comp <- 1
dfComponents <- predict(princ, newdata=pmatrix)[,1:n.comp]
gisette_df <- cbind(as.data.frame(dfComponents), cluster=g_labels$V1)

head(gisette_df)

# split data set into training and testing
set.seed(1234)
split <- sample(nrow(gisette_df), floor(0.5*nrow(gisette_df)))
traindf <- gisette_df[split,]
testdf<- gisette_df[-split,]

# force the outcome
traindf$cluster <- as.factor(traindf$cluster ) 
fitControl <- trainControl(method="none")
model <- train(cluster~., data=traindf,
               tuneGrid = expand.grid(n.trees = 50, interaction.depth = 3, shrinkage = 0.1, n.minobsinnode=10),
               trControl=fitControl, method="gbm",
               metric='roc')

testdf$cluster <- as.factor(testdf$cluster )
# note: here you need to force our single variable data set 'testdf' to a data frame, otherwise R tries to turn it into a vector
predictions <- predict(object=model, 
                       newdata=data.frame('dfComponents'=testdf[,setdiff(names(testdf),
                                                                         'cluster')]), type='raw')
print(postResample(pred=predictions, obs=testdf$cluster))

n.comp <- 2
dfComponents <- predict(princ, newdata=pmatrix)[,1:n.comp]
gisette_df <- cbind(as.data.frame(dfComponents), cluster=g_labels$V1)

# split data set into training and testing
set.seed(1234)
split <- sample(nrow(gisette_df), floor(0.5*nrow(gisette_df)))
traindf <- gisette_df[split,]
testdf<- gisette_df[-split,]

# force the outcome
traindf$cluster <- as.factor(traindf$cluster )
fitControl <- trainControl(method="none")
model <- train(cluster~., data=traindf,
               tuneGrid = expand.grid(n.trees = 50, interaction.depth = 3, shrinkage = 0.1, n.minobsinnode=10),
               trControl=fitControl, method="gbm",
               metric='roc')

testdf$cluster <- as.factor(testdf$cluster ) 
predictions <- predict(object=model, 
                       newdata=testdf[,setdiff(names(testdf), 'cluster')], type='raw')
print(postResample(pred=predictions, obs=testdf$cluster))


n.comp <- 10
dfComponents <- predict(princ, newdata=pmatrix)[,1:n.comp]
gisette_df <- cbind(as.data.frame(dfComponents), cluster=g_labels$V1)

# split data set into training and testing
set.seed(1234)
split <- sample(nrow(gisette_df), floor(0.5*nrow(gisette_df)))
traindf <- gisette_df[split,]
testdf<- gisette_df[-split,]

# force the outcome
traindf$cluster <- as.factor(traindf$cluster )
fitControl <- trainControl(method="none") 
model <- train(cluster~., data=traindf,
               tuneGrid = expand.grid(n.trees = 50, interaction.depth = 3, shrinkage = 0.1, n.minobsinnode=10),
               trControl=fitControl, method="gbm",
               metric='roc')

testdf$cluster <- as.factor(testdf$cluster ) 
predictions <- predict(object=model, newdata=testdf[,setdiff(names(testdf), 'cluster')], type='raw')
print(postResample(pred=predictions, obs=testdf$cluster))

n.comp <- 20
dfComponents <- predict(princ, newdata=pmatrix)[,1:n.comp]
gisette_df <- cbind(as.data.frame(dfComponents), cluster=g_labels$V1)

# split data set into training and testing
set.seed(1234)
split <- sample(nrow(gisette_df), floor(0.5*nrow(gisette_df))) 
traindf <- gisette_df[split,]
testdf<- gisette_df[-split,]

# force the outcome
traindf$cluster <- as.factor(traindf$cluster ) 
fitControl <- trainControl(method="none") 
model <- train(cluster~., data=traindf,
               tuneGrid = expand.grid(n.trees = 50, interaction.depth = 3, shrinkage = 0.1, n.minobsinnode=10),
               trControl=fitControl, method="gbm",
               metric='roc')

testdf$cluster <- as.factor(testdf$cluster ) 
predictions <- predict(object=model, newdata=testdf[,setdiff(names(testdf), 'cluster')], type='raw')
print(postResample(pred=predictions, obs=testdf$cluster))

