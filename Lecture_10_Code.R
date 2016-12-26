
library(RCurl) # download https data
urlfile <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.data'
x <- getURL(urlfile, ssl.verifypeer = FALSE)
gisetteRaw <- read.table(textConnection(x), sep = '', header = FALSE, stringsAsFactors = FALSE)

urlfile <- "https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.labels"
x <- getURL(urlfile, ssl.verifypeer = FALSE)
g_labels <- read.table(textConnection(x), sep = '', header = FALSE, stringsAsFactors = FALSE)
print(dim(gisetteRaw))

# build data set
gisette_df <- cbind(as.data.frame(sapply(gisetteRaw, as.numeric)), cluster=g_labels$V1)

# remove duplicate columns from entire data set
# http://stackoverflow.com/questions/9818125/identifying-duplicate-columns-in-an- r-data-frame
gisette_df <- gisette_df[!duplicated(lapply(gisette_df, summary))]

# turn outcome to classic format 0,1 instead of -1,1
gisette_df$cluster <- ifelse(gisette_df$cluster==-1,0,1)

set.seed(1234)
split <- sample(nrow(gisette_df), floor(0.5*nrow(gisette_df))) 
gisette_df_train_test <- gisette_df[split,] 
gisette_df_validate <- gisette_df[-split,]

# split gisette_df_train_test data set into training and testing
set.seed(1234)
split <- sample(nrow(gisette_df_train_test), floor(0.5*nrow(gisette_df_train_test)))
traindf <- gisette_df_train_test[split,]
testdf <- gisette_df_train_test[-split,]

# install the package if you don't already have 
# install.packages("mRMRe")
library(mRMRe)
mRMR_data <- mRMR.data(data = traindf)
print(mRMR_data)

# classic example
feats <- mRMR.classic(data = mRMR_data, target_indices = c(ncol(traindf)), feature_count = 20)
bestVars <-data.frame('features'=names(traindf)[solutions(feats)[[1]]], 'scores'= scores(feats)[[1]])
print(bestVars)

library(caret)
traindf_temp <- traindf[c(as.character(bestVars$features), 'cluster')]
# caret requires a factor of non-numeric value
traindf_temp$cluster <- ifelse(traindf_temp$cluster == 1, "yes", "no") 
traindf_temp$cluster <- as.factor(traindf_temp$cluster )
objControl <- trainControl(method='cv', number=3, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)
glmnet_model <- train(cluster~., data=traindf_temp, method="glmnet",
                      metric='roc', trControl=objControl)
glmnet_predictions <- predict(object=glmnet_model, newdata= gisette_df_validate[,as.character(bestVars$features)], type='raw')

# caret requires a factor of non-numeric value
gisette_df_validate$cluster <- ifelse(gisette_df_validate$cluster == 1, "yes", "no")
gisette_df_validate$cluster <- as.factor(gisette_df_validate$cluster ) 
print(postResample(pred=glmnet_predictions, obs=gisette_df_validate$cluster))

knn_model <- train(cluster~., data=traindf_temp, method="knn", metric='roc', trControl=objControl)
knn_predictions <- predict(object=knn_model, newdata=gisette_df_validate[,as.character(bestVars$features)], type='raw')
print(postResample(pred=knn_predictions, obs=as.factor(gisette_df_validate$cluster)))

# ensemble example
feats <- mRMR.ensemble(data = mRMR_data, target_indices = c(ncol(traindf)), solution_count = 5,feature_count = 10)
bestVars <-data.frame('features'=names(traindf)[solutions(feats)[[1]]], 'scores'= scores(feats)[[1]])
print(bestVars)

      