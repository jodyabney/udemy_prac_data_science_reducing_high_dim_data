
library(RCurl) # download https data
urlfile <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISE TTE/gisette_train.data'
x <- getURL(urlfile, ssl.verifypeer = FALSE)
gisetteRaw <- read.table(textConnection(x), sep = '', header= FALSE, stringsAsFact ors = FALSE)
urlfile <- "https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISE TTE/gisette_train.labels"
x <- getURL(urlfile, ssl.verifypeer = FALSE)
g_labels <- read.table(textConnection(x), sep = '', header = FALSE, stringsAsFacto rs = FALSE)

# build data set
gisette_df <- cbind(as.data.frame(sapply(gisetteRaw, as.numeric)), cluster=g_labels$V1)

set.seed(1234)
split <- sample(nrow(gisette_df), floor(0.5*nrow(gisette_df))) 
gisette_df_train_test <- gisette_df[split,] 
gisette_df_validate <- gisette_df[-split,]

# split gisette_df_train_test data set into training and testing
set.seed(1234)
split <- sample(nrow(gisette_df_train_test), floor(0.5*nrow(gisette_df_train_test)))
traindf <- gisette_df_train_test[split,]
testdf <- gisette_df_train_test[-split,]

library(caret)
outcome_name <- 'cluster'
predictors_names <- setdiff(names(traindf), outcome_name)

# caret requires a factor of non-numeric value 
traindf$cluster <- ifelse(traindf$cluster == 1, "yes", "no") 
traindf$cluster <- as.factor(traindf$cluster )
fitControl <- trainControl(method='cv', number=3, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)
glmnet_model <- train(x=traindf[,predictors_names], 
                      y=traindf[,outcome_name], 
                      method='glmnet', 
                      metric='roc',
                      trControl=fitControl)

print(glmnet_model)

# caret requires a factor of non-numeric value
testdf$cluster <- ifelse(testdf$cluster == 1, "yes", "no")
testdf$cluster <- as.factor(testdf$cluster )
predictions <- predict(object=glmnet_model, testdf[,setdiff(names(testdf), 'cluster')], type='raw')
head(predictions)

print(postResample(pred=predictions, obs=as.factor(testdf$cluster)))


head(varImp(glmnet_model,scale=F)$importance,100)

#  display variable importance on a +/- scale
vimp <- varImp(glmnet_model, scale=F)
results <- data.frame(row.names(vimp$importance),vimp$importance$Overall) 
results$VariableName <- rownames(vimp)
colnames(results) <- c('VariableName','Weight')
results <- results[order(results$Weight),]

# we do not want factors, just characters
results$VariableName <- as.character(results$VariableName)

par(mar=c(5,5,4,2)) # increase y-axis margin.

xx <- barplot(results$Weight, width = 0.85,
              main = paste("Variable Importance -",'cluster'), horiz = T, 
              xlab="<(-)importance> <neutral> <importance(+)>",
              axes=FALSE,
              col = ifelse((results$Weight > 0), 'blue', 'red'))
axis(2, at=xx, labels=results$VariableName, tick=FALSE, las=2, line=-0.3, cex.axis=0.6)

col = ifelse((results$Weight > 0), 'blue', 'red')
axis(2, at=xx, labels=results$VariableName, tick=FALSE, las=2, line=-0.3, cex.axis=0.6)

# display variable importance on a +/- scale
vimp <- varImp(glmnet_model, scale=F)
results <- data.frame(row.names(vimp$importance),vimp$importance$Overall) 
results$VariableName <- rownames(vimp)
colnames(results) <- c('VariableName','Weight')
results <- results[order(results$Weight),]

# remove all zero variables - non-predictive
results <- subset(results, results$Weight > 0.0001 | results$Weight < -0.0001 )

# we do not want factors, just characters
results$VariableName <- as.character(results$VariableName)

par(mar=c(5,5,4,2)) # increase y-axis margin.

xx <- barplot(results$Weight, width = 0.85,
              main = paste("Variable Importance -",'cluster'), horiz = T, 
              xlab="<(-)importance> <neutral> <importance(+)>", axes=FALSE,
              col = ifelse((results$Weight > 0), 'blue', 'red'))
axis(2, at=xx, labels=results$VariableName, tick=FALSE, las=2, line=-0.3, cex.axis=0.6)

traindf_truncated <- traindf[, c(results$VariableName, 'cluster')] 
dim(traindf_truncated)

fitControl <- trainControl(method="none")

predictors_names <- setdiff(names(traindf_truncated), 'cluster')
glmnet_model <- train(traindf_truncated[,predictors_names], 
                      traindf[,outcome_name],method='glmnet', 
                      metric='roc',
                      trControl=fitControl,tuneGrid=expand.grid(alpha=0.1, lambda=0.1))

predictions <- predict(object=glmnet_model, 
                       gisette_df_validate[,setdiff(names(traindf_truncated), 'cluster')], type='raw')

# caret requires a factor of non-numeric value
gisette_df_validate$cluster <- ifelse(gisette_df_validate$cluster == 1, "yes", "no")
gisette_df_validate$cluster <- as.factor(gisette_df_validate$cluster ) 
print(postResample(pred=predictions, obs=gisette_df_validate$cluster))
