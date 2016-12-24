
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

# caret requires a factor of non-numeric value
traindf$cluster <- ifelse(traindf$cluster == 1, "yes", "no") 
traindf$cluster <- as.factor(traindf$cluster )

library(caret)
fitControl <- trainControl(method='cv', number=3, returnResamp='none', verboseIter = FALSE,
                           summaryFunction = twoClassSummary, classProbs = TRUE) 
gbm_model <- train(cluster~., data=traindf, trControl=fitControl, method="gbm",
                   metric="roc")

print(gbm_model)

testdf$cluster <- ifelse(testdf$cluster == 1, "yes", "no")
testdf$cluster <- as.factor(testdf$cluster )
predictions <- predict(object=gbm_model, testdf[,setdiff(names(testdf), 'cluste r')], type='raw')
head(predictions)


print(postResample(pred=predictions, obs=testdf$cluster))

# only plot top 50 variables
plot(varImp(gbm_model,scale=F), top = 50)

# display variable importance on a +/- scale
vimp <- varImp(gbm_model, scale=F)
results <- data.frame(row.names(vimp$importance),vimp$importance$Overall) 
results$VariableName <- rownames(vimp)
colnames(results) <- c('VariableName','Weight')
results <- results[order(results$Weight),]
# we do not want factors, just characters
results$VariableName <- as.character(results$VariableName)
# let's display the best 20 features
results_temp <- tail(results,20) 
par(mar=c(5,5,4,2)) # increase y-axis margin.
xx <- barplot(results_temp$Weight, width = 0.85,
              main = paste("Variable Importance -",'cluster'), horiz = T, xlab="<(-)importance> <neutral> <importance(+)>",
              axes=FALSE,
              col = ifelse((results_temp$Weight > 0), 'blue', 'red'))
axis(2, at=xx, labels=results_temp$VariableName, tick=FALSE, las=2, line=-0.3, cex.axis=0.6)

head(results)

tail(results)

# Let's start by modeling only top 2 variable
traindf_truncated <- traindf[, c(tail(results$VariableName,2), 'cluster')]

# caret requires a factor of non-numeric value
dim(traindf_truncated)

# we don't need parameters as we already know the best ones from previous trainCon trol call
fitControl <- trainControl(method="none")
gbm_model <- train(cluster~., data=traindf_truncated,
                   tuneGrid = expand.grid(n.trees = 150, interaction.depth = 3, shrinkage = 0.1,
                                          n.minobsinnode=10),
                   trControl=fitControl, method="gbm", metric='roc')

predictions <- predict(object=gbm_model, gisette_df_validate[,setdiff(names(traindf_truncated), 'cluster')], type='raw')
head(predictions)

# caret requires a factor of non-numeric value
gisette_df_validate$cluster <- ifelse(gisette_df_validate$cluster == 1, "yes", "n o")
gisette_df_validate$cluster <- as.factor(gisette_df_validate$cluster) 
print(postResample(pred=predictions, obs=gisette_df_validate$cluster))

head(results)

tail(results)

# Let's start by modeling with different results$Weight values
traindf_truncated <- traindf[, c(results$VariableName[results$Weight > 10], 'cluster')]
dim(traindf_truncated)

fitControl <- trainControl(method="none")
gbm_model <- train(cluster~., data=traindf_truncated,
                   tuneGrid = expand.grid(n.trees = 150, interaction.depth = 3, shrinkage = 0.1, n.minobsinnode=10),
                   trControl=fitControl, method="gbm", metric='roc')
predictions <- predict(object=gbm_model, gisette_df_validate[,setdiff(names(traindf_truncated), 'cluster')], type='raw')
print(postResample(pred=predictions, obs=as.factor(gisette_df_validate$cluster)))

traindf_truncated <- traindf[, c(results$VariableName[results$Weight > 2], 'cluster')]
dim(traindf_truncated)

fitControl <- trainControl(method="none")
gbm_model <- train(cluster~., data=traindf_truncated,
                   tuneGrid = expand.grid(n.trees = 150, interaction.depth = 3, shrinkage = 0.1, n.minobsinnode=10),
                   trControl=fitControl, method="gbm", metric='roc')
predictions <- predict(object=gbm_model, gisette_df_validate[,setdiff(names(traindf_truncated), 'cluster')], type='raw')
print(postResample(pred=predictions, obs=as.factor(gisette_df_validate$cluster)))
