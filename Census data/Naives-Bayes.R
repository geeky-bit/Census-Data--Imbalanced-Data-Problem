naive_learner <- makeLearner("classif.naiveBayes",predict.type = "response")
naive_learner$par.vals <- list(laplace = 1)

folds <- makeResampleDesc("CV",iters=10,stratify = TRUE)

 fun_cv <- function(a)
{
 crv_val <- resample(naive_learner,a,folds,measures = list(acc,tpr,tnr,fpr,fp,fn))
 crv_val$aggr
}
 
# Check for the 4-datasets that we have - Original, Under-sampling, Over-sampling, SMOTE

fun_cv(train.task)
fun_cv(train.under)
fun_cv(train.over)
fun_cv(train.smote)

nB_model <- train(naive_learner, train.smote)
nB_predict <- predict(nB_model,test.task)

nB_prediction <- nB_predict$data$response
dCM <- confusionMatrix(d_test$income_level,nB_prediction)
precision <- dCM$byClass['Pos Pred Value']
recall <- dCM$byClass['Sensitivity']

f_measure <- 2*((precision*recall)/(precision+recall))
f_measure 