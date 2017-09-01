set.seed(2002)
xgb_learner <- makeLearner("classif.xgboost",predict.type = "response")
xgb_learner$par.vals <- list(objective = "binary:logistic", eval_metric = "error", nrounds = 150,print_every_n = 50)

xg_ps <- makeParamSet( 
  makeIntegerParam("max_depth",lower=3,upper=10),
  makeNumericParam("lambda",lower=0.05,upper=0.5),
  makeNumericParam("eta", lower = 0.01, upper = 0.5),
  makeNumericParam("subsample", lower = 0.50, upper = 1),
  makeNumericParam("min_child_weight",lower=2,upper=10),
  makeNumericParam("colsample_bytree",lower = 0.50,upper = 0.80)
)

rancontrol <- makeTuneControlRandom(maxit = 5L)

#set_cv <- makeResampleDesc("CV",iters = 4L,stratify = TRUE)
set_cv <- makeResampleDesc("CV",iters = 5L,stratify = TRUE)

#tune parameters
xgb_tune <- tuneParams(learner = xgb_learner, task = train.task, resampling = set_cv, measures = list(acc,tpr,tnr,fpr,fp,fn), par.set = xg_ps, control = rancontrol)

xgb_new <- setHyperPars(learner = xgb_learner, par.vals = xgb_tune$x)

xgmodel <- train(xgb_new, train.task)

predict.xg <- predict(xgmodel, test.task)

xg_prediction <- predict.xg$data$response

xg_confused <- confusionMatrix(d_test$income_level,xg_prediction)

precision <- xg_confused$byClass['Pos Pred Value']
recall <- xg_confused$byClass['Sensitivity']

f_measure <- 2*((precision*recall)/(precision+recall))



#filtered.data <- filterFeatures(train.task,method = "information.gain",abs = 15)
#filtered.data <- filterFeatures(train.task,method = "information.gain",abs = 20)
#filtered.data <- filterFeatures(train.task,method = "information.gain",abs = 25)

# xgb_boost <- train(xgb_new,filtered.data)

predict.xg$threshold # Threshold is 0.5

# Changing the model parameters 
# predict.type = "prob" instead of predict.type="response"


xgb_prob <- setPredictType(learner = xgb_new,predict.type = "prob")
xgmodel_prob <- train(xgb_prob,train.task)
predict.xgprob <- predict(xgmodel_prob,test.task)

#Change the Threshold to 0.4 and try for 0.25 or 0.3
# pred2 <- setThreshold(predict.xgprob,0.25)
# pred2 <- setThreshold(predict.xgprob,0.3)

pred2 <- setThreshold(predict.xgprob,0.4)
confusionMatrix(d_test$income_level,pred2$data$response)
