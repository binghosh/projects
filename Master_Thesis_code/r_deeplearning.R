library(h2o)
localH2O<- h2o.init()
train.hex<- h2o.Frame("Kopie_von_OAS_Datenauswertung_20170713")
names(train.hex)[19] <- "Slopping"
train.hex$Slopping <- h2o.asfactor(train.hex$Slopping)
splits <- h2o.splitFrame(train.hex, 0.7)
dl <- h2o.deeplearning(x=1:18, y="Slopping", training_frame=splits[[1]], validation_frame=splits[[1]], hidden=c(300,300,300), epochs = 100, nfolds = 10, fold_assignment="Stratified" )
perf <- h2o.performance(dl, splits[[2]])
h2o.confusionMatrix(perf)
