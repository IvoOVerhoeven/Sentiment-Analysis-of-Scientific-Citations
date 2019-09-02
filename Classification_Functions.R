k=10

libraries=function(pkgs=""){
  packages=c("readtext","quanteda","tidyverse","e1071","LiblineaR","ggplot2","ggthemes","foreach","doSNOW","Matrix")
  if(pkgs!=""){packages=append(packages,pkgs)}
  cat(sapply(packages, require, character.only = TRUE),"\n")
  cat("Package loading completed.")
  }

DictionaryBuilder=function(dfminput,wordsToKeep=1000){
  
  require(quanteda)
  require(magrittr)
  
  dictionary=dfminput %>%
    textstat_frequency(groups = 'Sentiment', ties_method = 'random', n = NULL, force = T)
  
  prune=dictionary %>%
    subset(rank==1000) %>%
    select(frequency,group)
  
  consolidated=character()
  
  for (i in 1:nrow(prune)) {
    dictionary.pruned=dictionary[dictionary$group==prune$group[[i]],]
    dictionary.pruned=dictionary.pruned[dictionary.pruned$frequency>=prune$frequency[[i]],]
    
    consolidated=append(consolidated,dictionary.pruned$feature)
  }
  
  consolidated=unique(consolidated)
  
  return(consolidated)
}

svm.parallel=function(data.svm,cores=7,k=10,c=1,g=1/ncol(data.svm$Features)){
  
  cat("Support Vector Machines for Sentiment Analysis\n")
  
  if (k<2) {
    stop("Parameter k<2: cannot cross-validate.\n\tUse regular SVM instead or set k to number of cores.")
  }
  
  cat("   I. Initializaing Function\n")
  
  time0=Sys.time()
  
  require(e1071, quietly = T)
  require(Matrix, quietly = T)
  require(foreach, quietly = T)
  require(doSNOW, quietly = T)
  
  cluster=snow::makeCluster(cores, type="SOCK")
  doSNOW::registerDoSNOW(cluster)
  
  cat("   II. Model Training\n")
  
  Fold=sample(cut(seq(1,length(data.svm$Sentiment)),breaks=k,labels=FALSE))
  
  progress=function(n) cat("\t",sprintf("Fold %d is complete.\n", n))
  
  i=numeric(1)
  
  svm.calc=foreach::foreach(i = 1:k, .options.snow = list(progress=progress)) %dopar% {
    
    library(e1071)
    library(Matrix)
    
    model.e1071=e1071::svm(
      x= data.svm$Features[which(Fold != i),], 
      y= data.svm$Sentiment[which(Fold != i)],
      scale = T,
      type = "C-classification", 
      kernel = "radial",
      cost = c, 
      gamma = g,
      cachesize = 400)
    
    model.e1071.predict=predict(model.e1071,newdata = data.svm$Features[which(Fold == i),])
    
    confusiontable=table("Actual"=data.svm$Sentiment[which(Fold == i)],"Predicted"=model.e1071.predict)
    
    precision=diag(confusiontable)/colSums(confusiontable)
    precision.tot=sum(diag(confusiontable))/sum(colSums(confusiontable))
    
    recall=diag(confusiontable)/rowSums(confusiontable)
    recall.tot=sum(diag(confusiontable))/sum(rowSums(confusiontable))
    
    f1=2*(precision*recall)/(precision+recall)
    
    f1.micro=2*(precision.tot*recall.tot)/(precision.tot+recall.tot)
    
    c(mean(f1),f1.micro,weighted.mean(f1,rowSums(confusiontable)))
    
  }
  
  cat("   III. Model Evaluation\n")
  
  metrics=data.frame(svm.calc)
  names(metrics)=1:k
  metrics=replace(metrics,is.na(metrics),0)
  cat("\t",paste(c("Macro:","Micro:","Weighted:"),format(apply(metrics,1,mean), digits = 3, droptrailing = F)," ", sep = " "),"\n", sep = "")
  time1=Sys.time()
  cat("\tDuration: ",format(difftime(time1,time0,units = "secs"), digits = 3, droptrailing = F), sep = "")
  
  snow::stopCluster(cluster)
  
}

svm.linear.parallel=function(data.svm,cores=7,k=10,c=1,g=1/ncol(data.svm$Features)){
  
  time0=Sys.time()
  
  cat("Support Vector Machines for Sentiment Analysis\n")
  
  if (k<2) {
    stop("Parameter k<2: cannot cross-validate.\n\tUse regular SVM instead or set k to number of cores.")
  }
  
  cat("   I. Initializaing Function\n")
  
  require(e1071, quietly = T)
  require(Matrix, quietly = T)
  require(foreach, quietly = T)
  require(doSNOW, quietly = T)
  
  cluster=snow::makeCluster(cores, type="SOCK")
  doSNOW::registerDoSNOW(cluster)
  
  cat("   II. Model Training\n")
  
  Fold=sample(cut(seq(1,length(data.svm$Sentiment)),breaks=k,labels=FALSE))
  
  progress=function(n) cat("\t",sprintf("Fold %d is complete.\n", n))
  
  i=numeric(1)
  
  svm.calc=foreach::foreach(i = 1:k, .options.snow = list(progress=progress)) %dopar% {
    
    library(e1071)
    library(Matrix)
    
    model.e1071=e1071::svm(
      x= data.svm$Features[which(Fold != i),], 
      y= data.svm$Sentiment[which(Fold != i)],
      scale = T,
      type = "C-classification", 
      kernel = "linear",
      cost = c,
      cachesize = 400)
    
    model.e1071.predict=predict(model.e1071,newdata = data.svm$Features[which(Fold == i),])
    
    confusiontable=table("Actual"=data.svm$Sentiment[which(Fold == i)],"Predicted"=model.e1071.predict)
    
    precision=diag(confusiontable)/colSums(confusiontable)
    precision.tot=sum(diag(confusiontable))/sum(colSums(confusiontable))
    
    recall=diag(confusiontable)/rowSums(confusiontable)
    recall.tot=sum(diag(confusiontable))/sum(rowSums(confusiontable))
    
    f1=2*(precision*recall)/(precision+recall)
    
    f1.micro=2*(precision.tot*recall.tot)/(precision.tot+recall.tot)
    
    c(precision,recall,f1,f1.micro)
    
  }
  
  cat("   III. Model Evaluation\n")
  
  metrics=data.frame(svm.calc)
  colnames(metrics)=1:k
  rownames(metrics)=c("Precision 1","Precision 2","Recall 1","Recall 2","F1 1","F1 2","F1 Micro")
  metrics=replace(metrics,is.na(metrics),0)
  print(format(metrics, digits = 3, droptrailing = F))
  time1=Sys.time()
  cat("\tDuration: ",format(difftime(time1,time0,units = "secs"), digits = 3, droptrailing = F), sep = "")
  
  snow::stopCluster(cluster)
  
}

svm.linear.parallel=function(data.svm,cores=7,k=10,c=1){
  
  cat("Support Vector Machines for Sentiment Analysis\n")
  
  time0=Sys.time()
  
  if (k<2) {
    stop("Parameter k<2: cannot cross-validate.\n\tUse regular SVM instead or set k to number of cores.")
  }
  
  cat("   I. Initializaing Function\n")
  
  require(e1071, quietly = T)
  require(Matrix, quietly = T)
  require(foreach, quietly = T)
  require(doSNOW, quietly = T)
  
  cluster=snow::makeCluster(cores, type="SOCK")
  doSNOW::registerDoSNOW(cluster)
  
  cat("   II. Model Training\n")
  
  Fold=sample(cut(seq(1,length(data.svm$Sentiment)),breaks=k,labels=FALSE))
  
  progress=function(n) cat("\t",sprintf("Fold %d is complete.\n", n))
  
  i=numeric(1)
  
  svm.calc=foreach::foreach(i = 1:k, .options.snow = list(progress=progress)) %dopar% {
    
    library(e1071)
    library(Matrix)
    
    model.e1071.1=e1071::svm(
      x= data.svm$Features[which(Fold != i),], 
      y= data.svm$Neutral[which(Fold != i)],
      scale = T,
      type = "C-classification", 
      kernel = "linear",
      cost = 1,
      cachesize = 400)
    
    test.data.1=data.svm$Features[which(Fold == i),]
    
    model.e1071.predict.1=predict(model.e1071.1,newdata = test.data.1)
    
    data.svm$Sentiment[(Fold != i & data.svm$Sentiment != 1)]
    
    model.e1071.2=e1071::svm(
      x= data.svm$Features[(Fold != i& data.svm$Sentiment != 1),],
      y= data.svm$Sentiment[(Fold != i& data.svm$Sentiment != 1)],
      scale = T,
      type = "C-classification", 
      kernel = "linear",
      cost = 1,
      cachesize = 400)
    
    test.data.2=test.data.1[model.e1071.predict.1==0,]
    
    model.e1071.predict.2=predict(model.e1071.2, newdata = test.data.2)
    
    levels(model.e1071.predict.1)=c("0","1","2","3")
    model.e1071.predict.1[which(model.e1071.predict.1==0)]=model.e1071.predict.2
    
    confusiontable=table("Actual"=data.svm$Sentiment[which(Fold == i)],"Predicted"=model.e1071.predict.1)[,2:4]
    
    precision=diag(confusiontable)/colSums(confusiontable)
    precision[!is.numeric(precision)]=0
    precision.tot=sum(diag(confusiontable))/sum(colSums(confusiontable))
    
    recall=diag(confusiontable)/rowSums(confusiontable)
    recall[!is.numeric(recall)]=0
    recall.tot=sum(diag(confusiontable))/sum(rowSums(confusiontable))
    
    f1=2*(precision*recall)/(precision+recall)
    
    f1.micro=2*(precision.tot*recall.tot)/(precision.tot+recall.tot)
    
    c(precision,recall,f1,f1.micro)
    
  }
  
  cat("   III. Model Evaluation\n")
  
  metrics=data.frame(svm.calc)
  names(metrics)=1:k
  metrics=replace(metrics,is.na(metrics),0)
  cat("\t",format(metrics, digits = 3, droptrailing = F),"\n", sep = "")
  time1=Sys.time()
  cat("\tDuration: ",format(difftime(time1,time0,units = "secs"), digits = 3, droptrailing = F), sep = "")
  
  snow::stopCluster(cluster)
  
}

nb.parallel=function(data.nb,cores=7,k=10,smoothing=0,priordist="docfreq",verbose=T){
  
  time0=Sys.time()
  
  if (verbose==T) {cat("Naive Bayes for Sentiment Analysis\n")}
  
  cluster=makeCluster(cores, type="SOCK")
  registerDoSNOW(cluster)
  
  if (verbose==T) {progress=function(n) cat("\t",sprintf("Fold %d is complete.\n", n))}
  
  Fold=sample(cut(seq(1,ndoc(data.nb)),breaks=k,labels=FALSE))
  
  if (verbose==T) {cat("\tI. Model Training\n")}
  
  nb.calc=foreach (i = 1:k) %dopar%  {
    
    library(quanteda)
    
    data.test=dfm_subset(data.nb, Fold == i)
    data.train=dfm_subset(data.nb, Fold != i) %>%
      dfm_match(features = featnames(data.test))
    
    nb=textmodel_nb(x = data.train, 
                    y = docvars(data.train,'Sentiment'), 
                    smooth = smoothing, 
                    prior = priordist)
    
    confusiontable=table(docvars(data.test)[,'Sentiment'],predict(nb,newdata=data.test))
    
    precision=diag(confusiontable)/colSums(confusiontable)
    precision[!is.numeric(precision)]=0
    precision.tot=sum(diag(confusiontable))/sum(colSums(confusiontable))
    
    recall=diag(confusiontable)/rowSums(confusiontable)
    recall[!is.numeric(recall)]=0
    recall.tot=sum(diag(confusiontable))/sum(rowSums(confusiontable))
    
    f1=2*(precision*recall)/(precision+recall)
    
    f1.micro=2*(precision.tot*recall.tot)/(precision.tot+recall.tot)
    
    c(mean(f1),f1.micro,weighted.mean(f1,rowSums(confusiontable)))
  }
  
  if (verbose==T) {cat("\tII. Model Evaluation\n")}
  
  metrics=data.frame(nb.calc)
  names(metrics)=1:k
  metrics=replace(metrics,is.na(metrics),0)
  cat("\t",paste(c("Macro","Micro","Weighted"),format(apply(metrics,1,mean), digits = 3, droptrailing = F), sep = ": "),"\n")
  time1=Sys.time()
  if (verbose==T) {cat("\t Duration: ",format(difftime(time1,time0,units = "secs"), digits = 3, droptrailing = F))}
  
  stopCluster(cluster)
  
}
