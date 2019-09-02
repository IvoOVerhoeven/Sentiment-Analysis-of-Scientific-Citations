###########################################################################################################################################
#################################### Replication of "Sentiment Analysis of Scientific Citations" ##########################################
########################################################## Awais Athar ####################################################################
##########################################################   3g+dep    ####################################################################
###########################################################################################################################################

setwd("C:\\Users\\ivoon\\OneDrive\\Documents\\Edu\\17-NU UCR\\S4(ii) Academic Internship")
source("Classification_Functions.R")

libraries("dplyr")

set.seed(1)

################################
### Data Import ################
################################
# Loads corpus from file
cit=readtext(file="C:\\Users\\ivoon\\OneDrive\\Documents\\Edu\\17-NU UCR\\S4(ii) Academic Internship\\Data Files\\Citation Sentiment Corpus v2.csv",
             text_field="Text", verbosity = 2)

cit.corp=corpus(cit[,-1],
                docid_field = 'ID',
                text_field="text",
                metacorpus = list(source = "Citation Sentiment Corpus",
                                  citation = "Athar, A. (2011). Sentiment Analysis of Citations using Sentence-Based Features. Proceedings of the ACL 2011 Student Session"))

cit.tokens.base=tokens(cit.corp,
                       remove_numbers = F, 
                       remove_punct = T,
                       remove_symbols = F, 
                       what="fastestword", 
                       verbose = T)

# Further tokenization rules. These weren't applied in Athar, but might improve classification by removing nonsensical
# snippets

#  tokens_select(pattern = "[^A-z]",selection = "remove",valuetype = "regex", verbose = T) %>% # Removes all non-alphabetic strings
#  tokens_select(pattern = "\\<*\\>*\\<*\\>",selection = "remove",valuetype = "regex", verbose = T) %>% # Removes xml-like fields
#  tokens_select(pattern = ".*\\[|.*\\[.*|.*\\]+.|.*\\]",selection = "remove",valuetype = "regex", verbose = T) %>% # Removes all bracket fields
#  tokens_select(pattern = ".*\\\\.*",selection = "remove",valuetype = "regex", verbose = T) %>% # Removes all backslashes
#  tokens_select(pattern = ".*OTH.*",selection = "remove",valuetype = "regex", verbose = T) %>% # Removes all author references
#  tokens_select(pattern = ".*null.*",selection = "remove",valuetype = "regex", verbose = T) # Removes null values


# Loads dependencies from file. These have already been generated as relation-governor-dependent triples using the 
# Standford Dependency parser as described by Athar
cit.dep=readtext(file="C:\\Users\\ivoon\\OneDrive\\Documents\\Edu\\17-NU UCR\\S4(ii) Academic Internship\\Data Files\\Citation Sentiment Corpus v2.csv",
                 text_field="Dependencies", verbosity = 3)
# cit # For quick check
cit.corp.dep=corpus(cit.dep,
                    docid_field = 'ID',
                    metacorpus = list(source = "Citation Sentiment Corpus",
                                      citation = "Athar, A. (2011). Sentiment Analysis of Citations using Sentence-Based Features. Proceedings of the ACL 2011 Student Session"
                    )) # Generates a quanteda corpus from the data

################################
### Text cleaning ##############
################################
# This emulates the functionality of the WEKA/Java code provided. The underlying methodology has not been explicitly stated
# by Athar or the creators of the WEKA package. Results are fairly similar after this trimming/cleaning has occured
tokens.3g=cit.tokens.base %>%
  tokens(ngrams = 1:3, verbose = T)

tfidf.3g=dfm(tokens.3g) %>%
  dfm_tfidf(scheme_tf = "logcount",scheme_df = "inverse")

# WEKA automatically removes the majority of the dataset by keeping the 1000 most frequent words by document class 
# (see StringToWordVector documentation). Words with similar frequency to the 1000th ranked word is also kept before
# joing the vectors. Ultimately, this removes the vast majority (>90%) of features from the DFM.
byclass.top.3g=DictionaryBuilder(dfminput = tfidf.3g,wordsToKeep = 1000)


bow3g.tfidf=dfm_select(x = tfidf.3g, pattern = byclass.top.3g, selection = "keep",valuetype = "fixed", verbose = T)
bow3g.dfm=tokens_select(x = tokens.3g, pattern = byclass.top.3g, selection = "keep",valuetype = "fixed", verbose = T) %>%
  dfm()

################################
### Dependencies  ##############
################################
# This emulates the white space tokenizer under the WEKA WhiteSpaceTokenizer command
fs1.tokens.dep=tokens(cit.corp.dep,
                      remove_numbers = F, 
                      remove_punct = F,
                      remove_symbols = F, 
                      what="fastestword",
                      ngrams = 1, 
                      verbose = T)

fs1.dfm.dep=dfm(fs1.tokens.dep) %>% 
  dfm_tfidf(scheme_tf = "logcount",scheme_df = "inverse")

fs1.byclass.top=DictionaryBuilder(dfminput = fs1.dfm.dep,wordsToKeep = 1000)

fs1.tfidf.dep.top=dfm_select(x = fs1.dfm.dep, pattern = fs1.byclass.top, selection = "keep",valuetype = "fixed", verbose = T)
fs1.dfm.dep.top=tokens_select(x = fs1.tokens.dep, pattern = fs1.byclass.top, selection = "keep",valuetype = "fixed", verbose = T) %>%
  dfm()

fs1.tfidf=rbind(bow3g.tfidf,fs1.tfidf.dep.top) %>%
  dfm_compress(margin = 'documents')
fs1.dfm=rbind(bow3g.dfm,fs1.dfm.dep.top) %>%
  dfm_compress(margin = 'documents')

# Makes certain the Sentiment label is attached to the DFM
docvars(fs1.tfidf, 'Sentiment')=docvars(cit.corp,'Sentiment')
docvars(fs1.dfm, 'Sentiment')=docvars(cit.corp,'Sentiment')



################################
### Classifiers ################
################################

rm(list=setdiff(ls(), c("fs1.dfm","fs1.data","cit.corp","k","nb.parallel","svm.parallel","libraries")))

# Runs the Naive Bayes model with parallel processing defined in the the Classification_Functions file
smos=0
for (smos in 0:5) {
  cat(smos,"\n")
  cat(nb.parallel(data.nb = fs1.dfm, cores = 7, k = 10,smoothing = smos,priordist = "docfreq",verbose = F),"\n")
}

print(nb.parallel(data.nb = fs1.dfm, cores = 7, k = 10,smoothing = smos,priordist = "docfreq",verbose = F))

nb.parallel(data.nb = fs1.dfm, cores = 7, k = 10,smoothing = 1, priordist = "docfreq", verbose = F)

# Runs a model on the whole dataset for posterior distribution
nb=textmodel_nb(x = bow3g.dfm, 
                y = docvars(bow1g.dfm)[,'Sentiment'], 
                smooth = 1, 
                prior = "docfreq")

summary(nb) # A summary of the last model ran

nb$Pc

install.packages("hexbin")
library(hexbin)

wordpolarity=data.frame(t(data.frame(nb$PcGw)))
colnames(wordpolarity)=c('n','o','p')
is.data.frame(wordpolarity)

ggtern(data=wordpolarity,aes(wordpolarity[,1],wordpolarity[,2],wordpolarity[,3])) +
  geom_point(alpha=0.2,)  +
  xlab("") + ylab("") + zlab("") +
  theme_custom(base_size = 12, base_family = "",
                                               tern.plot.background = NULL, tern.panel.background = NULL,
                                               col.T = "black", col.L = "darkred", col.R = "darkgreen",
                                               col.grid.minor = "white")+
  theme_showarrows() + 
  theme_legend_position('middleright') + 
  labs(title  = "Posterior Probability of Class Given Word",
       Tarrow = "Objective",
       Larrow = "Negative",
       Rarrow = "Positive")+
  theme_rotate(degrees = 240) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

dat=data.frame(L=c(0, 1, 2, 3, 4, 5),
Mean=c(0.660666666666667, 0.684, 0.6355, 0.586833333333333, 0.557, 0.519),
SE=c(0.00181842422626478, 0.000916515138991159, 0.0014525839046334, 0.00159269164205337, 0.0014696938456699, 0.00124899959967968))

ggplot(data=dat,aes(x=L,y=Mean)) + 
  geom_point()


names(head((nb$PcGw[1,] %>% sort(decreasing = T))),10) # Most likely negative features
(nb$PcGw[2,] %>% sort(decreasing = T))[1:10] # Most likely objective features
(nb$PcGw[3,] %>% sort(decreasing = T))[1:10] # Most likely positive features

left_join((nb$PcGw[1,] %>% sort(decreasing = T)),(nb$PcGw[3,] %>% sort(decreasing = T)))

(nb$PcGw %>% 
    sort(decreasing = T))

PCGW.tbl=t(data.frame(nb$PcGw))
which(head(PCGW.tbl[,1] %>% sort(decreasing = T)))

# Processes the SVM Data. Converts into sparce matrix (package: Matrix) for much faster computation
features.fs1=fs1.tfidf %>% convert(to="matrix")
features.fs1.sparse=features.fs1 %>% Matrix(sparse=T)

data.svm.fs1=list(Sentiment=as.factor(docvars(fs1.tfidf)[,'Sentiment']),
                  Features=features.fs1.sparse)

# Runs the Support Vector Machine with RBF kernel model with parallel processing defined in the the Classification_Functions file
for (i in 1:10) {
  cat(svm.parallel(data.svm.fs1,cores = 7,k = 10,c = 80))
}

# Parameter tuning. Only appying to the c parameter currently.
for(cos in 10^seq(1,5,by=1)){
  cat("log10(c):",log10(cos),"\n")
  cat(svm.parallel(data.svm=fs1.data,cores=7,k=10,c=cos),"\n")
}

Fold=sample(cut(seq(1,length(data.svm.fs1$Sentiment)),breaks=k,labels=FALSE))

model.e1071=e1071::svm(
  x= data.svm.fs1$Features[which(Fold != 2),], 
  y= data.svm.fs1$Sentiment[which(Fold != 2)],
  scale = T,
  type = "C-classification", 
  kernel = "radial",
  cost = 80, 
  cachesize = 400)

model.e1071.predict=predict(model.e1071,newdata = data.svm.fs1$Features[which(Fold == 2),])

confusiontable=table("Actual"=data.svm.fs1$Sentiment[which(Fold == 2)],"Predicted"=model.e1071.predict)
confusiontable
