###########################################################################################################################################
#################################### Replication of "Sentiment Analysis of Scientific Citations" ##########################################
########################################################## Awais Athar ####################################################################
######################################################### 3g+dep + neg ############################################################
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

###################################
### Negation Tagging ##############
###################################
# Implements the window based negation described by Athar. In short: a dictionary with negation words is created. Tokens that
# match these are removed, have a "_neg" tag appended to the tokens, and are then put back into place.  This method redefines
# the text data pulled from the corpus multiple times and is thus dangerous. Run all commands to completion.

w = 15 # Window size for detecting context of negation words. 15 was found to be optimal by Athar

neg.dict=dictionary(list(neg=c('no *', 'not *', '*n\'t *', 'never  *', 'neither *', 'nor *', 'none *', 'nobody *', 'nowhere *', 'nothing *',
                               'cannot *', 'can not *', 'without *','no one *', 'no way *'))) # Defines the negation words dictionary

negation=tokens_select(cit.tokens.base, neg.dict, selection = "keep", window = c(0,w)) # Pulls the negation words + window out
neg.replace=negation %>% tokens_remove(neg.dict) %>% types() %>% as.character() %>% paste('_neg',sep="") # Tags tokens with _neg
neg.find=negation  %>% tokens_remove(neg.dict) %>% types() %>% as.character() # A list of tokens for finding in original documents

tokens.temp=as.list(tokens_select(cit.tokens.base, neg.dict, selection = "remove", padding = T, window = c(0,w))) 
# Removes negation words from the documents and replaces with padding
tokens.insert=as.list(tokens_replace(negation,neg.find,neg.replace)) 
# Generates a list with tokens that should replace padding tokens

# Uses a nested loop function to scan trhough the lists of words to replace and their desired replacements
for (i in 1:length(tokens.temp)) {
  
  index=grep("^$",tokens.temp[[i]]) # Finds the tokens that need replacing within the document (i.e. "")
  
  for (j in 1:length(index)) {
    tokens.temp[[i]][index[j]]=tokens.insert[[i]][j] # Replaces them with the element in the _neg tagged token
  }
}

cit.tokens.neg=as.tokens(tokens.temp)
cit.tokens.neg
docvars(cit.tokens.neg)=docvars(cit.tokens.base)

################################
### Text cleaning ##############
################################
# This emulates the functionality of the WEKA/Java code provided. The underlying methodology has not been explicitly stated
# by Athar or the creators of the WEKA package. Results are fairly similar after this trimming/cleaning has occured
tokens.3g.neg=cit.tokens.neg %>%
  tokens(ngrams = 1:3, verbose = T)

tfidf.3g.neg=dfm(tokens.3g) %>%
  dfm_tfidf(scheme_tf = "logcount",scheme_df = "inverse")

# WEKA automatically removes the majority of the dataset by keeping the 1000 most frequent words by document class 
# (see StringToWordVector documentation). Words with similar frequency to the 1000th ranked word is also kept before
# joing the vectors. Ultimately, this removes the vast majority (>90%) of features from the DFM.
byclass.top.3g.neg=DictionaryBuilder(dfminput = tfidf.3g.neg,wordsToKeep = 1000)


bow3g.tfidf.neg=dfm_select(x = tfidf.3g.neg, pattern = byclass.top.3g.neg, selection = "keep",valuetype = "fixed", verbose = T)
bow3g.dfm.neg=tokens_select(x = tokens.3g.neg, pattern = byclass.top.3g.neg, selection = "keep",valuetype = "fixed", verbose = T) %>%
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

fs2.tfidf=rbind(bow3g.tfidf.neg,fs1.tfidf.dep.top) %>%
  dfm_compress(margin = 'documents')

fs2.dfm=rbind(bow3g.dfm.neg,fs1.dfm.dep.top) %>%
  dfm_compress(margin = 'documents')

# Makes certain the Sentiment label is attached to the DFM
docvars(fs2.tfidf, 'Sentiment')=docvars(cit.corp,'Sentiment')
docvars(fs2.dfm, 'Sentiment')=docvars(cit.corp,'Sentiment')


################################
### Classifiers ################
################################

rm(list=setdiff(ls(), c("fs1.dfm","fs1.data","cit.corp","k","nb.parallel","svm.parallel","libraries")))

# Runs the Naive Bayes model with parallel processing defined in the the Classification_Functions file
for (i in 1:10) {
  cat(nb.parallel(data.nb = fs2.dfm, cores = 7, k = 10,smoothing = 1,priordist = "docfreq",verbose = F),"\n")
}

nb.parallel(data.nb = fs2.dfm, cores = 7, k = 10,smoothing = 1, priordist = "docfreq")

# Runs a model on the whole dataset for posterior distribution
nb=textmodel_nb(x = fs2.dfm, 
                y = docvars(fs2.dfm)[,'Sentiment'], 
                smooth = 1, 
                prior = "docfreq")

summary(nb) # A summary of the last model ran

(nb$PcGw[1,] %>% sort(decreasing = T))[1:10] # Most likely negative features
(nb$PcGw[2,] %>% sort(decreasing = T))[1:10] # Most likely objective features
(nb$PcGw[3,] %>% sort(decreasing = T))[1:10] # Most likely positive features

left_join((nb$PcGw[1,] %>% sort(decreasing = T)),(nb$PcGw[3,] %>% sort(decreasing = T)))

(nb$PcGw %>% 
    sort(decreasing = T))

PCGW.tbl=t(list(nb$PcGw))
PCGW.tbl[1:10]
which(head(PCGW.tbl[,1] %>% sort(decreasing = T)))

# Processes the SVM Data. Converts into sparce matrix (package: Matrix) for much faster computation
features.fs2=fs2.tfidf %>% convert(to="matrix") %>% Matrix(sparse=T)

data.svm.fs2=list(Sentiment=as.factor(docvars(fs2.tfidf)[,'Sentiment']),
                  Features=features.fs2)

# Runs the Support Vector Machine with RBF kernel model with parallel processing defined in the the Classification_Functions file
for(i in 1:10){
svm.parallel(data.svm.fs2,cores = 7,k = 10,c = 100)
  }

# Parameter tuning. Only appying to the c parameter currently.
for(cos in 10^seq(1,5,by=1)){
  cat("log10(c):",log10(cos),"\n")
  cat(svm.parallel(data.svm=data.svm.fs2,cores=7,k=10,c=cos),"\n")
}
