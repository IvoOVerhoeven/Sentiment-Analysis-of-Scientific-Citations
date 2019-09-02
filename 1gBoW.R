###########################################################################################################################################
#################################### Replication of "Sentiment Analysis of Scientific Citations" ##########################################
########################################################## Awais Athar ####################################################################
##########################################################   1g-BoW    ####################################################################
###########################################################################################################################################

setwd("C:\\Users\\ivoon\\OneDrive\\Documents\\Edu\\17-NU UCR\\S4(ii) Academic Internship")
source("Classification_Functions.R")

libaries()


################################
### Data Import ################
################################
# Loads corpus from file
cit=readtext(file="C:\\Users\\ivoon\\OneDrive\\Documents\\Edu\\17-NU UCR\\S4(ii) Academic Internship\\Data Files\\Citation Sentiment Corpus v2.csv",
                 text_field="Text", verbosity = 0)

cit.corp=corpus(cit, # Generates a Quanteda corpus object
                docid_field = 'ID',
                text_field="text",
                metacorpus = list(source = "Citation Sentiment Corpus",
                                  citation = "Athar, A. (2011). Sentiment Analysis of Citations using Sentence-Based Features. Proceedings of the ACL 2011 Student Session"))

tokens.1g=tokens(cit.corp, # Tokenizes the corpus
                     remove_numbers = T,
                     remove_punct = T,
                     remove_symbols = T,
                     remove_separators = T,
                     remove_twitter = T,
                     remove_hyphens = F,
                     remove_url = T,
                     ngrams = 1)

# Further tokenization rules. These weren't applied in Athar, but might improve classification by removing nonsensical
# snippets

#  tokens_select(pattern = "[^A-z]",selection = "remove",valuetype = "regex", verbose = T) %>% # Removes all non-alphabetic strings
#  tokens_select(pattern = "\\<*\\>*\\<*\\>",selection = "remove",valuetype = "regex", verbose = T) %>% # Removes xml-like fields
#  tokens_select(pattern = ".*\\[|.*\\[.*|.*\\]+.|.*\\]",selection = "remove",valuetype = "regex", verbose = T) %>% # Removes all bracket fields
#  tokens_select(pattern = ".*\\\\.*",selection = "remove",valuetype = "regex", verbose = T) %>% # Removes all backslashes
#  tokens_select(pattern = ".*OTH.*",selection = "remove",valuetype = "regex", verbose = T) %>%
#  tokens_select(pattern = ".*null.*",selection = "remove",valuetype = "regex", verbose = T)


################################
### Data cleansing #############
################################
# This emulates the functionality of the WEKA/Java code provided. The underlying methodology has not been explicitly stated
# by Athar or the creators of the WEKA package. Results are fairly similar after this trimming/cleaning has occured

cit.tfidf.1g=dfm(tokens.1g) %>%
  dfm_tfidf(scheme_tf = "logcount",scheme_df = "inverse")

# WEKA automatically removes the majority of the dataset by keeping the 1000 most frequent words by document class 
# (see StringToWordVector documentation). Words with similar frequency to the 1000th ranked word is also kept before
# joing the vectors. Ultimately, this removes the vast majority (>90%) of features from the DFM.
tfidf.1g.byclass=DictionaryBuilder(dfminput = cit.tfidf.1g,wordsToKeep = 1000)


bow1g.tfidf=dfm_select(cit.tfidf.1g, tfidf.1g.byclass, selection = "keep", verbose = T) # For SVM; takes tf-idf weights
bow1g.dfm=tokens_select(x = tokens.1g,pattern = tfidf.1g.byclass,valuetype = "fixed", selection = "keep", verbose = T) %>% dfm() # For NB; needs clean counts

docvars(bow1g.tfidf, 'Sentiment')=docvars(cit.corp,'Sentiment')


################################
### Classifiers ################
################################

rm(list=setdiff(ls(), c("bow1g.dfm","bow1g.data","cit.corp","k","nb.parallel","svm.parallel","libraries")))

# Runs the Naive Bayes model with parallel processing defined in the the Classification_Functions file
nb.parallel(data.nb = bow1g.dfm, cores = 7, k = 10,sm = 0,pr = "docfreq")

# Runs a model on the whole dataset for posterior distribution
nb=textmodel_nb(x = bow1g.dfm, 
                y = docvars(bow1g.dfm)[,'Sentiment'], 
                smooth = 0, 
                prior = "docfreq")

summary(nb) # A summary of the last model ran

(nb$PcGw[1,] %>% sort(decreasing = T))[1:10] # Most likely negative features
(nb$PcGw[2,] %>% sort(decreasing = T))[1:10] # Most likely objective features
(nb$PcGw[3,] %>% sort(decreasing = T))[1:10] # Most likely positive features

# Runs the Support Vector Machine with RBF kernel model with parallel processing defined in the the Classification_Functions file
# This converts to a format e1071 can handle
bow1g.features=bow1g.tfidf %>% convert(to="matrix") %>% Matrix(sparse=T)

bow1g.data=list(Sentiment=as.factor(docvars(bow1g.tfidf, 'Sentiment')),
                Features=bow1g.features)

svm.parallel(data.svm=bow1g.data,cores=7,k=10,c=100)

# Parameter tuning. Only appying to the c parameter currently.
for(cos in 10^seq(1,3,by=0.25)){
  cat("log10(c):",log10(cos),"\n")
  cat(svm.parallel(data.svm=bow1g.data,cores=7,k=10,c=cos),"\n")
}
