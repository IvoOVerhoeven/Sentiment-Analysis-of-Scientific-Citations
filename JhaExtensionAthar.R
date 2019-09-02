###########################################################################################################################################
################################################## Extension of Athar (2011) ######################################################
###########################################################################################################################################

setwd("C:\\Users\\ivoon\\OneDrive\\Documents\\Edu\\17-NU UCR\\S4(ii) Academic Internship")
source("Classification_Functions.R") # Loads some custom functions

libraries(c("cleanNLP","udpipe")) # Loads packages used. Specify additional needed packages using a character vector of package names

cnlp_init_udpipe(model_name = "english")

set.seed(1) # RNG seed

##### ---------------------------------------------------- Data Import ----------------------------------------------------- #####
cit=readtext(file="C:\\Users\\ivoon\\OneDrive\\Documents\\Edu\\17-NU UCR\\S4(ii) Academic Internship\\Data Files\\Citation Sentiment Corpus v2.csv",
             text_field = 'Text')

cit.import=corpus(cit,
                  text_field="text")
docnames(cit.import)=cit$ID

cit.prep.1=cit.import$documents$texts

cit.import$documents

cit.prep.2=cit.prep.1 %>%
  gsub(pattern = "<(CIT)>",replacement = "CIT", ignore.case = T) %>%
  gsub(pattern = "<(OTH)>",replacement = "OTH", ignore.case = T) %>%
  gsub(pattern = "(CIT)+(((;|,|_|) |( and )|( or ))(OTH))+",replacement = "GCIT", ignore.case = T) %>%
  gsub(pattern = "((((OTH(;|,|_|))|(OTH and)|(OTH or))) )+(GCIT|CIT)",replacement = "GCIT", ignore.case = T) %>%
  gsub(pattern = "(OTH)+(((;|,|_|) |( and )|( or ))(OTH))+",replacement = "GOTH", ignore.case = T) %>%
  gsub(pattern = "((((OTH(;|,|_|))|(OTH and)|(OTH or))) )+(GOTH|OTH)",replacement = "GOTH", ignore.case = T)

prepositions=read.csv("C:\\Users\\ivoon\\OneDrive\\Documents\\Edu\\17-NU UCR\\S4(ii) Academic Internship\\Data Files\\prepositions.csv")

cit.prep.3=cit.prep.2 %>%
  gsub(pattern = "((^OTH)|((?<=[,;'\".] )OTH))(SYN)*(?= [[:alnum:]])",replacement = "OTHSYN",perl = T) %>%
  gsub(pattern = "((^CIT)|((?<=[,;'\".] )CIT))(SYN)*(?= [[:alnum:]])",replacement = "CITSYN",perl = T) %>%
  gsub(pattern = "((^GOTH)|((?<=[,;'\".] )GOTH))(SYN)*(?= [[:alnum:]])",replacement = "GOTHSYN",perl = T) %>%
  gsub(pattern = "((^GCIT)|((?<=[,;'\".] )GCIT))(SYN)*(?= [[:alnum:]])",replacement = "GCITSYN",perl = T)

cit.prep.4 = cit.prep.3 %>%
  gsub(pattern=paste("(?<=",paste(prepositions$Word,collapse="|"),") (OTH)",collapse = "",sep = ""), replacement = " OTHSYN", perl = T) %>%
  gsub(pattern=paste("(?<=",paste(prepositions$Word,collapse="|"),") (CIT)",collapse = "",sep = ""), replacement = " CITSYN", perl = T) %>%
  gsub(pattern=paste("(?<=",paste(prepositions$Word,collapse="|"),") (GOTH)",collapse = "",sep = ""), replacement = " GOTHSYN", perl = T) %>%
  gsub(pattern=paste("(?<=",paste(prepositions$Word,collapse="|"),") (GCIT)",collapse = "",sep = ""), replacement = " GCITSYN", perl = T)

cit.prep.5 = cit.prep.4 %>%
  gsub(pattern = " (OTH)([^A-z]|$)",replacement = "",perl = T) %>%
  gsub(pattern = " (CIT)([^A-z]|$)",replacement = "",perl = T) %>%
  gsub(pattern = " (GOTH)([^A-z]|$)",replacement = "",perl = T) %>%
  gsub(pattern = " (GCIT)([^A-z]|$)",replacement = "",perl = T)

athar.corp=corpus(cit.prep.5)
docnames(athar.corp)=docnames(cit.import)
docvars(athar.corp)=docvars(cit.import)


athar.toks=athar.corp %>% tokens(remove_numbers = T, remove_punct= T, remove_url = T) %>% tokens_select(min_nchar = 3)
athar.toks.vec=vector(length = length(athar.toks))

for (i in 1:length(athar.toks)) {
  athar.toks.vec[i]=paste(unlist(as.list(athar.toks[i])), collapse = " ")
}

##### ----------------------------------------------------- Annotation ------------------------------------------------------ #####
cit.pos=cnlp_annotate(input = athar.toks.vec, doc_ids = docnames(athar.corp), as_strings = T, backend = "udpipe") # Annotates using UDPipe
cit.pos.tbl=cnlp_get_token(cit.pos) # Gets the tokens, lemmas and (U)POS tags
cit.pos.tbl$tags=paste(cit.pos.tbl$lemma,cit.pos.tbl$upos,sep ="_") # Generates POS tags as token_POS

cit.pos.tbl

# Constructs a Quanteda compatible POS-tagged corpus, but remains in base R format for now
cit.pos.corp=data.frame(doc_id=cit.pos.tbl$id,text=as.character(cit.pos.tbl$tags),stringsAsFactors = F) 
cit.pos.corp=aggregate(cit.pos.corp$text, list(cit.pos.corp$doc_id), paste, collapse=" ")
names(cit.pos.corp)=c('doc_id','text')


cit.pos.temp=as.list(cit.pos.corp %>% corpus() %>% tokens(what="fastestword"))


##### --------------------------------------------------- Reference Count --------------------------------------------------- #####
# Counts the occurences of TREFs and REFs within a sentence

athar.ref.count=cit.import$documents$texts %>%
  gsub(pattern = "<(CIT|OTH)>",replacement = "REF", ignore.case = T) %>% 
  strsplit(split = " ", fixed = t)

athar.cit_count=numeric(length(athar.ref.count))

athar.ref.count[2]

for(i in 1:length(athar.ref.count)){
  athar.cit_count[i]=sum(grepl(x=athar.ref.count[[i]],pattern=".*REF.*", ignore.case = T))
}

##### ----------------------------------------------------- Is Separate ----------------------------------------------------- #####
# Checks if distance between TREF and any REF is 1, implying neighbouring citations
athar.cit_sep=numeric(length(cit.prep.2))

for(i in 1:length(cit.prep.2)){
  if(!grepl(x=cit.prep.2[[i]],pattern="(GCIT)", ignore.case = T)== T)
  {athar.cit_sep[i]=1}
  else{athar.cit_sep[i]=0}
}

athar.cit_sep


##### ------------------------------------------- Closest Verb/Adjective/Adverb --------------------------------------------- #####
# Finds the nearest verb, adjective and adverb to the TREF. Contrary to Jha et al., the distance in the actual sentence is used
# rather than the shortest path in the dependency tree

closest_verb=character(nrow(cit.pos.corp))
closest_adjective=character(nrow(cit.pos.corp))
closest_adverb=character(nrow(cit.pos.corp))

for(i in 1:nrow(cit.pos.corp)){
  textstring = unlist(strsplit(cit.pos.corp$text[[i]], split = " "))
  
  if(!any(grep(x= textstring, pattern="_VERB"))==T){next}
  
  verbs=textstring[grep(x= textstring, pattern=".*_VERB")]
  
  if(!any(grepl(x=textstring,pattern="(CIT|GCIT)(SYN)*_", ignore.case = TRUE))==T){next}
  
  dist=abs(grep(x= textstring, pattern="_VERB") - grep(x= textstring,pattern="(CIT|GCIT)(SYN)*_", ignore.case = TRUE))
  closest_verb[i]=verbs[which(dist==min(dist))]
}
for(i in 1:nrow(cit.pos.corp)){
  textstring = unlist(strsplit(cit.pos.corp$text[[i]], split = " "))
  
  if(!any(grep(x= textstring, pattern="_ADJ"))==T){next}
  
  adjectives=textstring[grep(x= textstring, pattern="_ADJ")]
  
  if(!any(grepl(x=textstring,pattern="(CIT|GCIT)(SYN)*_", ignore.case = TRUE))==T){next}
  
  dist=abs(grep(x= textstring, pattern="_ADJ") - grep(x= textstring,pattern="(CIT|GCIT)(SYN)*_", ignore.case = TRUE))
  closest_adjective[i]=adjectives[which(dist==min(dist))]
}
for(i in 1:nrow(cit.pos.corp)){
  textstring = unlist(strsplit(cit.pos.corp$text[[i]], split = " "))
  
  if(!any(grep(x= textstring, pattern="_ADV"))==T){next}
  
  adverbs=textstring[grep(x= textstring, pattern="_ADV")]
  
  if(!any(grepl(x=textstring,pattern="(CIT|GCIT)(SYN)*_", ignore.case = TRUE))==T){next}
  
  dist=abs(grep(x= textstring, pattern="_ADV") - grep(x= textstring,pattern="(CIT|GCIT)(SYN)*_", ignore.case = TRUE))
  closest_adverb[i]=adverbs[which(dist==min(dist))]
}

closest_verb=closest_verb %>% gsub(pattern = "_VERB",replacement = "",)
closest_adjective=closest_adjective %>% gsub(pattern = "_ADJ",replacement = "",)
closest_adverb=closest_adverb %>% gsub(pattern = "_ADV",replacement = "",)

closestpos.txt=inner_join(data.frame(doc_id=cit.pos.corp$doc_id,closest_POS=paste(closest_verb,closest_adjective,closest_adverb), stringsAsFactors = F),
                          data.frame(doc_id=docnames(cit.import),Sentiment=docvars(cit.import,'Sentiment'), stringsAsFactors = F),
                          by = 'doc_id') %>% as.tbl()
names(closestpos.txt)=c("doc_id","text","Sentiment")

closestpos.txt[1:25,]

closest.dfm=corpus(closestpos.txt) %>%
  tokens(what="fastestword") %>%
  dfm()

head(closest.dfm %>% dfm_group(groups='Sentiment') %>%
       textstat_keyness(target = 'o'))


##### ----------------------------------------------- Lexicon Based Approach ------------------------------------------------ #####
# Generates dictionaries from the described keyword features
Pronouns=list(FPP=c("I","me","my","mine","myself","we","us","our","ours","ourselves"),
              TPP=c("he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","themself"))

WilsonSubjective=read.csv("C:\\Users\\ivoon\\OneDrive\\Documents\\Edu\\17-NU UCR\\S4(ii) Academic Internship\\Data Files\\OpinionFinder Subjectivity Clues.csv")
WilsonSubjective.dict=WilsonSubjective

WilsonSubjective$Word

closest_subjective=character(ndoc(athar.corp))

athar.corp

for (i in 1:ndoc(athar.corp)) {
  
  textstring = unlist(strsplit(athar.corp[i], split = " "))
  
  if(any(textstring %in% WilsonSubjective$Word)==F){next}
  
  subj=textstring[which(textstring %in% WilsonSubjective$Word)]
  
  if(any(grepl(x=textstring,pattern="(CIT|GCIT)(SYN)*", ignore.case = TRUE))==F){next}
  
  dist=abs(which(textstring %in% WilsonSubjective$Word) - grep(x= textstring,pattern="(CIT|GCIT)(SYN)*", ignore.case = TRUE))
  closest_subjective[i]=subj[which(dist==min(dist))]
}

docnames(athar.corp)

closestsubj.txt=inner_join(data.frame(doc_id=docnames(athar.corp),closest_POS=closest_subjective, stringsAsFactors = F),
                           data.frame(doc_id=docnames(cit.import),Sentiment=docvars(cit.import,'Sentiment'), stringsAsFactors = F),
                           by = 'doc_id') %>% as.tbl()
names(closestsubj.txt)=c("doc_id","text","Sentiment")

closestsubj.dfm=corpus(closestsubj.txt) %>%
  tokens() %>%
  dfm()

SEMNegation.dict=list(Negation=c("no","not","without","never","nothing","none","nor","nobody","contrary","failure","fail","except","prevent","neglected","nowhere","refused","absence","neither"))

Lexicon=dictionary(c(Pronouns,SEMNegation.dict))

cit.dfm.lex=athar.toks %>%
  dfm() %>% 
  dfm_lookup(dictionary = Lexicon)

cit.dfm.lex=dfm_compress(rbind(cit.dfm.lex,closestsubj.dfm),margin="documents")

##### -------------------------------------------------- Classification ---------------------------------------------------- #####

docnames(citstats.dfm)=docnames(jha.import)

jha.dfm=dfm_compress(rbind(citstats.dfm,jha.dfm.lex,jha.dep.dfm),margin = "documents")

docvars(jha.dfm,"Polarity")=docvars(jha.corp,"Polarity")
docvars(jha.dfm,"Neutral")=as.integer(docvars(jha.corp,"Polarity")==1)

jha.dfm.top=jha.dfm %>% 
  textstat_frequency(groups = 'Polarity', ties_method = 'random', n = 10000) %>%
  subset(!is.na('feature')) %>%
  select('feature') %>% 
  unique() %>%
  subset(feature!="na")

jha.dfm=jha.dfm %>% 
  dfm_keep(pattern = jha.dfm.top, verbose = T)

features.jha=jha.dfm %>% convert(to="matrix")
features.jha.sparse=features.jha %>% Matrix(sparse=T)

data.svm.jha=list(Sentiment=as.factor(docvars(jha.dfm,'Polarity')),
                  Features=features.jha.sparse)

svm.parallel(data.svm = data.svm.jha,cores = 7,k = 10,c = 1000)

svm.linear.parallel(data.svm=data.svm.jha)
