###########################################################################################################################################
############################################### Replication of Jha et al. (2017) ###################################################
###########################################################################################################################################

setwd("C:\\Users\\ivoon\\OneDrive\\Documents\\Edu\\17-NU UCR\\S4(ii) Academic Internship")
source("Classification_Functions.R") # Loads some custom functions

libraries(c("cleanNLP","udpipe")) # Loads packages used. Specify additional needed packages using a character vector of package names

cnlp_init_udpipe(model_name = "english")

set.seed(1) # RNG seed

##### ---------------------------------------------------- Data Import ----------------------------------------------------- #####
jha=readtext(file="C:\\Users\\ivoon\\OneDrive\\Documents\\Edu\\17-NU UCR\\S4(ii) Academic Internship\\Data Files\\Citation Sentiment Jha et al..csv",
             text_field = 'Text')

jha$doc_id[apply(cbind(One=jha$Prior_Use==1,Two=jha$Next1_Use==1,Three=jha$Next2_Use==1),MARGIN = 1,FUN = any)]

jha.import=corpus(jha,
                text_field="text")
docnames(jha.import)=seq(1:nrow(jha))

##### ------------------------------------------------- CIT Preprocessing --------------------------------------------------- #####
jha.temp.1=jha.import$documents$texts

jha.temp.2=jha.temp.1 %>%
  gsub(pattern = "<(TREF)>.+?</(TREF)>",replacement = "TREF", ignore.case = T) %>%
  gsub(pattern = "<(REF)>.+?</(REF)>",replacement = "REF", ignore.case = T) %>%
  gsub(pattern = "(TREF)+(((;|,|_|) |( and )|( or ))(REF))+",replacement = "GTREF", ignore.case = T) %>%
  gsub(pattern = "((((REF(;|,|_|))|(REF and)|(REF or))) )+(GTREF|TREF)",replacement = "GTREF", ignore.case = T) %>%
  gsub(pattern = "(REF)+(((;|,|_|) |( and )|( or ))(REF))+",replacement = "GREF", ignore.case = T) %>%
  gsub(pattern = "((((REF(;|,|_|))|(REF and)|(REF or))) )+(GREF|REF)",replacement = "GREF", ignore.case = T)

prepositions=read.csv("C:\\Users\\ivoon\\OneDrive\\Documents\\Edu\\17-NU UCR\\S4(ii) Academic Internship\\Data Files\\prepositions.csv")

jha.temp.3=jha.temp.2 %>%
  gsub(pattern = "((^REF)|((?<=[,;'\".] )REF))(SYN)*(?= [[:alnum:]])",replacement = "REFSYN",perl = T) %>%
  gsub(pattern = "((^TREF)|((?<=[,;'\".] )TREF))(SYN)*(?= [[:alnum:]])",replacement = "TREFSYN",perl = T) %>%
  gsub(pattern = "((^GREF)|((?<=[,;'\".] )GREF))(SYN)*(?= [[:alnum:]])",replacement = "GREFSYN",perl = T) %>%
  gsub(pattern = "((^GTREF)|((?<=[,;'\".] )GTREF))(SYN)*(?= [[:alnum:]])",replacement = "GTREFSYN",perl = T)

jha.temp.4= jha.temp.3 %>%
  gsub(pattern=paste("(?<=",paste(prepositions$Word,collapse="|"),") (REF)",collapse = "",sep = ""), replacement = " REFSYN", perl = T) %>%
  gsub(pattern=paste("(?<=",paste(prepositions$Word,collapse="|"),") (TREF)",collapse = "",sep = ""), replacement = " TREFSYN", perl = T) %>%
  gsub(pattern=paste("(?<=",paste(prepositions$Word,collapse="|"),") (GREF)",collapse = "",sep = ""), replacement = " GREFSYN", perl = T) %>%
  gsub(pattern=paste("(?<=",paste(prepositions$Word,collapse="|"),") (GTREF)",collapse = "",sep = ""), replacement = " GTREFSYN", perl = T)

jha.temp.5=jha.temp.4 %>%
  gsub(pattern = " (REF)([^A-z]|$)",replacement = "",perl = T) %>%
  gsub(pattern = " (TREF)([^A-z]|$)",replacement = "",perl = T) %>%
  gsub(pattern = " (GREF)([^A-z]|$)",replacement = "",perl = T) %>%
  gsub(pattern = " (GTREF)([^A-z]|$)",replacement = "",perl = T)

jha.corp=corpus(jha.temp.5)
docnames(jha.corp)=docnames(jha.import)
docvars(jha.corp)=docvars(jha.import)


jha.toks=jha.corp %>% tokens(remove_numbers = T, remove_punct= T, remove_url = T) %>% tokens_select(min_nchar = 3)
jha.toks.vec=vector(length = length(jha.toks))

for (i in 1:length(jha.toks)) {
  jha.toks.vec[i]=paste(unlist(as.list(jha.toks[i])), collapse = " ")
}

##### ----------------------------------------------------- Annotation ------------------------------------------------------ #####
# Annotates using UDPipe. This can take a long time for larger datasets
cit.pos=cnlp_annotate(input = jha.toks.vec, doc_ids = jha.corp$documents$`_document`, as_strings = T, backend = "udpipe")

cit.pos.tbl=cnlp_get_token(cit.pos) # Gets the tokens, lemmas and (U)POS tags
cit.pos.tbl$tags=paste(cit.pos.tbl$lemma,cit.pos.tbl$upos,sep ="_") # Generates POS tags as token_POS

# Constructs a Quanteda compatible POS-tagged corpus, but remains in base R format for now
cit.pos.corp=data.frame(doc_id=cit.pos.tbl$id,text=as.character(cit.pos.tbl$tags),stringsAsFactors = F) 
cit.pos.corp=aggregate(cit.pos.corp$text, list(cit.pos.corp$doc_id), paste, collapse=" ")
names(cit.pos.corp)=c('doc_id','text')
cit.pos.corp[1:10,]

cit.pos.temp=as.list(cit.pos.corp %>% corpus() %>% tokens(what="fastestword"))


##### --------------------------------------------------- Reference Count --------------------------------------------------- #####
# Counts the occurences of TREFs and REFs within a sentence

jha.ref.count=jha.import$documents$texts %>%
  gsub(pattern = "<T*(REF)>.+?</T*(REF)>",replacement = "REF", ignore.case = T) %>% 
  strsplit(split = " ", fixed = t)

cit_count=numeric(length(jha.ref.count))

for(i in 1:length(jha.ref.count)){
  cit_count[i]=sum(grepl(x=jha.ref.count[[i]],pattern=".*REF.*", ignore.case = T))
}

data.frame(Count=cit_count,Polarity=docvars(jha.import,'Polarity')) %>% 
  ggplot(aes(color=factor(Polarity), fill=factor(Polarity),x=Count)) +
  geom_histogram(alpha=0.5, position="identity") +
  theme_classic() +
  scale_fill_manual(values=rep(c("blue","green","red"),5))
##### ----------------------------------------------------- Is Separate ----------------------------------------------------- #####
# Checks if distance between TREF and any REF is 1, implying neighbouring citations
cit_sep=numeric(length(jha.temp.2))

for(i in 1:length(jha.temp.2)){
  if(!any(grepl(x=jha.temp.2[[i]],pattern="GTREF", ignore.case = T))== T)
    {cit_sep[i]=1}
  else{cit_sep[i]=0}
}

cit.stats=as.dfm(data.frame(Count=cit_count,Separate=cit_sep))

##### ------------------------------------------- Closest Verb/Adjective/Adverb --------------------------------------------- #####
# Finds the nearest verb, adjective and adverb to the TREF. Contrary to Jha et al., the distance in the actual sentence is used
# rather than the shortest path in the dependency tree


cit.pos.tbl$tags[1:8]

closest_verb=character(nrow(cit.pos.corp))
closest_adjective=character(nrow(cit.pos.corp))
closest_adverb=character(nrow(cit.pos.corp))

for(i in 1:nrow(cit.pos.corp)){
  textstring = unlist(strsplit(cit.pos.corp$text[[i]], split = " "))
  
  if(!any(grep(x= textstring, pattern="_VERB"))==T){next}
  
  verbs=textstring[grep(x= textstring, pattern=".*_VERB")]
  
  if(!any(grepl(x=textstring,pattern="(TREF|GTREF)(SYN)*_", ignore.case = TRUE))==T){next}
  
  dist=abs(grep(x= textstring, pattern="_VERB") - grep(x= textstring,pattern="(TREF|GTREF)(SYN)*_", ignore.case = TRUE))
  closest_verb[i]=verbs[which(dist==min(dist))]
}
for(i in 1:nrow(cit.pos.corp)){
  textstring = unlist(strsplit(cit.pos.corp$text[[i]], split = " "))
  
  if(!any(grep(x= textstring, pattern="_ADJ"))==T){next}
  
  adjectives=textstring[grep(x= textstring, pattern="_ADJ")]
  
  if(!any(grepl(x=textstring,pattern="(TREF|GTREF)(SYN)*_", ignore.case = TRUE))==T){next}
  
  dist=abs(grep(x= textstring, pattern="_ADJ") - grep(x= textstring,pattern="(TREF|GTREF)(SYN)*_", ignore.case = TRUE))
  closest_adjective[i]=adjectives[which(dist==min(dist))]
}
for(i in 1:nrow(cit.pos.corp)){
  textstring = unlist(strsplit(cit.pos.corp$text[[i]], split = " "))
  
  if(!any(grep(x= textstring, pattern="_ADV"))==T){next}
  
  adverbs=textstring[grep(x= textstring, pattern="_ADV")]
  
  if(!any(grepl(x=textstring,pattern="(TREF|GTREF)(SYN)*_", ignore.case = TRUE))==T){next}
  
  dist=abs(grep(x= textstring, pattern="_ADV") - grep(x= textstring,pattern="(TREF|GTREF)(SYN)*_", ignore.case = TRUE))
  closest_adverb[i]=adverbs[which(dist==min(dist))]
}

closest_verb=closest_verb %>% gsub(pattern = "_VERB",replacement = "",)
closest_adjective=closest_adjective %>% gsub(pattern = "_ADJ",replacement = "",)
closest_adverb=closest_adverb %>% gsub(pattern = "_ADV",replacement = "",)

closestpos.txt=inner_join(data.frame(doc_id=cit.pos.corp$doc_id,closest_POS=paste(closest_verb,closest_adjective,closest_adverb), stringsAsFactors = F),
          data.frame(doc_id=docnames(jha.import),Polarity=docvars(jha.import,'Polarity'), stringsAsFactors = F),
          by = 'doc_id') %>% as.tbl()
names(closestpos.txt)=c("doc_id","text","Polarity")

closestpos.txt[1:10,]

closest.dfm=corpus(closestpos.txt) %>%
  tokens(what="fastestword") %>%
  dfm()

tail(closest.dfm %>% dfm_group(groups='Polarity') %>%
  textstat_keyness())

##### ----------------------------------------------- Lexicon Based Approach ------------------------------------------------ #####
# Generates dictionaries from the described keyword features
Pronouns=list(FPP=c("I","me","my","mine","myself","we","us","our","ours","ourselves"),
              TPP=c("he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","themself"))

WilsonSubjective=read.csv("C:\\Users\\ivoon\\OneDrive\\Documents\\Edu\\17-NU UCR\\S4(ii) Academic Internship\\Data Files\\OpinionFinder Subjectivity Clues.csv")

closest_subjective=character(length(jha.temp.5))

for (i in 1:length(jha.temp.5)) {
  
  textstring = unlist(strsplit(jha.temp.5[[i]], split = " "))
  
  if(any(textstring %in% WilsonSubjective$Word)==F){next}
  
  subj=textstring[which(textstring %in% WilsonSubjective$Word)]
  
  if(any(grepl(x=textstring,pattern="(TREF|GTREF)(SYN)*", ignore.case = TRUE))==F){next}
  
  dist=abs(which(textstring %in% WilsonSubjective$Word) - grep(x= textstring,pattern="(TREF|GTREF)(SYN)*", ignore.case = TRUE))
  closest_subjective[i]=subj[which(dist==min(dist))]
}

closest_subjective

closestsubj.txt=inner_join(data.frame(doc_id=docnames(jha.corp),closest_SUBJ=closest_subjective, stringsAsFactors = F),
                          data.frame(doc_id=docnames(jha.import),Polarity=docvars(jha.import,'Polarity'), stringsAsFactors = F),
                          by = 'doc_id') %>% as.tbl()
names(closestsubj.txt)=c("doc_id","text","Polarity")

closestsubj.dfm=corpus(closestsubj.txt) %>%
  tokens() %>%
  dfm(verbose = T)

closestsubj.dfm

SEMNegation.dict=list(Negation=c(".+n't","no","not","without","never","nothing","none","nor","nobody","contrary","failure","fail","except","prevent","neglected","nowhere","refused","absence","neither"))

Lexicon=dictionary(c(Pronouns,SEMNegation.dict))

jha.dfm.lex=jha.corp %>%
  tokens() %>%
  dfm() %>% 
  dfm_lookup(dictionary = Lexicon) %>% 
  dfm_weight(scheme = "boolean")

jha.dfm.lex=dfm_compress(rbind(jha.dfm.lex,closestsubj.dfm),margin="documents")
docvars(jha.dfm.lex,'Polarity')=docvars(jha.corp,'Polarity')
docvars(jha.dfm.lex,'Neutral')=(docvars(jha.dfm.lex,'Polarity')==1)

data.svm.jha=list(Sentiment=as.factor(docvars(jha.dfm.lex,'Neutral')),
                  Features=Matrix(as.matrix(jha.dfm.lex), sparse = T))

svm.parallel(data.svm.jha,c=1000)

##### ------------------------------------------------------ Section -------------------------------------------------------- #####

jha.toks %>% 
  tokens_select(pattern="[A-Z]", valuetype = "regex", case_insensitive = F)

jha.dfm.section=jha.toks %>%
  dfm()%>% 
  dfm_lookup(dictionary = dictionary(list(One=c("Introduction","Motivation"),
     Two=c("Background","Prior Work","Previous Work"),
     Three=c("Experiments","Data","Results","Evaluation"),
     Four=c("Discussion","Conclusion","Future Work")),tolower = F))
View(jha.dfm.section)

jha.toks.vec[which(jha.toks.vec %>% grepl(pattern="[:digit:][:space:][:upper:]"))]
  
##### ------------------------------------------------- Dependency Parsing -------------------------------------------------- #####
# Uses the same UDPipe annotated tokens to generate dependency triplets as described by Athar
jha.dep.tbl=cnlp_get_dependency(cit.pos, get_token = T)
jha.dep.tbl$dep=paste(jha.dep.tbl$relation,jha.dep.tbl$lemma,jha.dep.tbl$lemma_target,sep ="_")

jha.dep.corp=data.frame(doc_id=jha.dep.tbl$id,text=as.character(jha.dep.tbl$dep),stringsAsFactors = F) 
jha.dep.corp=aggregate(jha.dep.corp$text, list(jha.dep.corp$doc_id), paste, collapse=" ")
names(jha.dep.corp)=c('doc_id','text')
jha.dep.corp=corpus(jha.dep.corp)

jha.dep.dfm=jha.dep.corp %>% 
  tokens(remove_punct=T,
         remove_numbers=T, what = "fastestword") %>% 
  dfm(verbose=T)

#  tokens_remove(pattern="punct_.*", valuetype = "regex") %>%
#  tokens_remove(pattern="det_.*", valuetype = "regex") %>%
#  tokens_remove(pattern="flat_.*", valuetype = "regex") %>%
#  tokens_remove(pattern="compound_.*", valuetype = "regex") %>%
#  tokens_remove(pattern="case_.*", valuetype = "regex") %>%
#  tokens_remove(pattern="fixed_.*", valuetype = "regex") %>%
#  tokens_remove(pattern="nummod_.*", valuetype = "regex") %>%

docvars(jha.dep.dfm,"Polarity")=docvars(jha.import,"Polarity")[(docnames(jha.import) %in% docnames(jha.dep.corp))]

# Selects the top 1000 features by class
jha.dep.dfm.top=jha.dep.dfm %>% 
  textstat_frequency(groups = 'Polarity', ties_method = 'min', n = 100000, force = T) %>%
  subset(rank<=1000) %>%
  select('feature') %>% 
  unique()
jha.dep.dfm.top

jha.dep.dfm=jha.dep.dfm %>% 
  dfm_keep(pattern = jha.dep.dfm.top)

tail(jha.dep.dfm %>%
  dfm_group(groups='Polarity', force = T) %>%
  textstat_keyness(target='1', measure = 'pmi'),10)

##### -------------------------------------------------- Classification ---------------------------------------------------- #####

jha.dfm=dfm_compress(rbind(cit.stats,jha.dfm.lex,closest.dfm,jha.dep.dfm),margin = "documents")

docvars(jha.dfm,"Polarity")=left_join(data.frame(ID=docnames(jha.dfm)),data.frame(ID=docnames(jha.corp),Polarity=docvars(jha.corp,"Polarity")))[,2]
docvars(jha.dfm,"Neutral")=as.integer(docvars(jha.dfm,"Polarity")==1)

jha.dfm.n.top=jha.dfm %>%
  textstat_frequency(groups = 'Neutral', ties_method = 'min', n = 100000, force = T) %>%
  subset(rank<=1000) %>%
  select('feature') %>% 
  unique()

jha.dfm.n=jha.dfm %>% 
  dfm_keep(pattern = jha.dfm.n.top)

jha.dfm.p.top=jha.dfm %>%
  dfm_subset(subset = docvars(jha.dfm,'Neutral')!=1) %>%
  textstat_frequency(groups = 'Polarity', ties_method = 'min', n = 100000, force = T) %>%
  subset(rank<=1000) %>%
  select('feature') %>% 
  unique()

jha.dfm.p=jha.dfm %>%
  dfm_subset(subset = docvars(jha.dfm,'Neutral')!=1) %>%
  dfm_keep(pattern = jha.dfm.top)

# Subjectivity classifier
features.jha= jha.dfm.n %>% convert(to="matrix") %>% Matrix(sparse=T)

data.svm.jha=list(Sentiment=as.factor(docvars(jha.dfm,'Neutral')),
  Features=features.jha)

svm.linear.parallel(data.svm = data.svm.jha,cores = 7,k = 10,c = 1)

# Polarity Classifier
features.jha= jha.dfm.p %>% convert(to="matrix") %>% Matrix(sparse=T)

data.svm.jha.polar=list(Sentiment=as.factor(docvars(jha.dfm.p,'Polarity')),
                        Features=features.jha)

svm.linear.parallel(data.svm = data.svm.jha.polar,cores = 7,k = 10,c = 1)
