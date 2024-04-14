#####################################################
#                                                   #       
#  Text Analysis for Social Scientists and Leaders  #
#                                                   #
#                  Assignment 3                     #
#                                                   #
#                                                   #
#####################################################

source("vectorFunctions.R") # a new one!

############### Word Vectors

# The real word vector files are ~ 6GB - too big for dropbox! 
# This is a smaller version,
# containing only the 50,000 most common words
vecSmall<-readRDS("data/vecSmall.RDS")

# You can download the full version from here if you like
# https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
#
# Big files need to be loaded using data.table - much faster
# library(data.table)
# vecFile<-data.table::fread("crawl-300d-2M.vec",
#                            quote="",header=F,col.names = c("word",paste0("vec",1:300)))

# remember: ALWAYS clear big files out of the workspace to reduce memory load before closing RStudio
#rm(vecSmall)
head(vecSmall)

# Word frequency file - to reweight common words
load("data/wfFile.RData")

# one column with words, and 300 with vector projections (uninterpretable!)
head(vecSmall[,1:20])

head(wfFile)

# Calculating similarity using bag of words doesn't know the difference between sad and happy!
bowSimCalc(x=c("I am very sad","I am very happy"),
           y="I am thrilled")

# However, processing the text as dense vectors allows the meaning to emerge. 
vecSimCalc(x=c("I am very sad","I am very happy"),
           y="I am thrilled",
           vecfile=vecSmall)

# word frequency weighting removes influence of most (globally) common words
vecSimCalc(x=c("I am very sad","I am very happy"),
           y="I am thrilled",
           vecfile=vecSmall,
           wffile=wfFile)

# PCAtrim removes influence of (locally) overlapping words
vecSimCalc(x=c("I am very sad","I am very happy"),
           y="I am thrilled",
           vecfile=vecSmall,
           wffile=wfFile,
           PCAtrim = 1)

###################################
# Let's get some data
###################################
set.seed(2022)

rev_med<-readRDS("data/rev_med.RDS")

train_split=sample(1:nrow(rev_med),9000)

rev_med_train<-rev_med[train_split,]
rev_med_test<-rev_med[-train_split,]


rev_med_dfm_train<-TASSL_dfm(rev_med_train$text)
rev_med_dfm_test<-TASSL_dfm(rev_med_test$text,min.prop = 0) %>%
  dfm_match(colnames(rev_med_dfm_train))

#############################################
# project data to embedding space
vdat<-vecCheck(rev_med$text,
               vecSmall,
               wfFile,
               PCAtrim=1)


vdat_train<-vdat[train_split,]
vdat_test<-vdat[-train_split,]

#############################################
# Train a vector classifier

lasso_vec<-glmnet::cv.glmnet(x=vdat_train,
                             y=rev_med_train$stars)

# notice two lines - one is at the minimum, the other is more conservative 
plot(lasso_vec)

# the default chooses the more conservative one, with fewer features
test_all_predict<-predict(lasso_vec,
                          newx = vdat_test)

kendall_acc(test_all_predict,rev_med_test$stars)

# this is how you use the minimum one - usually it produces better accuracy
test_vec_predict<-predict(lasso_vec,newx = vdat_test,
                          s="lambda.min")

kendall_acc(test_vec_predict,rev_med_test$stars)

#############################################
# vector embeddings + ngrams
combined_x_train=cbind(vdat_train,rev_med_dfm_train)
combined_x_test=cbind(vdat_test,rev_med_dfm_test)

lasso_all<-glmnet::cv.glmnet(x=combined_x_train,
                             y=rev_med_train$stars)

plot(lasso_all)

test_all_predict<-predict(lasso_all,
                          newx = combined_x_test,
                          s="lambda.min")

kendall_acc(test_all_predict,rev_med_test$stars)


#############################################
# ngrams alone
lasso_dfm<-glmnet::cv.glmnet(x=rev_med_dfm_train,
                             y=rev_med_train$stars)

plot(lasso_dfm)

test_dfm_predict<-predict(lasso_dfm,newx = rev_med_dfm_test,
                          s="lambda.min")

kendall_acc(test_dfm_predict,rev_med_test$stars)


########################################
# similarity calculation
########################################

sort(table(as.numeric(rev_med_train$funny)))

rev_med_train %>%
  filter(funny==37) %>%
  pull(text)

which.max(rev_med_train$funny)

target<-rev_med_train %>%
  slice(which.max(funny)) %>%
  pull(text)

sims<-vecSimCalc(x=rev_med_train$text,
                 y=target,
                 vecfile=vecSmall,
                 wffile = wfFile,
                 PCAtrim=1)

hist(sims)
max(sims)


rev_med_train %>%
  arrange(-sims) %>%
  slice(1:2) %>%
  pull(text)

rev_med_train$sims<-sims

######################################################################
# Distributed Dictionary
######################################################################

loughran_words<-textdata::lexicon_loughran()

# extract dictionary
positive_dict<-loughran_words %>%
  filter(sentiment=="positive") %>%
  pull(word)

# collapse into a "document"
positive_dict_doc<-positive_dict %>%
  paste(collapse=" ")

# calculate similarities to dictionary "document"
lsims<-vecSimCalc(x=rev_med_train$text,
                  y=positive_dict_doc,
                  vecfile=vecSmall,
                  wffile = wfFile,
                  PCAtrim=1)

# estimate accuracy
kendall_acc(lsims,rev_med_train$stars)

# add the similarity scores to the data.frame
rev_med_train$positive_sim<-lsims

#############################################
# extract dictionary the normal way
#############################################

# Traditional dictionary approach using dfm_lookup()
rev_med_train_dicts<-rev_med_train %>%
  pull(text) %>%
  tokens() %>%
  dfm() %>%
  dfm_lookup(as.dictionary(loughran_words)) %>%
  convert(to="data.frame")


# all the dictionaries are in there!
head(rev_med_train_dicts)

# usually you want to divide by the word count
rev_med_train_dicts<-rev_med_train_dicts %>%
  select(-doc_id) %>%
  mutate_all(~./rev_med_train$word_count)

# Accuracy score using traditional dictionary
kendall_acc(rev_med_train_dicts$positive,
            rev_med_train$stars)

########################################################
# using L&M dictionary with grammar awareness
########################################################

rev_med_train_dicts<-rev_med_train_dicts %>%
  mutate(sentiment=positive-negative)

kendall_acc(rev_med_train_dicts$sentiment,
            rev_med_train$stars)

rev_med_train<-rev_med_train %>%
  mutate(LMsentiment=sentiment_by(text,
                                  polarity_dt=lexicon::hash_sentiment_loughran_mcdonald) %>%
           pull(ave_sentiment))

kendall_acc(rev_med_train$LMsentiment,
            rev_med_train$stars)

# examples - 
c("this is a bad product","this is not a bad product") %>%
  sentiment_by(polarity_dt=lexicon::hash_sentiment_loughran_mcdonald) 

c("this is a bad product","this is not a bad product") %>%
  tokens() %>%
  dfm() %>%
  dfm_lookup(as.dictionary(loughran_words))

# ALWAYS clear big files out of the workspace to reduce memory load before closing RStudio
rm(vecSmall,wfFile)



################################################
################################################
#     an introduction to some spacy features
################################################
spacy_install()


spacyr::spacy_initialize()


# Politeness

rev_med_train_polite<-politeness(rev_med_train$text,parser="spacy")

politenessPlot(rev_med_train_polite,
               cfpb_small_test$disputed,
               middle_out = .05)


rev_tiny <- rev_med %>%
  slice(1:200)



rev_tiny_sp<-spacy_parse(rev_tiny$text,
                         nounphrase = T,
                         lemma = T,
                         dependency = T,
                         pos = T,
                         tag=T)

head(rev_tiny_sp,20)

##################################################
# Use lemmas instead of stems!
##################################################

# recreate documents from the lemmas
rev_lemma_docs<-rev_tiny_sp %>%
  group_by(doc_id) %>%
  summarize(text=paste(lemma, collapse=" ")) %>%
  mutate(doc_id=as.numeric(str_replace_all(doc_id,"text",""))) %>%
  arrange(doc_id)

#extract lemmas as words from the document
lemmas<-rev_lemma_docs$text %>%
  tokens() %>%
  tokens_select(pattern = stopwords("en"), 
                selection = "remove") %>%
  dfm() %>%
  colMeans() %>%
  sort(decreasing=TRUE) %>%
  names()

# the normal approach of stemming
stems<-TASSL_dfm(rev_lemma_docs$text) %>%
  colMeans() %>%
  sort(decreasing=TRUE) %>%
  names()

#lots of shortened non-words
stems[!stems%in%lemmas][1:100]

#this makes sense at least
lemmas[!lemmas%in%stems][1:100]

##################################################
# Negation Scoping
##################################################

# requires spacyr and sentimentr
negation_scoper<-function(text,lemmas=T){
  text_p<-spacy_parse(text,lemma =lemmas,entity = F,pos=F)
  text_p<-text_p %>%
    mutate(negation=1*(token%in%c(lexicon::hash_valence_shifters %>%
                                    filter(y==1) %>%
                                    pull(x))),
           clause_end=1*grepl("^[,.:;!?]$",token)) %>%
    group_by(sentence_id) %>%
    mutate(clause_count=cumsum(clause_end)) %>%
    group_by(doc_id,sentence_id,clause_count) %>%
    mutate(negated=cumsum(negation)-negation) %>%
    ungroup()
  if(lemmas){
    text<-text_p %>%
      mutate(lemma_neg=ifelse(negated==0,lemma,
                              paste0(lemma,"_NEG"))) %>%
      mutate(doc_id=as.numeric(gsub("text","",doc_id))) %>% 
      arrange(doc_id) %>%
      group_by(doc_id) %>%
      summarize(text=paste(lemma_neg,collapse=" ")) %>%
      pull(text)
  } else{
    text<-text_p  %>%
      mutate(token_neg=ifelse(negated==0,token,
                              paste0(token,"_NEG"))) %>%
      mutate(doc_id=as.numeric(gsub("text","",doc_id))) %>% 
      arrange(doc_id) %>%
      mutate(token_neg=ifelse(grepl("'",token),token_neg,paste0(" ",token_neg))) %>%
      group_by(doc_id) %>%
      summarize(text=paste(token_neg,collapse="")) %>%
      pull(text)
  }
  return(text)
}

negation_scoper(c("the food is not very good for the price",
                  "another test of the scoper: it's not too bad, is it?"),
                lemmas=F)


##################################################
# Using POS tags to disambiguate words
##################################################

# words with two senses
two_senses<-rev_tiny_sp %>%
  group_by(token,pos) %>%
  summarize(pos_ct=n()) %>%
  left_join(rev_tiny_sp %>%
              group_by(token) %>%
              summarize(all_ct=n())) %>%
  mutate(pos_ratio=pos_ct/all_ct) %>%
  filter(all_ct>5) %>%
  filter(pos_ratio>.2 & pos_ratio<.8) %>%
  as.data.frame()

# a few examples of words with multiple POS
two_senses

rev_sp_tagged <- rev_tiny_sp %>%
  left_join(two_senses %>%
              mutate(token_tag=paste0(token,"_",pos)) %>%
              select(token,pos,token_tag)) %>%
  mutate(tagged_tokens=ifelse(is.na(token_tag),token,token_tag))

# create a dfm from this
rev_tagged_docs<-rev_sp_tagged %>%
  group_by(doc_id) %>%
  summarize(text=paste(tagged_tokens, collapse=" ")) %>%
  mutate(doc_id=as.numeric(str_replace_all(doc_id,"text",""))) %>%
  arrange(doc_id)

TASSL_dfm(rev_tagged_docs$text) %>%
  colnames() %>%
  sort()


##################################################
# named entity recognition
##################################################

rev_ner<-spacy_extract_entity(rev_tiny$text)

rev_ner %>%
  filter(ent_type=="GPE") %>%
  with(rev(sort(table(text))))

rev_ner <- rev_ner %>%
  uncount(length) %>%
  group_by(doc_id,start_id) %>%
  mutate(doc_token_id=start_id+0:(n()-1),
         first=1*(start_id==doc_token_id)) %>%
  ungroup() %>%
  mutate(text=str_replace_all(text," ","_")) %>%
  select(doc_id,ner_text="text",first,doc_token_id) 

rev_sp_ner <- rev_tiny_sp %>%
  group_by(doc_id) %>%
  # annoying that the nounphrase counts doc tokens, not sentence tokens
  # but we do what we must
  mutate(doc_token_id=1:n()) %>%
  ungroup()%>%
  left_join(rev_ner) %>%
  filter(is.na(ner_text)|first==1) %>%
  mutate(ner_token=ifelse(is.na(ner_text),token,ner_text)) %>%
  select(-pos,-tag,-head_token_id,-first,-dep_rel,-nounphrase,-ner_text)

# generate a dfm from this

rev_ner_docs<-rev_sp_ner %>%
  group_by(doc_id) %>%
  summarize(text=paste(ner_token, collapse=" ")) %>%
  mutate(doc_id=as.numeric(str_replace_all(doc_id,"text",""))) %>%
  arrange(doc_id)

# extract all the common noun phrases
phrases<-TASSL_dfm(rev_ner_docs$text,
                 min.prop = .001) %>%
  as.data.frame() %>%
  select(contains("_"),-doc_id) %>%
  colMeans() %>%
  sort(decreasing = T) %>%
  names()

phrases[1:100]

rm(rev_sp_ner,rev_lemma_docs,
   rev_sp_tagged)

##################################################
# Extracting motifs 
##################################################

library(semgram)

extract_motifs(tokens = rev_tiny_sp,
               parse_multi_token_entities = T,
               entities = c("server","waiter","waitress"),
               add_sentence = T,
               markup = T)



extract_motifs(tokens = rev_tiny_sp,
               parse_multi_token_entities = T,
               entities = c("husband","wife","spouse","partner"),
               add_sentence = T,
               markup = T)


extract_motifs(tokens = rev_tiny_sp,
               parse_multi_token_entities = T,
               entities = c("Toronto"),
               add_sentence = T,
               markup = T)


