#####################################################
#                                                   #       
#  Text Analysis for Social Scientists and Leaders  #
#                                                   #
#              Assignment 3 Answers                 #
#                                                   #
#                                                   #
#####################################################

source("vectorFunctions.R")

############### Word Vectors

# The real word vector files are ~ 6GB - too big! This is a smaller version,
# containing only the 50,000 most common words
vecSmall<-readRDS("data/vecSmall.RDS")

# Word frequency file - to reweight common words
load("data/wfFile.RData")

set.seed(2022)

cfpb_small<-readRDS("data/cfpb_small.RDS") %>%
  filter(Product=="Credit reporting")

train_split=sample(1:nrow(cfpb_small),12000)

cfpb_small_train<-cfpb_small[train_split,]
cfpb_small_test<-cfpb_small[-train_split,]


#############################################
# project data to embedding space
vdat<-vecCheck(cfpb_small$narrative,
               vecSmall,
               wfFile,
               PCAtrim=1)


vdat_train<-vdat[train_split,]
vdat_test<-vdat[-train_split,]

#############################################
# Train a vector classifier

lasso_vec<-glmnet::cv.glmnet(x=vdat_train,
                             y=cfpb_small_train$disputed)

plot(lasso_vec)

test_vec_predict<-predict(lasso_vec,newx = vdat_test,
                          s="lambda.min")

vec_acc<-kendall_acc(test_vec_predict,cfpb_small_test$disputed)

#############################################
# vector embeddings + ngrams

cfpb_small_dfm_train<-TASSL_dfm(cfpb_small_train$narrative)
cfpb_small_dfm_test<-TASSL_dfm(cfpb_small_test$narrative,min.prop = 0) %>%
  dfm_match(colnames(cfpb_small_dfm_train))

combined_x_train=cbind(vdat_train,cfpb_small_dfm_train)
combined_x_test=cbind(vdat_test,cfpb_small_dfm_test)

lasso_all<-glmnet::cv.glmnet(x=combined_x_train,
                             y=cfpb_small_train$disputed)

plot(lasso_all)

test_all_predict<-predict(lasso_all,
                          newx = combined_x_test,
                          s="lambda.min")

ngram_vec_acc<-kendall_acc(test_all_predict,cfpb_small_test$disputed)


#############################################
# ngrams alone
lasso_dfm<-glmnet::cv.glmnet(x=cfpb_small_dfm_train,
                             y=cfpb_small_train$disputed)

plot(lasso_dfm)

test_dfm_predict<-predict(lasso_dfm,newx = cfpb_small_dfm_test,
                          s="lambda.min")

ngram_acc<-kendall_acc(test_dfm_predict,cfpb_small_test$disputed)

########################################
# Benchmarks
########################################

cfpb_small_test <- cfpb_small_test %>%
  mutate(wdct=str_count(narrative,"[[:alpha:]]+"),
         sentiment=narrative %>%
           sentiment_by() %>%
           pull(ave_sentiment)
  ) 

wdct_acc<-kendall_acc(cfpb_small_test$wdct,cfpb_small_test$disputed)

sentiment_acc<-kendall_acc(cfpb_small_test$sentiment,cfpb_small_test$disputed)

########################################
# Combine accuracy estimates for a plot
########################################
bind_rows(ngram_acc %>%
            mutate(features="ngrams"),
          vec_acc %>%
            mutate(features="w2v"),
          ngram_vec_acc %>%
            mutate(features="ngrams+w2v"),
          wdct_acc %>%
            mutate(features="word count"),
          sentiment_acc %>%
            mutate(features="sentiment")) %>%
  ggplot(aes(x=features,color=features,
             y=acc,ymin=lower,ymax=upper)) +
  geom_point() +
  geom_errorbar(width=.4) +
  theme_bw() +
  labs(x="Feature Set",y="Accuracy") +
  geom_hline(yintercept = 50) +
  coord_flip() +
  theme(axis.text = element_text(size=24),
        axis.title = element_text(size=24),
        panel.grid=element_blank(),
        legend.position="none")

######################################################################
# Distributed Dictionary
######################################################################

# extract dictionary as document
uncertainty_dict<-textdata::lexicon_loughran() %>%
  filter(sentiment=="uncertainty") %>%
  pull(word) %>%
  paste(collapse=" ")

# calculate similarities to dictionary "document"
lsims<-vecSimCalc(x=cfpb_small_train$narrative,
                  y=uncertainty_dict,
                  vecfile=vecSmall,
                  wffile = wfFile,
                  PCAtrim=1)


# add the similarity scores to the data.frame
cfpb_small_train$uncertain_sim<-lsims

cfpb_small_train %>%
  group_by(`Sub-issue`) %>%
  summarize(m=mean(uncertain_sim),
            se=sd(uncertain_sim)/sqrt(n())) %>%
  # reorder() re-orders the group names according to the mean values
  mutate(`Sub-issue`=reorder(`Sub-issue`,-m)) %>%
  ggplot(aes(x=`Sub-issue`,y=m,
             ymin=m-se,ymax=m+se)) +
  geom_point() +
  geom_errorbar() +
  theme_bw() +
  labs(y="Normalised Similarity with Uncertainty Dictionary") +
  coord_flip() # This line puts the long names on the left axis!

# estimate accuracy
kendall_acc(lsims,cfpb_small_train$disputed)

#############################################
# extract dictionary the normal way
#############################################

loughran_words<-textdata::lexicon_loughran()

uncertain_dict<-dictionary(list(
  loughran_uncertainty=loughran_words %>%
    filter(sentiment=="uncertainty") %>%
    pull(word)))

# Traditional dictionary approach using dfm_lookup()
cfpb_small_train_dicts<-cfpb_small_train %>%
  pull(narrative) %>%
  tokens() %>%
  dfm() %>%
  dfm_lookup(uncertain_dict) %>%
  convert(to="data.frame")

# Accuracy score using traditional dictionary
kendall_acc(cfpb_small_train_dicts$loughran_uncertainty,
            cfpb_small_train$disputed)

# ALWAYS clear big files out of the workspace to reduce memory load before closing RStudio
rm(vecSmall,wfFile)

################################################
# Politeness
################################################
cfpb_train_polite<-politeness(cfpb_small_train$narrative,parser="spacy")

politenessPlot(cfpb_train_polite,
               cfpb_small_train$disputed,
               middle_out = .05)


################################################
#     an introduction to some spacy features
################################################
spacy_install()


cfpb_tiny <- cfpb_small_train %>%
  group_by(Company) %>%
  mutate(company_count=n()) %>%
  filter(company_count>20) %>%
  slice(1:20)%>%
  ungroup()
  

spacyr::spacy_initialize()


cfpb_tiny_sp<-spacy_parse(cfpb_tiny$narrative,
                         nounphrase = T,
                         lemma = T,
                         dependency = T,
                         pos = T,
                         tag=T)

head(cfpb_tiny_sp,20)

##################################################
# Use lemmas instead of stems!
##################################################

# recreate documents from the lemmas
cfpb_lemma_docs<-cfpb_tiny_sp %>%
  group_by(doc_id) %>%
  summarize(text=paste(lemma, collapse=" ")) %>%
  mutate(doc_id=as.numeric(str_replace_all(doc_id,"text",""))) %>%
  arrange(doc_id)

#extract lemmas as words from the document
lemmas<-cfpb_lemma_docs$narrative %>%
  tokens() %>%
  tokens_select(pattern = stopwords("en"), 
                selection = "remove") %>%
  dfm() %>%
  colMeans() %>%
  sort(decreasing=TRUE) %>%
  names()

# the normal approach of stemming
stems<-TASSL_dfm(cfpb_lemma_docs$narrative) %>%
  colMeans() %>%
  sort(decreasing=TRUE) %>%
  names()

#lots of shortened non-words
stems[!stems%in%lemmas][1:100]

#this makes sense at least
lemmas[!lemmas%in%stems][1:100]


##################################################
# named entity recognition
##################################################

cfpb_ner<-spacy_extract_entity(cfpb_tiny$narrative)

cfpb_ner <- cfpb_ner %>%
  uncount(length) %>%
  group_by(doc_id,start_id) %>%
  mutate(doc_token_id=start_id+0:(n()-1),
         first=1*(start_id==doc_token_id)) %>%
  ungroup() %>%
  mutate(text=str_replace_all(text," ","_")) %>%
  select(doc_id,ner_text="text",first,doc_token_id) 

cfpb_sp_ner <- cfpb_tiny_sp %>%
  group_by(doc_id) %>%
  # annoying that the nounphrase counts doc tokens, not sentence tokens
  # but we do what we must
  mutate(doc_token_id=1:n()) %>%
  ungroup()%>%
  left_join(cfpb_ner) %>%
  filter(is.na(ner_text)|first==1) %>%
  mutate(ner_token=ifelse(is.na(ner_text),token,ner_text)) %>%
  select(-pos,-tag,-head_token_id,-first,-dep_rel,-nounphrase,-ner_text)

# generate a dfm from this

cfpb_ner_docs<-cfpb_sp_ner %>%
  group_by(doc_id) %>%
  summarize(text=paste(ner_token, collapse=" ")) %>%
  mutate(doc_id=as.numeric(str_replace_all(doc_id,"text",""))) %>%
  arrange(doc_id)

# extract all the common noun phrases
phrases<-TASSL_dfm(cfpb_ner_docs$text,
                 min.prop = .001) %>%
  as.data.frame() %>%
  select(contains("_"),-doc_id) %>%
  colMeans() %>%
  sort(decreasing = T) %>%
  names()

phrases[1:30]

rm(cfpb_sp_ner,cfpb_lemma_docs,
   cfpb_sp_tagged)

