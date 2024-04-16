#####################################################
#                                                   #       
#  Text Analysis for Social Scientists and Leaders  #
#                                                   #
#              Assignment 4 Answers                 #
#                                                   #
#                                                   #
#####################################################


source("vectorFunctions.R")

# The real word vector files are ~ 6GB - too big! This is a smaller version,
# containing only the 50,000 most common words
vecSmall<-readRDS("data/vecSmall.RDS")

# Word frequency file - to reweight common words
load("data/wfFile.RData")

set.seed(2022)

#####################################################################################

ecMain<-readRDS("data/earningsDat.RDS") %>%
  mutate(EPS_surprise=(EPS_actual-EPS_consens)/EPS_actual)

ecQA<-readRDS("data/earningsQandA.RDS")

########################################
# Q1
########################################
ec_train<-ecMain %>%
  filter(FY%in%c("2010","2011"))
ec_test<-ecMain %>%
  filter(FY=="2012")

ecQA_train<-ecQA %>%
  filter(callID%in%ec_train$callID)
ecQA_test<-ecQA %>%
  filter(callID%in%ec_test$callID)

########################################
# Q2
########################################


ec_dfm_train<-TASSL_dfm(ec_train$opening_speech,ngrams=2:3)

ec_dfm_test<-TASSL_dfm(ec_test$opening_speech,ngrams=2:3) %>%
  dfm_match(colnames(ec_dfm_train))


ec_ngram_model<-glmnet::cv.glmnet(x=ec_dfm_train %>%
                                    as.matrix(),
                                  y=ec_train$EPS_actual)
plot(ec_ngram_model)

#### Interpret with a coefficient plot
ec_ngram_model %>%
  coef(s="lambda.min") %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  %>%
  # add ngram frequencies for plotting
  left_join(data.frame(ngram=colnames(ec_dfm_train),
                       freq=colMeans(ec_dfm_train))) %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient(low="blue",high="red")+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 40,force = 6)+
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Speech")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))


########################################
# Q3
########################################

# vectors

vdat_train<-vecCheck(ec_train$opening_speech,vecSmall,wfFile)

vdat_test<-vecCheck(ec_test$opening_speech,vecSmall,wfFile)

# train models 
lasso_vec<-glmnet::cv.glmnet(x=vdat_train,
                             y=ec_train$EPS_actual)

lasso_combo<-glmnet::cv.glmnet(x=cbind(vdat_train,
                                       ec_dfm_train %>%
                                         as.matrix()),
                               y=ec_train$EPS_actual)

# apply to test set
test_ngram_predict<-predict(ec_ngram_model,
                            s="lambda.min",
                            newx = ec_dfm_test %>%
                              as.matrix())[,1]

test_vec_predict<-predict(lasso_vec,newx = vdat_test,
                          s="lambda.min")


test_combo_predict<-predict(lasso_combo,
                            newx = cbind(vdat_test,
                                         ec_dfm_test %>%
                                           as.matrix()),
                            s="lambda.min")


#### Evaluate Accuracy

acc_vec<-kendall_acc(test_vec_predict,ec_test$EPS_actual)

acc_combo<-kendall_acc(test_combo_predict,ec_test$EPS_actual)

acc_ngram<-kendall_acc(ec_test$EPS_actual,test_ngram_predict)


########################################
# Q4
########################################

# Other benchmarks

ec_test <- ec_test %>%
  mutate(speech_sent=syuzhet::get_sentiment(opening_speech),
         speech_wdct=str_count(opening_speech,"[[:alpha:]]+"))


acc_wdct<-kendall_acc(ec_test$EPS_actual,ec_test$speech_wdct)

acc_sent<-kendall_acc(ec_test$EPS_actual,ec_test$speech_sent)

acc_wdct
acc_sent


bind_rows(acc_ngram %>%
            mutate(model="Ngrams"),
          acc_vec%>%
            mutate(model="word2vec"),
          acc_combo%>%
            mutate(model="Ngrams + word2vec"),
          acc_wdct %>%
            mutate(model="Word Count"),
          acc_sent %>%
            mutate(model="Traditional Sentiment")
) %>% 
  ggplot(aes(x=model,color=model,
             y=acc,ymin=lower,ymax=upper)) +
  geom_hline(yintercept=50) +              
  geom_point() +                           
  geom_errorbar(width=.4) +                
  labs(x="Model",                 
       y="Accuracy") +  
  coord_flip() + 
  theme_bw() +                             
  theme(axis.text=element_text(size=20),
        axis.title=element_text(size=24),
        panel.grid = element_blank(),
        strip.text=element_text(size=24),
        strip.background = element_rect(fill="white"),
        legend.position = "none")          # other design options


###################################################
# Q5 
###################################################

ec_test <- ec_test %>%
  mutate(vec_pred=test_vec_predict,
         ngram_pred=test_ngram_predict,
         vec_error=abs(vec_pred-EPS_actual),
         ngram_error=abs(ngram_pred-EPS_actual))

# find the range for the predictions 
ec_test %>%
  ggplot(aes(x=vec_error,y=ngram_error)) +
  geom_point() +
  scale_x_continuous(trans='sqrt') +
  scale_y_continuous(trans='sqrt') +
  theme_bw()

# 
ec_test %>%
  filter(vec_error<.25 & ngram_error > 1) %>%
  arrange(EPS_actual) %>%
  slice(1) %>%
  pull(opening_speech)

ec_test %>%
  filter(vec_error<.25 & ngram_error > 1) %>%
  arrange(-EPS_actual) %>%
  slice(1) %>%
  pull(opening_speech)

###################################################
# Q6 
###################################################

positive_dict<-textdata::lexicon_loughran() %>%
  filter(sentiment=="positive") %>%
  pull(word)

positive_dict_doc<-positive_dict %>%
  paste(collapse=" ")

pos_dict<-list(pos=positive_dict) %>%
  dictionary()

pos_dict_bow<-ec_test$opening_speech %>%
  tokens() %>%
  dfm() %>%
  dfm_lookup(pos_dict) %>%
  convert(to="matrix") %>%
  apply(2, function(x) x/ec_test$speech_wdct)


pos_sims<-vecSimCalc(x=ec_test$opening_speech,
                     y=positive_dict,
                     vecfile=vecSmall,
                     wffile = wfFile,
                     PCAtrim = 1)

acc_possims<-kendall_acc(ec_test$EPS_actual,pos_sims)

acc_possims

acc_posbow<-kendall_acc(ec_test$EPS_actual,pos_dict_bow)

bind_rows(acc_posbow %>%
            mutate(model="L-M Positive Dictionary"),
          acc_possims%>%
            mutate(model="L-M Positive DDR"),
          acc_wdct %>%
            mutate(model="Word Count"),
          acc_sent %>%
            mutate(model="Traditional Sentiment")
) %>% 
  ggplot(aes(x=model,color=model,
             y=acc,ymin=lower,ymax=upper)) +
  geom_hline(yintercept=50) +              
  geom_point() +                           
  geom_errorbar(width=.4) +                
  labs(x="Annotator Name",                 
       y="Accuracy") +                     
  theme_bw() +     
  coord_flip() + 
  theme(axis.text=element_text(size=20),
        axis.title=element_text(size=24),
        panel.grid = element_blank(),
        strip.text=element_text(size=24),
        strip.background = element_rect(fill="white"),
        legend.position = "none")          # other design options

####################################
# Q7
####################################

all8<-ecMain %>%
  filter(FY %in% c(2011,2012)) %>%
  group_by(IBES_ID) %>%
  summarize(n=n())%>%
  filter(n==8)%>% 
  pull(IBES_ID)

# list of companies that have all 8 quarters in the data

all8Set<-ecMain %>%
  filter(FY %in% c(2011,2012)) %>%
  filter(IBES_ID%in% all8)

# merge initial opening speeches back into data
all8Set<-all8Set %>%
  left_join(all8Set %>%
              filter(FY=="2011" & FQ==1) %>%
              select(IBES_ID,
                     first_speech="opening_speech")) %>%
  filter(FY %in% c(2011,2012)) %>%
  filter(!(FY=="2011" & FQ==1))


all8Set$vecSim=NA
tpb<-txtProgressBar(0,nrow(all8Set))
for (z in 1:nrow(all8Set)){ 
  all8Set[z,]$vecSim=vecSimCalc(x=all8Set[z,]$opening_speech,
                                y=all8Set[z,]$first_speech,
                                vecfile=vecSmall,
                                wffile=wfFile)
  setTxtProgressBar(tpb,z)
}

all8Set %>%
  group_by(FY,FQ) %>%
  summarize(m=mean(vecSim),
            se=sd(vecSim)/sqrt(n())) %>%
  ggplot(aes(x=FQ,y=m,ymin=m-se,ymax=m+se)) +
  theme_bw() +
  geom_point() +
  geom_errorbar()+
  facet_wrap(~FY)


# An alternate approach - if you want to use PCAtrim

all8Set$vecSim=NA
tpb<-txtProgressBar(0,length(unique(all8Set$IBES_ID)))
z=0
for (ID in unique(all8Set$IBES_ID)){ 
  ID_rows=which(all8Set$IBES_ID==ID)
  
  all8Set[ID_rows,]$vecSim=vecSimCalc(x=all8Set[ID_rows,]$opening_speech,
                                      # we only need one because the first speech is the same for everyone
                                      y=all8Set[ID_rows[1],]$first_speech,
                                      vecfile=vecSmall,
                                      wffile=wfFile,
                                      PCAtrim=1)
  setTxtProgressBar(tpb,z)
}

all8Set %>%
  group_by(FY,FQ) %>%
  summarize(m=mean(vecSim),
            se=sd(vecSim)/sqrt(n())) %>%
  ggplot(aes(x=FQ,y=m,ymin=m-se,ymax=m+se)) +
  theme_bw() +
  geom_point() +
  geom_errorbar()+
  facet_wrap(~FY)


################################################
# Q 8
################################################

ecQA %>%
  filter(asker==1) %>%
  group_by(callID,askerID) %>%
  summarize(qcount=n()) %>%
  group_by(askerID) %>%
  summarize(m=mean(qcount),
            se=sd(qcount)/sqrt(n())) %>%
  mutate(askerID=as.numeric(askerID)) %>%
  filter(askerID<20) %>%
  ggplot(aes(x=askerID,y=m,
             ymin=m-se,ymax=m+se)) +
  geom_point() +
  geom_errorbar() +
  theme_bw() +
  labs(x="Asker Order",y="Number of Questions Asked")


################################################
# Q 9
################################################

# Join turn-level data (from the first ten questions) to the call-level data
ec_train_merged <- ec_train %>%
  # combine answer text as a single document and merge
  left_join(ecQA_train %>%
              filter(asker==0 & question <=10) %>%
              group_by(callID) %>%
              summarize(answertext=paste(text,collapse=" "),
                        answer_wdct=str_count(answertext,"[[:alpha:]]+"))) %>%
  # combine question text as a single document and merge
  left_join(ecQA_train %>%
              filter(asker==1 & question <=10) %>%
              group_by(callID) %>%
              summarize(questiontext=paste(text,collapse=" "),
                        question_wdct=str_count(questiontext,"[[:alpha:]]+")))

# Bonus Lemmas
spacyr::spacy_initialize()

ec_train_merged$question_lemmas<-spacyr::spacy_parse(ec_train_merged$questiontext,
                                                     nounphrase = F,
                                                     lemma = T,
                                                     dependency = F,
                                                     pos = F,
                                                     tag=F) %>%
  group_by(doc_id) %>%
  summarize(text=paste(lemma, collapse=" ")) %>%
  mutate(doc_id=as.numeric(str_replace_all(doc_id,"text",""))) %>%
  arrange(doc_id) %>%
  pull(text)

ec_train_merged$answer_lemmas<-spacyr::spacy_parse(ec_train_merged$answertext,
                                                   nounphrase = F,
                                                   lemma = T,
                                                   dependency = F,
                                                   pos = F,
                                                   tag=F) %>%
  group_by(doc_id) %>%
  summarize(text=paste(lemma, collapse=" ")) %>%
  mutate(doc_id=as.numeric(str_replace_all(doc_id,"text",""))) %>%
  arrange(doc_id) %>%
  pull(text)

# Questions First

ec_train_dfm_q<-TASSL_dfm(ec_train_merged$question_lemmas,ngrams = 1:2)

EPSmodel_q<-cv.glmnet(x=as.matrix(ec_train_dfm_q),
                      y=(ec_train_merged$EPS_actual))

plot(EPSmodel_q)

#### Interpret with a coefficient plot
plotDat<-EPSmodel_q %>%
  coef(s="lambda.min") %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  %>%
  # add ngram frequencies for plotting
  left_join(data.frame(ngram=colnames(ec_train_dfm_q),
                       freq=colMeans(ec_train_dfm_q)/10))

plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient(low="blue",high="red")+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 20,force = 6)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Question")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

########## Answers next

ec_train_dfm_a<-TASSL_dfm(ec_train_merged$answer_lemmas,ngrams = 1:2)


EPSmodel_a<-cv.glmnet(x=as.matrix(ec_train_dfm_a),
                      y=(ec_train_merged$EPS_actual))

plot(EPSmodel_a)

#### Interpret with a coefficient plot
plotDat<-EPSmodel_a %>%
  coef(s="lambda.min") %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  %>%
  # add ngram frequencies for plotting
  left_join(data.frame(ngram=colnames(ec_train_dfm_a),
                       freq=colMeans(ec_train_dfm_a)/10))

plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient(low="blue",high="red")+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 20,force = 6)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Answer")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

################################################
# Q 10
################################################

ec_train_polite_q<-politeness(ec_train_merged$questiontext,parser="spacy")

ec_train_polite_a<-politeness(ec_train_merged$answertext,parser="spacy")

qa_data<-bind_rows(ec_train_polite_q %>%
                     mutate(source="questions"),
                   ec_train_polite_a%>%
                     mutate(source="answers"))


politenessPlot(qa_data %>%
                 select(-source),
               qa_data$source,
               middle_out=.05,
               drop_blank = .1,
               top_title="Questions vs Answers")

EPSmodel_pol_qa<-cv.glmnet(x=qa_data %>%
                            select(-source) %>%
                            as.matrix(),
                          y=1*(qa_data$source=="questions"))

plot(EPSmodel_pol_qa)

#### Interpret with a coefficient plot
plotDat<-EPSmodel_pol_qa %>%
  coef(s="lambda.min") %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  %>%
  # add ngram frequencies for plotting
  left_join(data.frame(ngram=colnames(qa_data %>%
                                        select(-source)),
                       freq=colMeans(qa_data %>%
                                       select(-source))))

plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient(low="blue",high="red")+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 20,force = 6)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Turn")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))


################################################
# Question 11
################################################

# Join call-level data to the turn-level data, focus on answers
ecQA_train_merged<-ecQA_train %>%
  left_join(ec_train %>%
              select(callID,FY,FQ,EPS_actual,EPS_consens)) %>%
  mutate(first_quarter=1*(FQ==1),
         quarter=as.character(FQ)) %>%
  filter(asker==0 & question<=5)


ecQA_dfmx_train<-TASSL_dfm(ecQA_train_merged$text,ngrams=1:2)

quarter1model<-cv.glmnet(x=ecQA_dfmx_train %>%
                          as.matrix(),
                        y=ecQA_train_merged$first_quarter)


#### Interpret with a coefficient plot
plotDat<-quarter1model %>%
  coef(s="lambda.min") %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  %>%
  # add ngram frequencies for plotting
  left_join(data.frame(ngram=colnames(ecQA_dfmx_train),
                       freq=colMeans(ecQA_dfmx_train)))

plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient(low="blue",high="red")+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 20,force = 6)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Call")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

################################################
# Q 12
################################################

# multinomial

allquartermodel<-cv.glmnet(x=ecQA_dfmx_train %>%
                          as.matrix(),
                        y=ecQA_train_merged$quarter,
                        family="multinomial")



# Join call-level data to the turn-level data, focus on answers
ecQA_test_merged<-ecQA_test %>%
  left_join(ec_test %>%
              select(callID,FY,FQ,EPS_actual,EPS_consens)) %>%
  mutate(first_quarter=1*(FQ==1),
         quarter=as.character(FQ)) %>%
  filter(asker==0 & question<=5)


ecQA_dfmx_test<-TASSL_dfm(ecQA_test_merged$text,
                        ngrams=1:2,min.prop=0)  %>%
  dfm_match(colnames(ecQA_dfmx_train))


test_q1_predict<-predict(quarter1model,
                            newx = ecQA_dfmx_test,
                            s="lambda.min")


test_qall_predict<-predict(allquartermodel,
                         newx = ecQA_dfmx_test,
                         type="response",
                         s="lambda.min")


#### Evaluate Accuracy

acc_q1<-kendall_acc(test_q1_predict,
                    ecQA_test_merged$first_quarter)


acc_qall<-kendall_acc(test_qall_predict[,1,1],
                    ecQA_test_merged$first_quarter)

acc_q1
acc_qall
