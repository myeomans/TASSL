#####################################################
#                                                   #       
#  Text Analysis for Social Scientists and Leaders  #
#                                                   #
#              Assignment 2 Answers                 #
#                                                   #
#                                                   #
#####################################################


######################################################################
# Load descriptions data
######################################################################

jd_small<-readRDS("data/jd_small.RDS") %>%
  rename(salary="SalaryNormalized")

set.seed(01238)
train_split=sample(1:nrow(jd_small),8000)

jd_small_train<-jd_small[train_split,]
jd_small_test<-jd_small[-train_split,]


jd_small_dfm_train<-TASSL_dfm(jd_small_train$FullDescription,ngrams=1)

jd_small_dfm_test<-TASSL_dfm(jd_small_test$FullDescription,
                            ngrams=1,min.prop = 0) %>%
  dfm_match(colnames(jd_small_dfm_train))


jd_model<-glmnet::cv.glmnet(x=jd_small_dfm_train %>%
                               as.matrix(),
                             y=jd_small_train$salary)

plot(jd_model)

#### Interpret with a coefficient plot
plotDat<-jd_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  %>%
  # add ngram frequencies for plotting
  left_join(data.frame(ngram=colnames(jd_small_dfm_train),
                       freq=colMeans(jd_small_dfm_train)))

plotDat %>%
  mutate_at(vars(score,freq),~round(.,3))

plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient(low="green",high="purple")+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 15,force = 6)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Description")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))


#### Evaluate Accuracy
test_ngram_predict<-predict(jd_model,
                            newx = jd_small_dfm_test %>%
                              as.matrix())[,1]

acc_ngram<-kendall_acc(jd_small_test$salary,test_ngram_predict)

acc_ngram

hist(test_ngram_predict)

hist(jd_small_test$salary)

############### Benchmarks

# Create benchmarks

jd_small_test <- jd_small_test %>%
  mutate(text_wdct=str_count(FullDescription,"[[:alpha:]]+"),
         model_random=sample(test_ngram_predict),
         sentiment=sentiment_by(FullDescription)$ave_sentiment)

acc_wdct<-kendall_acc(jd_small_test$salary,
                      jd_small_test$text_wdct)

acc_wdct



acc_random<-kendall_acc(jd_small_test$salary,
                        jd_small_test$model_random)

acc_random



acc_sentiment<-kendall_acc(jd_small_test$salary,
                        jd_small_test$sentiment)

acc_sentiment


############ Find examples

# store predictions in data, calculate accuracy
jd_small_test<-jd_small_test %>%
  mutate(prediction=test_ngram_predict,
         error=abs(salary-prediction),
         bias=salary-prediction)

close_high<-jd_small_test %>%
  filter(salary>60000 & error<5000) %>%
  select(FullDescription,salary,prediction)

close_low<-jd_small_test %>%
  filter(salary<20000 & error<5000) %>%
  select(FullDescription,salary,prediction)

close_high
close_high %>%
  slice(1:2) %>%
  pull(FullDescription)

close_low
close_low %>%
  slice(1:2) %>%
  pull(FullDescription)

# Error analysis - find biggest misses

jd_small_test %>%
  ggplot(aes(x=prediction)) +
  geom_histogram()

jd_small_test %>%
  ggplot(aes(x=salary)) +
  geom_histogram()

miss_high<-jd_small_test %>%
  arrange(bias) %>%
  slice(1:10) %>%
  select(FullDescription,salary,prediction)

miss_low<-jd_small_test %>%
  arrange(-bias) %>%
  slice(1:10) %>%
  select(FullDescription,salary,prediction)

miss_low
miss_low%>%
  slice(1:2) %>%
  pull(FullDescription)

miss_high
miss_high%>%
  slice(3) %>%
  pull(FullDescription)
