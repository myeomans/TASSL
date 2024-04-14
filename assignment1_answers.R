#####################################################
#                                                   #       
#  Text Analysis for Social Scientists and Leaders  #
#                                                   #
#              Assignment 1 Answers                 #
#                                                   #
#                                                   #
#####################################################

###############################################################
###############################################################
# Part 1 - Yelp Data
###############################################################
###############################################################

# Review data
review_dat<-readRDS("data/review_dat.RDS")

# Business data
businesses<-readRDS("data/businessset.RDS")
# First thing - check variables

names(review_dat)

names(businesses)

businesses<-businesses %>%
  # remove the ones we don't need
  filter(business_id%in%review_dat$business_id) %>%
  # One variable name overlaps, so we rename one
  rename(average_stars="stars") %>%
  # convert to numeric 
  mutate(price=as.numeric(RestaurantsPriceRange2))


# We want to use reviews to predict price data, but price is in businesses, not review_dat

# To move the business data over to the review data, we use left_join

review_dat <- review_dat %>%
  left_join(businesses,
            by="business_id")

# First, we need to split the data into training and testing samples
train_split=sample(1:nrow(review_dat),round(nrow(review_dat)/2))

# create our prediction variables
dfm3<-TASSL_dfm(review_dat$text,ngrams=1) %>%
  convert(to="data.frame") %>%
  select(-doc_id)

###############################################################
# FIRST, THE GENDER MODEL
###############################################################

trainX<-dfm3 %>%
  slice(train_split) %>%
  as.matrix()

trainY<-review_dat %>%
  slice(train_split) %>%
  pull(male)

testX<-dfm3 %>% 
  slice(-train_split) %>%
  as.matrix()

testY<-review_dat %>%
  slice(-train_split) %>%
  pull(male)

# Put training data into LASSO model (note - glmnet requires a matrix)

lasso_model<-cv.glmnet(x=trainX,y=trainY)

# generate predictions for test data
test_predict<-predict(lasso_model,newx = testX)[,1]

# split the predictions in two, using the median

# note - the target Y variable is 1/0 so we have to convert to 1/0, not 2/1
test_predict_binary=ifelse(test_predict>median(test_predict),
                           1,
                           0) 

# calculate accuracy

round(100*mean(test_predict_binary==testY),3)

#####################
# Build a plot
#####################

# extract coefficients
plotCoefs<-lasso_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  

# merge frequencies
plotDat<-plotCoefs %>%
  left_join(data.frame(ngram=colnames(trainX),
                       freq=colMeans(trainX))) %>%
  mutate_at(vars(score,freq),~round(.,3))

# pipe into ggplot
plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient2(low="navyblue",
                        mid = "grey",
                        high="forestgreen",
                        midpoint = 0)+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 15)+  
  scale_x_continuous(limits = c(-.2,.25),
                     breaks = seq(-.2,.2,.1)) +
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Gender Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

###############################################################
# NEXT, THE STAR RATING MODEL
###############################################################

# it's five levels, not two! So let's convert to a binary split 
hist(review_dat$stars)


trainX<-dfm3 %>%
  slice(train_split) %>%
  as.matrix()

trainY<-review_dat %>%
  slice(train_split) %>%
  pull(stars)

testX<-dfm3 %>% 
  slice(-train_split) %>%
  as.matrix()

testY<-review_dat %>%
  slice(-train_split) %>%
  pull(stars)

# Put training data into LASSO model (note - glmnet requires a matrix)

lasso_model<-cv.glmnet(x=trainX,y=trainY)

# generate predictions for test data
test_predict<-predict(lasso_model,newx = testX)[,1]

# split the predictions in two, using the median

# note - the target Y variable is 4/2 so we have to convert to 4/2, not 2/1
test_predict_binary=ifelse(test_predict>median(test_predict),
                           4,
                           2) 

# calculate accuracy

round(100*mean(test_predict_binary==testY),3)

#####################
# Build a plot
#####################

# extract coefficients
plotCoefs<-lasso_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  

# merge frequencies
plotDat<-plotCoefs %>%
  left_join(data.frame(ngram=colnames(trainX),
                       freq=colMeans(trainX))) %>%
  mutate_at(vars(score,freq),~round(.,3))

# pipe into ggplot
plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient2(low="navyblue",
                        mid = "grey",
                        high="forestgreen",
                        midpoint = 0)+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 15)+  
  scale_x_continuous(#limits = c(-.2,.1),
                     breaks = seq(-.8,.2,.1)) +
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Star Rating Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))


###############################################################
###############################################################
# Part 2 - Glassdoor Data
###############################################################
###############################################################


glassdoor<-readRDS("data/glassdoorReviews.RDS")

# Note - there are two different text boxes!! "pros" and "cons"
glassdoor <- glassdoor%>%
  mutate(pros_wordcount=str_count(pros,"[[:alpha:]]+"),
         cons_wordcount=str_count(cons,"[[:alpha:]]+"))

gd_small<-glassdoor %>%
  filter(pros_wordcount>5 & cons_wordcount>5)

# before we randomize, use set.seed() to all get the same split
set.seed(02138)

# grab the first 40,000 rows after randomizing
gd_small<-gd_small %>%
  arrange(sample(1:n())) %>%
  slice(1:40000)

##############################################################
# split into train and test
train_split=sample(1:nrow(gd_small),20000)

gd_train<-gd_small%>%
  slice(train_split)

gd_test<-gd_small%>%
  slice(-train_split)

##############################################################
# Let's just look at amazon for now

gd_amazon_train<-gd_train %>%
  filter(company=="amazon")

gd_amazon_test<-gd_test %>%
  filter(company=="amazon")

# create our prediction variables from the pros text
dfm_amazon_train_pros<-TASSL_dfm(gd_amazon_train$pros,ngrams=1:2) %>%
  convert(to="matrix")

amazon_train_Y<-gd_amazon_train %>%
  pull(overall)

# Put training data into LASSO model

amazon_model_pros<-cv.glmnet(x=dfm_amazon_train_pros,
                             y=amazon_train_Y)

# First, let's test the model on the pros text from amazon
dfm_amazon_test_pros<-TASSL_dfm(gd_amazon_test$pros,
                                ngrams=1:2,
                                min.prop = 0) %>%
  dfm_match(colnames(dfm_amazon_train_pros)) %>%
  convert(to="matrix")

amazon_test_Y<-gd_amazon_test %>%
  pull(overall)

# generate predictions for test data
amazon_test_predict_pros<-predict(amazon_model_pros,
                                  newx = dfm_amazon_test_pros)[,1]

# estimate accuracy - use kendall's tau
pros_acc_p<-kendall_acc(amazon_test_predict_pros,amazon_test_Y)

############################################

# Let's apply the same model to the cons text

dfm_amazon_test_cons<-TASSL_dfm(gd_amazon_test$cons,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_amazon_train_pros)) %>%
  convert(to="matrix")

# generate predictions for test data
amazon_test_predict_cons<-predict(amazon_model_pros,
                                  newx = dfm_amazon_test_cons)[,1]

# estimate accuracy
cons_acc_p<-kendall_acc(amazon_test_predict_cons,amazon_test_Y)

#################################################
# Question 3
#################################################

# create our prediction variables from the cons text
dfm_amazon_train_cons<-TASSL_dfm(gd_amazon_train$cons,ngrams=1:2) %>%
  convert(to="matrix")

amazon_model_cons<-cv.glmnet(x=dfm_amazon_train_cons,
                             y=amazon_train_Y)

dfm_amazon_test_cons<-TASSL_dfm(gd_amazon_test$cons,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_amazon_train_cons)) %>%
  convert(to="matrix")

# generate predictions for test data
amazon_test_predict_cons<-predict(amazon_model_cons,
                                  newx = dfm_amazon_test_cons)[,1]
# estimate accuracy
cons_acc_c<-kendall_acc(amazon_test_predict_cons,amazon_test_Y)

dfm_amazon_test_pros<-TASSL_dfm(gd_amazon_test$pros,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_amazon_train_cons)) %>%
  convert(to="matrix")

# generate predictions for test data
amazon_test_predict_pros<-predict(amazon_model_cons,
                                  newx = dfm_amazon_test_pros)[,1]
# estimate accuracy
pros_acc_c<-kendall_acc(amazon_test_predict_pros,amazon_test_Y)

#############################################
# Combine accuracy estimates for a plot
#############################################
bind_rows(pros_acc_p %>%
            mutate(test="Pros",
                   train="Pros Model"),
          cons_acc_p %>%
            mutate(test="Cons",
                   train="Pros Model"),
          pros_acc_c %>%
            mutate(test="Pros",
                   train="Cons Model"),
          cons_acc_c %>%
            mutate(test="Cons",
                   train="Cons Model")) %>%
  ggplot(aes(x=test,color=test,
             y=acc,ymin=lower,ymax=upper)) +
  geom_point() +
  facet_wrap(~train) +
  geom_errorbar(width=.4) +
  theme_bw() +
  labs(x="Test Data",y="Accuracy") +
  geom_hline(yintercept = 50) +
  
  theme(axis.text = element_text(size=24),
        axis.title = element_text(size=24),
        strip.text = element_text(size=24),
        panel.grid=element_blank(),
        strip.background = element_rect(fill="white"),
        legend.position="none")

#################################################
# Question 4
#################################################

# Pros Plot

# extract coefficients
plotCoefs<-amazon_model_pros %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  

# merge frequencies
plotDat<-plotCoefs %>%
  left_join(data.frame(ngram=colnames(dfm_amazon_train_pros),
                       freq=colMeans(dfm_amazon_train_pros))) %>%
  mutate_at(vars(score,freq),~round(.,3))

# pipe into ggplot
plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient2(low="navyblue",
                        mid = "grey",
                        high="forestgreen",
                        midpoint = 0)+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 15)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Pros Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

#################
# Cons Plot
#################

# extract coefficients
plotCoefs<-amazon_model_cons %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  

# merge frequencies
plotDat<-plotCoefs %>%
  left_join(data.frame(ngram=colnames(dfm_amazon_train_cons),
                       freq=colMeans(dfm_amazon_train_cons))) %>%
  mutate_at(vars(score,freq),~round(.,3))

# pipe into ggplot
plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient2(low="navyblue",
                        mid = "grey",
                        high="forestgreen",
                        midpoint = 0)+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 15)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Cons Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

#################################################
# Question 5
#################################################

# Let's do bi- and tri-grams

################
################
# Pros text
################
################
dfm_amazon_train_pros<-TASSL_dfm(gd_amazon_train$pros,
                                 ngrams=2:3) %>%
  convert(to="matrix")

amazon_model_pros<-cv.glmnet(x=dfm_amazon_train_pros,
                             y=amazon_train_Y)

dfm_amazon_test_pros<-TASSL_dfm(gd_amazon_test$pros,
                                ngrams=2:3, 
                                min.prop = 0)  %>%
  dfm_match(colnames(dfm_amazon_train_pros)) %>%
  convert(to="matrix")

# generate predictions for test data
amazon_test_predict_pros<-predict(amazon_model_pros,
                                  newx = dfm_amazon_test_pros)[,1]
# estimate accuracy
pros_acc_c<-kendall_acc(amazon_test_predict_pros,amazon_test_Y)

pros_acc_c

# Pros Plot

# extract coefficients
plotCoefs<-amazon_model_pros %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  

# merge frequencies
plotDat<-plotCoefs %>%
  left_join(data.frame(ngram=colnames(dfm_amazon_train_pros),
                       freq=colMeans(dfm_amazon_train_pros))) %>%
  mutate_at(vars(score,freq),~round(.,3))

# pipe into ggplot
plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient2(low="navyblue",
                        mid = "grey",
                        high="forestgreen",
                        midpoint = 0)+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 15)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Pros Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

################
################
# Cons text
################
################

dfm_amazon_train_cons<-TASSL_dfm(gd_amazon_train$cons,
                                 ngrams=2:3) %>%
  convert(to="matrix")

amazon_model_cons<-cv.glmnet(x=dfm_amazon_train_cons,
                             y=amazon_train_Y)

dfm_amazon_test_cons<-TASSL_dfm(gd_amazon_test$cons,
                                ngrams=2:3, 
                                min.prop = 0)  %>%
  dfm_match(colnames(dfm_amazon_train_cons)) %>%
  convert(to="matrix")

# generate predictions for test data
amazon_test_predict_cons<-predict(amazon_model_cons,
                                  newx = dfm_amazon_test_cons)[,1]
# estimate accuracy
cons_acc_c<-kendall_acc(amazon_test_predict_cons,amazon_test_Y)


# extract coefficients
plotCoefs<-amazon_model_cons %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  

# merge frequencies
plotDat<-plotCoefs %>%
  left_join(data.frame(ngram=colnames(dfm_amazon_train_cons),
                       freq=colMeans(dfm_amazon_train_cons))) %>%
  mutate_at(vars(score,freq),~round(.,3))

# pipe into ggplot
plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient2(low="navyblue",
                        mid = "grey",
                        high="forestgreen",
                        midpoint = 0)+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 15)+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Cons Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))


#################################################
# Question 6
#################################################


gd_microsoft_train<-gd_train %>%
  filter(company=="microsoft")

gd_microsoft_test<-gd_test %>%
  filter(company=="microsoft")

microsoft_train_Y<-gd_microsoft_train %>%
  pull(overall)

microsoft_test_Y<-gd_microsoft_test %>%
  pull(overall)


# create our prediction variables from the cons text
dfm_microsoft_train_cons<-TASSL_dfm(gd_microsoft_train$cons,ngrams=1:2) %>%
  convert(to="matrix")

microsoft_model_cons<-cv.glmnet(x=dfm_microsoft_train_cons,
                                y=microsoft_train_Y)

dfm_microsoft_test_cons<-TASSL_dfm(gd_microsoft_test$cons,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_microsoft_train_cons)) %>%
  convert(to="matrix")

# generate predictions for test data
microsoft_test_predict_msft<-predict(microsoft_model_cons,
                                     newx = dfm_microsoft_test_cons)[,1]
# estimate accuracy
cons_acc_msft<-kendall_acc(microsoft_test_predict_msft,microsoft_test_Y)


dfm_microsoft_test_cons<-TASSL_dfm(gd_microsoft_test$cons,ngrams=1:2)  %>%
  dfm_match(colnames(dfm_amazon_train_cons)) %>%
  convert(to="matrix")

# generate predictions for test data
microsoft_test_predict_amzn<-predict(amazon_model_cons,
                                     newx = dfm_microsoft_test_cons)[,1]
# estimate accuracy
cons_acc_amzn<-kendall_acc(microsoft_test_predict_amzn,microsoft_test_Y)


cons_acc_msft

cons_acc_amzn

