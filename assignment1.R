#####################################################
#                                                   #       
#  Text Analysis for Social Scientists and Leaders  #
#                                                   #
#                  Assignment 1                     #
#                                                   #
#                                                   #
#####################################################
library(tidyverse) # useful for almost everything
library(quanteda) # text analysis workhorse
library(textclean) # extra pre-processing
library(ggrepel) # for plots
library(glmnet) # Our estimation model
library(pROC)  # binary prediction accuracy
library(doc2concrete) # ngramTokens


#################################################
#################################################
# Part 1 - DFM basics
#################################################
#################################################


######### Simple bag of words

testDocs<-c("This is a test sentence.", 
            "I am providing another sentence to test this.",
            "This isn't a sentence",
            "This is a test document. It has 2 sentences")


# a quick word on tidyverse - the %>% is called "pipe"
# it takes the finished object from the current line
# and inserts it as the first argument to the function on the next line

# so, these two commands are identical
testDocs %>%
  tokens()

tokens(testDocs)


# Anyways, first we need to split up the sentences into "tokens" - (usually words)

testDocs %>%
  tokens()

# We then count how often each token occurs in each document 
# This produces a "document feature matrix" (or document term matrix)
# One row for each doc, one column for each feature
testDocs %>%
  tokens() %>%
  dfm()

# We can also combine adjoining words into "bigrams"

testDocs %>%
  tokens() %>%
  tokens_ngrams(2) %>%
  dfm()

# often people combine multiple token lengths together, as ngrams
testDocs %>%
  tokens() %>%
  tokens_ngrams(1:2) %>%
  dfm()

# Many different ways to tokenize - see the help file for options

?tokens

# We can stem words

testDocs %>%
  tokens(remove_punct=TRUE) %>%
  tokens_wordstem()

# we can remove punctuation
testDocs %>%
  tokens(remove_punct=TRUE) %>%
  tokens_ngrams(1:2)

# we can remove numbers
testDocs %>%
  tokens(remove_numbers=TRUE) %>%
  tokens_ngrams(1:2)

# contractions are done with a function from textclean
testDocs %>%
  replace_contraction() %>%
  tokens()


# dfm converts everything to lower case by default, but we can turn this off
testDocs %>%
  tokens() %>%
  dfm()

testDocs %>%
  tokens() %>%
  dfm(tolower=FALSE)

# we can also remove "stop words"
testDocs %>%
  tokens() %>%
  tokens_select(pattern = stopwords("en"), 
                selection = "remove") %>%
  tokens_ngrams(1:2)

# This is the built-in quanteda stopword list
stopwords("en")

# we can create our own custom list if we like
testDocs %>%
  tokens() %>%
  tokens_select(pattern = c("a","is","the"), 
                selection = "remove") %>%
  tokens_ngrams(1:2)


# Instead of removing common words, we can downweight them, using tfidf

dox<-c("This is a sentence.",
       "this is also a sentence.",
       "here is a rare word",
       "here is another word.",
       "and other sentences")

# Without tfidf, all words are given the same weight
dox %>%
  tokens(remove_punct= TRUE) %>%
  dfm() %>%
  convert(to="data.frame") %>%
  select(-doc_id) %>%
  round(2)

# Here, rare words are given more weight
dox %>%
  tokens(remove_punct= TRUE) %>%
  dfm() %>%
  dfm_tfidf() %>%
  convert(to="data.frame") %>%
  select(-doc_id) %>%
  round(2)

# We can also remove words that are too rare to learn anything about

dox %>%
  tokens(remove_punct= TRUE) %>%
  dfm() %>%
  dfm_trim(min_docfreq = 2) %>%
  convert(to="data.frame") %>%
  select(-doc_id) %>%
  round(2)

# Usually we do this by proportion of words

dox %>%
  tokens(remove_punct= TRUE) %>%
  dfm() %>%
  dfm_trim(min_docfreq = .25,docfreq_type="prop") %>%
  convert(to="data.frame") %>%
  select(-doc_id) %>%
  round(2)

# Typically the cut-off gets set around 1% of documents

# Here  I am loading a function that saves all of our defaults in one place
source("TASSL_dfm.R")


TASSL_dfm(dox)

# we can easily modify the defaults of our custom arguments
TASSL_dfm(dox, ngrams=2)

TASSL_dfm(dox, stop.words = FALSE)

TASSL_dfm(dox, min.prop=.25)

# Note... this is a bit rudimentary
# If you prefer, you can use a more robust function I wrote for a different package
# install.packages("doc2concrete")
library(doc2concrete)

ngramTokens(dox)

#################################################
#################################################
# Part 2 - Yelp data
#################################################
#################################################


######### New data - restaurant reviews

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

# To merge the business data into the review data, we use left_join()

review_dat <- review_dat %>%
  left_join(businesses,
            by="business_id")

names(review_dat)

# Calculate a 1-gram feature count matrix for the review data, with no dropped words
dfm1<-TASSL_dfm(review_dat$text,
              ngrams=1,
              min.prop=0,
              stop.words = FALSE)

dim(dfm1) # >10k ngrams! Too many

# most common words - obvious
sort(colMeans(dfm1),decreasing=TRUE)[1:20]

# least common words
sort(colMeans(dfm1))[1:20]

######## Ok, let's build a model to predict price!

# First, let's look at our price data

table(review_dat$price)

# Let's only use 1-grams for now
dfm3<-TASSL_dfm(review_dat$text,ngrams=1) %>%
  convert(to="data.frame") %>%
  select(-doc_id)

# Lots of words
dim(dfm3)

#  Most common words in 1- and 2-price reviews... lots of the same words!
sort(colMeans(dfm3[review_dat$price==2,]),decreasing=T)[1:20]

sort(colMeans(dfm3[review_dat$price==1,]),decreasing=T)[1:20]

# What we really care about is - does the presence of a word predict price?

# A simple start - correlate each word with star rating

correlations<-dfm3 %>%
  summarise_all(~round(cor(.,review_dat$price),3)) %>%
  unlist()

# Ten lowest associations
sort(correlations)[1:10]

# Ten highest associations
rev(sort(correlations))[1:10]

# note - same as:
sort(correlations,decreasing=TRUE)[1:10]

# As we said in class we are not often interested in the effects of individual words
# Instead, we care more about how all the words perform as a class

# To do this, we will use the cv.glmnet() function to build a model

# First, we need to split the data into training and testing samples
train_split=sample(1:nrow(review_dat),round(nrow(review_dat)/2))

length(train_split)

# create our prediction variables
dfm3<-TASSL_dfm(review_dat$text,ngrams=1) %>%
  convert(to="data.frame") %>%
  select(-doc_id)


trainX<-dfm3 %>%
  slice(train_split) %>%
  as.matrix()

trainY<-review_dat %>%
  slice(train_split) %>%
  pull(price)

testX<-dfm3 %>% 
  slice(-train_split) %>%
  as.matrix()

testY<-review_dat %>%
  slice(-train_split) %>%
  pull(price)

# Put training data into LASSO model (note - glmnet requires a matrix)

lasso_model<-cv.glmnet(x=trainX,y=trainY)

# let's plot the cross-validation curve to see if it's finding any signal
plot(lasso_model)

# generate predictions for test data
test_predict<-predict(lasso_model,newx = testX)[,1]

# Note that while the true answers are binary, the predictions are continuous
# Always check these distributions!!
hist(testY)
hist(test_predict)

# For now, let's just split the predictions in two, using the median

test_predict_binary=ifelse(test_predict>median(test_predict),
                           2,
                           1)
hist(test_predict_binary)

# quick plot of the split to make sure it looks right
plot(x=test_predict,y=test_predict_binary)


# This should have the same values as testY
hist(test_predict_binary)

# and we can calculate accuracy from that

round(100*mean(test_predict_binary==testY),3)

#### What is in the model? We can extract the coefficients

# lots of zeros
lasso_model %>%
  coef() %>%
  drop()

# let's get this in a data frame
lasso_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".")

# just the top
lasso_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  head(20)

# drop zeros, and save
plotCoefs<-lasso_model %>%
  coef() %>%
  drop() %>%
  as.data.frame() %>%
  rownames_to_column(var = "ngram") %>%
  rename(score=".") %>%
  filter(score!=0 & ngram!="(Intercept)" & !is.na(score))  

plotCoefs

# create a similar data frame with ngram frequencies
plotFreqs<-data.frame(ngram=colnames(trainX),
                      freq=colMeans(trainX))


# combine data, round for easy reading
plotDat<-plotCoefs %>%
  left_join(plotFreqs) %>%
  mutate_at(vars(score,freq),~round(.,3))

head(plotDat)

# here's our first plot, with minimal customization
plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  geom_point()

# Problems:
# Bad axis labels
# no point labels
# I don't like the default grey background
# legend is redundant

plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  geom_point() +
  geom_label() +
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Review")+
  theme(legend.position = "none")

# More problems:
# wasted space in Y axis
# lots of overlapping labels
# small axis labels
# i don't like the default colors

# colors we can set manually

plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient2(low="blue",
                        mid = "grey",
                        high="green",
                        midpoint = 0)+
  geom_point() +
  geom_label_repel()+  
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

# let's get more words on the plot
# also make the X axis clearer
# use darker colors

plotDat %>%
  ggplot(aes(x=score,y=freq,label=ngram,color=score)) +
  scale_color_gradient2(low="navyblue",
                        mid = "grey",
                        high="forestgreen",
                        midpoint = 0)+
  geom_vline(xintercept=0)+
  geom_point() +
  geom_label_repel(max.overlaps = 50)+  
  scale_x_continuous(limits = c(-.2,.1),
                     breaks = seq(-.2,.2,.05)) +
  scale_y_continuous(trans="log2",
                     breaks=c(.01,.05,.1,.2,.5,1,2,5))+
  theme_bw() +
  labs(x="Coefficient in Model",y="Uses per Review")+
  theme(legend.position = "none",
        axis.title=element_text(size=20),
        axis.text=element_text(size=16))

#################################################
#################################################
# Part 3 - glassdoor data
#################################################
#################################################

# new data! let's explore
glassdoor<-readRDS("data/glassdoorReviews.RDS")


# Only a few companies in this data
glassdoor %>%
  with(table(company))

# Split in categories - some big companies, some smaller ones
glassdoor %>%
  with(table(company,FAANG))

# Other important metadata - Overall rating
glassdoor %>%
  with(hist(overall))

# More exploring.... do companies differ by overall rating?
overall_avgs<-glassdoor %>%
  group_by(company) %>%
  summarize(m=mean(overall),
            se=sd(overall)/sqrt(n())) 

# note how we calculate a standard error above
# it is included through ymin and ymax on line 55

overall_avgs %>%
  ggplot(aes(x=company,color=company,
             y=m,ymin=m-se,ymax=m+se)) +
  geom_point() +
  geom_errorbar(width=.2) +
  theme_bw() +
  coord_flip() + # coord_flip makes the axis labels readable!
  scale_y_continuous(limits = c(3,5)) +
  labs(y="Overall Rating")+
  theme(legend.position="none")

# Let's explore the text.... 

# Note - there are two different text boxes!! "pros" and "cons"
glassdoor <- glassdoor%>%
  mutate(pros_wordcount=str_count(pros,"[[:alpha:]]+"),
         cons_wordcount=str_count(cons,"[[:alpha:]]+"))

# for showing a single continuous variable, we use a histogram
glassdoor %>%
  ggplot(aes(x=pros_wordcount)) +
  geom_histogram(bins = 100) +
  theme_bw() +
  xlim(0,100)

glassdoor %>%
  ggplot(aes(x=cons_wordcount)) +
  geom_histogram(bins = 100) +
  theme_bw() +
  xlim(0,100)

# Let's focus on people who actually wrote text in both boxes

gd_small<-glassdoor %>%
  filter(pros_wordcount>5 & cons_wordcount>5)

dim(gd_small)
# Even that's too big so let's get it down to 40,000 texts

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

# check the tuning to see if there is useful information
plot(amazon_model_pros)

##################################################################

# let's apply our model to two test sets

# We need the same X features in the test as in training 

# we use dfm_match() to make sure they are the same features

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

# check distributions - continuous predictor, continuous outcome
hist(amazon_test_predict_pros)
hist(amazon_test_Y)

# estimate accuracy - use kendall's tau
source("kendall_acc.R")

pros_acc<-kendall_acc(amazon_test_predict_pros,amazon_test_Y)

pros_acc

############################################

# Let's apply the same model to the cons text

dfm_amazon_test_cons<-TASSL_dfm(gd_amazon_test$cons,ngrams=1:2,
                                min.prop = 0)  %>%
  dfm_match(colnames(dfm_amazon_train_pros)) %>%
  convert(to="matrix")

# generate predictions for test data
amazon_test_predict_cons<-predict(amazon_model_pros,
                                  newx = dfm_amazon_test_cons)[,1]

hist(amazon_test_predict_cons)
hist(amazon_test_Y)

# estimate accuracy
cons_acc<-kendall_acc(amazon_test_predict_cons,amazon_test_Y)

# why is accuracy so low?
cons_acc

# Combine accuracy estimates for a plot
bind_rows(pros_acc %>%
            mutate(field="Pros ngrams"),
          cons_acc %>%
            mutate(field="Cons ngrams")) %>%
  ggplot(aes(x=field,color=field,
             y=acc,ymin=lower,ymax=upper)) +
  geom_point() +
  geom_errorbar(width=.4) +
  theme_bw() +
  labs(x="Test Data",y="Accuracy") +
  geom_hline(yintercept = 50) +
  theme(axis.text = element_text(size=24),
        axis.title = element_text(size=24),
        panel.grid=element_blank(),
        legend.position="none")

#################################################

