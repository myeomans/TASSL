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


# ALWAYS clear big files out of the workspace to reduce memory load before closing RStudio
rm(vecSmall,wfFile)


