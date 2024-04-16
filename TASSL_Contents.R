#####################################################
#                                                   #       
#  Text Analysis for Social Scientists and Leaders  #
#                                                   #
#                Table of Contents                  #
#                                                   #
#                                                   #
#####################################################


# Run this once for each package that isn't installed yet
# install.packages("tidyverse")

# Run these every time
library(tidyverse) # useful for almost everything
library(quanteda) # text analysis workhorse
library(textclean) # extra pre-processing
library(ggrepel) # for plots
library(glmnet) # Our estimation model
library(pROC)  # binary prediction accuracy
library(doc2concrete) # ngramTokens
library(sentimentr) # sentiment
library(spacyr) #  for grammar parsing
library(politeness) # dialogue acts
library(stm) # topic models
library(semgram) # motif analysis


# functions we use more than once have their own scripts
source("TASSL_dfm.R")
source("kendall_acc.R")

# extra tutorials - please complete on your own!
source("text_basics.R")
source("ggplot_tips.R")

# Initial Code
source("assignment1.R")
source("assignment2.R")
source("assignment3.R")
source("assignment4.R")

# Solutions - no peeking :)
source("assignment1_answers.R")
source("assignment2_answers.R")
source("assignment3_answers.R")
source("assignment4_answers.R")
