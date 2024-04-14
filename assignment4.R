#####################################################
#                                                   #       
#  Text Analysis for Social Scientists and Leaders  #
#                                                   #
#                  Assignment 4                     #
#                                                   #
#                                                   #
#####################################################



ecMain<-readRDS("earningsDat.RDS")

ecQA<-readRDS("earningsQandA.RDS") %>%
  mutate(wordcount=str_count(text,"[[:alpha:]]+"))

# filter the Q&A to match the filtering on the main dataset
ecQA<-ecQA %>%
  filter(callID%in%ecMain$callID)

# grab an example transcript from the Q&A data
ecQA %>%
  filter(callID=="100521") %>%
  as_tibble() %>%
  head(30) 

# how many questions per call are there?
ecQA %>%
  filter(asker==1) %>%
  group_by(callID) %>%
  summarize(qcount=n()) %>%
  ggplot(aes(x=qcount)) +
  geom_histogram()

# how many questions per questioner are there?
ecQA %>%
  filter(asker==1) %>%
  group_by(callID,askerID) %>%
  summarize(qcount=n()) %>%
  ggplot(aes(x=qcount)) +
  geom_histogram()


# how many questioners per call are there?
ecQA %>%
  filter(asker==1) %>%
  group_by(callID) %>%
  summarize(qcount=n_distinct(askerID)) %>%
  ggplot(aes(x=qcount)) +
  geom_histogram()


# askerID indicates the order of askers in each call
ecQA %>%
  filter(asker==1 & followup == 0) %>%
  with(table(as.numeric(askerID)))

# The question variable indicated the order of questions in each call
ecQA %>%
  with(table(question,asker))

####################################################################################
# calculate a turn-level feature and add it to the conversation-level data
####################################################################################

# Average word count of first questions in the Q&A data and join to the main data
ecMain_merged <- ecMain %>%
  left_join(ecQA %>%
              filter(asker==1 & question==1) %>%
              group_by(callID) %>%
              summarize_at(vars(wordcount),sum) %>%
            rename(wordcount_q1="wordcount"))

# compare average answer word count to earnings per share
kendall_acc(ecMain_merged$wordcount_q1,
            ecMain_merged$EPS_actual)

# Average word count of first answers in the Q&A data and join to the main data
ecMain_merged <- ecMain %>%
  left_join(ecQA %>%
              filter(asker==0 & question==1) %>%
              group_by(callID) %>%
              summarize_at(vars(wordcount),sum) %>%
              rename(wordcount_a1="wordcount")
            ) %>%
  mutate(wordcount_a1=replace_na(wordcount_a1,0))

# compare average answer word count to earnings per share
kendall_acc(ecMain_merged$wordcount_a1,
            ecMain_merged$EPS_actual)

