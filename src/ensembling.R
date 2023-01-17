library(tidyverse)
library(tidymodels)
library(stacks)
library(doParallel)
library(butcher)
library(ranger)
library(xgboost)
library(baguette)
library(rules)
library(kknn)
library(naivebayes)

setwd("~/NSL-KDD")

bag_bayes <- readRDS('data/interim/bag_bayes_tune.RDS')
# boost_bayes <- readRDS('data/interim/boost_bayes_tune.RDS')
C5_bayes <- readRDS('data/interim/C5_bayes_tune.RDS')
# knn_bayes <- readRDS('data/interim/knn_bayes_tune.RDS')
# nbay_bayes <- readRDS('data/interim/nbay_bayes_tune.RDS')
rf_bayes <- readRDS('data/interim/rf_bayes_tune.RDS')


kdd_stacks <- stacks() %>%
  add_candidates(bag_bayes) %>%
  # add_candidates(boost_bayes) %>%
  add_candidates(C5_bayes)
  # add_candidates(knn_bayes) %>%
  # add_candidates(nbay_bayes) %>%
  # add_candidates(rf_bayes)

n_cores <- parallel::detectCores()

cl <- makeForkCluster(n_cores)
registerDoParallel(cl)

set.seed(4960)
kdd_ens <- blend_predictions(kdd_stacks,
                             control = control_grid(parallel_over = 'everything'))

stopImplicitCluster()

# Unregister parallel workers
#
# unregister_dopar <- function() {
#   env <- foreach:::.foreachGlobals
#   rm(list=ls(name=env), pos=env)
# }
# 
# unregister_dopar()

autoplot(kdd_ens)

autoplot(kdd_ens, "weights")

kdd_ens_fit <- fit_members(kdd_ens)

kdd_test_baked <- readRDS('data/interim/kdd_test_baked.RDS')

kdd_ens_pred <- predict(kdd_ens_fit,
                        kdd_test_baked)

test_metrics <- metric_set(roc_auc,
                           recall,
                           accuracy,
                           precision)

table(kdd_test_baked$Class,
           kdd_ens_pred$.pred_class)

recall_vec(kdd_test_baked$Class,
           kdd_ens_pred$.pred_class)


