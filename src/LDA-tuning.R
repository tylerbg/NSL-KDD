library(tidyverse)
library(tidymodels)
library(doParallel)
library(mda)
library(discrim)
library(finetune)
library(butcher)
library(bundle)

options(tidymodels.dark = TRUE)

setwd("~/NSL-KDD")

kdd_train2_ds_baked <- readRDS('data/interim/kdd_train2_ds_baked.RDS')

# Set cross-validation folds
set.seed(4960)
cv_folds <- vfold_cv(kdd_train2_ds_baked,
                     v = 10)

lda_spec <- discrim_linear(penalty = tune()) %>% 
  set_engine("mda") %>% 
  set_mode("classification") %>%
  translate()

lda_wf <- workflow() %>%
  add_model(lda_spec) %>%
  add_formula(Class ~ .) %>%
  add_case_weights(case_wts)

lda_param <- lda_spec %>% 
  extract_parameter_set_dials()

bayes_control <- control_bayes(verbose = TRUE,
                               verbose_iter = TRUE,
                               no_improve = 10,
                               # Need both below = TRUE for stacking
                               save_pred = TRUE,
                               save_workflow = TRUE,
                               parallel_over = 'everything')

bayes_metrics <- metric_set(roc_auc,
                            recall,
                            accuracy,
                            precision)

n_cores <- parallel::detectCores()

cl <- makeForkCluster(n_cores)
registerDoParallel(cl)

set.seed(4960)
lda_bayes <- lda_wf %>%
  tune_bayes(resamples = cv_folds,
             iter = 50,
             param_info = lda_param,
             metrics = bayes_metrics,
             initial = 5,
             control = bayes_control)

stopImplicitCluster()

autoplot(lda_bayes)
show_best(lda_bayes,
          metric = 'roc_auc')

lda_best_fit_params <- select_best(lda_bayes,
                                     metric = 'roc_auc')

lda_final_wf <- lda_wf %>%
  finalize_workflow(lda_best_fit_params)

lda_final_fit <- lda_final_wf %>%
  fit(kdd_train2_ds_baked)

saveRDS(lda_final_fit, 'models/tuning/lda_fit.RDS')

rm(list = ls())