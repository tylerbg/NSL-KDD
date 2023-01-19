library(tidyverse)
library(tidymodels)
library(doParallel)
library(glmnet)
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

multinom_spec <- multinom_reg(penalty = tune(),
                              mixture = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification") %>%
  translate()

multinom_wf <- workflow() %>%
  add_model(multinom_spec) %>%
  add_formula(Class ~ .) %>%
  add_case_weights(case_wts)

multinom_param <- multinom_spec %>% 
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
multinom_bayes <- multinom_wf %>%
  tune_bayes(resamples = cv_folds,
             iter = 50,
             param_info = multinom_param,
             metrics = bayes_metrics,
             initial = 5,
             control = bayes_control)

stopImplicitCluster()

autoplot(multinom_bayes)
show_best(multinom_bayes,
          metric = 'roc_auc')

multinom_best_fit_params <- select_best(multinom_bayes,
                                     metric = 'roc_auc')

multinom_final_wf <- multinom_wf %>%
  finalize_workflow(multinom_best_fit_params)

multinom_final_fit <- multinom_final_wf %>%
  fit(kdd_train2_ds_baked)

saveRDS(multinom_final_fit, 'models/tuning/multinom_fit.RDS')

rm(list = ls())