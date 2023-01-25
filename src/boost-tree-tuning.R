library(tidyverse)
library(tidymodels)
library(doParallel)
library(xgboost)
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

boost_spec <- boost_tree(tree_depth = tune(),
                         trees = tune(),
                         learn_rate = tune(),
                         mtry = tune(),
                         min_n = 2, # Low min_n predicts U2R better without affecting other classes
                         loss_reduction = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification") %>%
  translate()

# Set the workflow and include pre-specified case weights
boost_wf <- workflow() %>%
  add_model(boost_spec) %>%
  add_formula(Class ~ .) %>%
  add_case_weights(case_wts)

# Extract the parameters to be used for tuning
boost_param <- boost_spec %>% 
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(1, ncol(kdd_train2_ds_baked) - 2)))

bayes_control <- control_bayes(verbose = TRUE,
                               verbose_iter = TRUE,
                               no_improve = 10,
                               # Need both below = TRUE for stacking
                               save_pred = TRUE,
                               save_workflow = TRUE)

# bayes_control <- control_stacks_bayes()

bayes_metrics <- metric_set(roc_auc,
                            recall,
                            accuracy,
                            precision)

n_cores <- parallel::detectCores()

cl <- makeForkCluster(n_cores)
registerDoParallel(cl)

set.seed(6)
boost_bayes <- boost_wf %>%
  tune_bayes(resamples = cv_folds,
             iter = 10, # Set lower than other models to avoid errors in boost_bayes
             param_info = boost_param,
             metrics = bayes_metrics,
             initial = 6,
             control = bayes_control)

stopImplicitCluster()

autoplot(boost_bayes)
show_best(boost_bayes,
          metric = 'roc_auc')

boost_best_fit_params <- select_best(boost_bayes,
                                     metric = 'roc_auc')

boost_final_wf <- boost_wf %>%
  finalize_workflow(boost_best_fit_params)

boost_final_fit <- boost_final_wf %>%
  fit(kdd_train2_ds_baked)

saveRDS(boost_final_fit, 'models/tuning/boost_fit.RDS')

rm(list = ls())