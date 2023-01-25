library(tidyverse)
library(tidymodels)
library(doParallel)
library(rules)
library(finetune)
library(butcher)
library(bundle)

options(tidymodels.dark = TRUE)

kdd_train2_ds_baked <- readRDS('data/interim/kdd_train2_ds_baked.RDS')

# Set cross-validation folds
set.seed(4960)
cv_folds <- vfold_cv(kdd_train2_ds_baked,
                     v = 10)

C5_spec <- C5_rules(trees = tune(),
                    min_n = 2) %>% 
  set_engine("C5.0") %>% 
  set_mode("classification") %>%
  translate()

C5_wf <- workflow() %>%
  add_model(C5_spec) %>%
  add_formula(Class ~ .) %>%
  add_case_weights(case_wts)

C5_param <- C5_spec %>% 
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

cl <- makePSOCKcluster(n_cores)
registerDoParallel(cl)

set.seed(4960)
C5_bayes <- C5_wf %>%
  tune_bayes(resamples = cv_folds,
             iter = 50,
             param_info = C5_param,
             metrics = bayes_metrics,
             initial = 5,
             control = bayes_control)

stopImplicitCluster()

autoplot(C5_bayes)
show_best(C5_bayes,
          metric = 'roc_auc')

C5_best_fit_params <- select_best(C5_bayes,
                                     metric = 'roc_auc')

C5_final_wf <- C5_wf %>%
  finalize_workflow(C5_best_fit_params)

C5_final_fit <- C5_final_wf %>%
  fit(kdd_train2_ds_baked)

saveRDS(C5_final_fit, 'models/tuning/C5_fit.RDS')

rm(list = ls())