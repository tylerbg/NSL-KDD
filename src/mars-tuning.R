library(tidyverse)
library(tidymodels)
library(doParallel)
library(finetune)
library(bundle)

options(tidymodels.dark = TRUE)

kdd_train2_ds_baked <- readRDS('data/interim/kdd_train2_ds_baked.RDS')

# Set cross-validation folds
set.seed(4960)
cv_folds <- vfold_cv(kdd_train2_ds_baked,
                     v = 10)

mars_spec <- mars(num_terms = tune(),
                  prod_degree = tune(),
                  prune_method = tune()) %>% 
  set_engine("earth") %>% 
  set_mode("classification") %>%
  translate()

mars_wf <- workflow() %>%
  add_model(mars_spec) %>%
  add_formula(Class ~ .) %>%
  add_case_weights(case_wts)

mars_param <- mars_spec %>% 
  extract_parameter_set_dials()

bayes_control <- control_bayes(verbose = TRUE,
                               verbose_iter = TRUE,
                               no_improve = 10,
                               # Need both below = TRUE for stacking
                               save_pred = TRUE,
                               save_workflow = TRUE)

bayes_metrics <- metric_set(roc_auc,
                            recall,
                            accuracy,
                            precision)

n_cores <- parallel::detectCores()

cl <- makePSOCKcluster(n_cores)
registerDoParallel(cl)

set.seed(4960)
mars_bayes <- mars_wf %>%
  tune_bayes(resamples = cv_folds,
             iter = 50,
             param_info = mars_param,
             metrics = bayes_metrics,
             initial = 5,
             control = bayes_control)

stopImplicitCluster()

autoplot(mars_bayes)
show_best(mars_bayes,
          metric = 'roc_auc')

mars_best_fit_params <- select_best(mars_bayes,
                                  metric = 'roc_auc')

mars_final_wf <- mars_wf %>%
  finalize_workflow(mars_best_fit_params)

mars_final_fit <- mars_final_wf %>%
  fit(kdd_train2_ds_baked)

saveRDS(mars_final_fit, 'models/tuning/mars_fit.RDS')

rm(list = ls())