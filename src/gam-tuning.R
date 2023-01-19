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

gam_spec <- gam(select_features = tune(),
                adjust_deg_free = tune()) %>% 
  set_engine("mgcv") %>% 
  set_mode("classification") %>%
  translate()

gam_wf <- workflow() %>%
  add_model(gam_spec) %>%
  add_formula(Class ~ .) %>%
  add_case_weights(case_wts)

gam_param <- gam_spec %>% 
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
gam_bayes <- gam_wf %>%
  tune_bayes(resamples = cv_folds,
             iter = 50,
             param_info = gam_param,
             metrics = bayes_metrics,
             initial = 5,
             control = bayes_control)

stopImplicitCluster()

autoplot(gam_bayes)
show_best(gam_bayes,
          metric = 'roc_auc')

gam_best_fit_params <- select_best(gam_bayes,
                                    metric = 'roc_auc')

gam_final_wf <- gam_wf %>%
  finalize_workflow(gam_best_fit_params)

gam_final_fit <- gam_final_wf %>%
  fit(kdd_train2_ds_baked)

saveRDS(gam_final_fit, 'models/tuning/gam_fit.RDS')

rm(list = ls())