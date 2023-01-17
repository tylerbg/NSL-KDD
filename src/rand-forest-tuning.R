library(tidyverse)
library(tidymodels)
library(ranger)
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

# Use parallel processing within the models
n_cores <- parallel::detectCores()

rf_spec <- rand_forest(mtry = tune(),
                       min_n = 2,
                       trees = tune()) %>% 
  set_engine("ranger",
             num.threads = n_cores) %>% 
  set_mode("classification") %>%
  translate()

rf_wf <- workflow() %>%
  add_model(rf_spec) %>%
  add_formula(Class ~ .) %>%
  add_case_weights(case_wts)

rf_param <- rf_spec %>% 
  extract_parameter_set_dials() %>% 
  update(mtry = mtry(c(1, ncol(kdd_train2_ds_baked) - 2)))

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

set.seed(4960)
rf_bayes <- rf_wf %>%
  tune_bayes(resamples = cv_folds,
             iter = 50,
             param_info = rf_param,
             metrics = bayes_metrics,
             initial = 5,
             control = bayes_control)

autoplot(rf_bayes)
show_best(rf_bayes,
          metric = 'roc_auc')

rf_bayes_butchered <- rf_bayes %>%
  # butcher(verbose = TRUE) %>%
  bundle()

saveRDS(rf_bayes_butchered,
        'data/interim/rf_bayes_tune.RDS')