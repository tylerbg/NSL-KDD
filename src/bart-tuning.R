library(tidyverse)
library(tidymodels)
library(dbarts)
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

bart_spec <- parsnip::bart(trees = tune(),
                         prior_terminal_node_coef = tune(),
                         prior_terminal_node_expo = tune(),
                         prior_outcome_range = tune()) %>% 
  set_engine("dbarts",
             nthread = n_cores) %>% 
  set_mode("classification") %>%
  translate()

bart_wf <- workflow() %>%
  add_model(bart_spec) %>%
  add_formula(Class ~ .)

bart_param <- bart_spec %>% 
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

set.seed(4960)
bart_bayes <- bart_wf %>%
  tune_bayes(resamples = cv_folds,
             iter = 50,
             param_info = bart_param,
             metrics = bayes_metrics,
             initial = 5,
             control = bayes_control)

autoplot(bart_bayes)
show_best(bart_bayes,
          metric = 'roc_auc')

bart_bayes_butchered <- bart_bayes %>%
  # butcher(verbose = TRUE) %>%
  bundle()

saveRDS(bart_bayes_butchered,
        'data/interim/bart_bayes_tune.RDS')


