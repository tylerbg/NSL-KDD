library(tidyverse)
library(tidymodels)
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

# Use parallel processing within the models
n_cores <- parallel::detectCores()

# To speed up tuning and avoid errors with boost_tree, use some params selected with rand_forest
boost_spec <- boost_tree(tree_depth = tune(),
                         trees = 1874, # Selected from best param in rand_forest
                         learn_rate = tune(),
                         mtry = 25, # Selected from best param in rand_forest
                         min_n = 2, # Low min_n predicts U2R better without affecting other classes
                         loss_reduction = tune()) %>% 
  set_engine("xgboost",
             nthread = n_cores) %>% 
  set_mode("classification") %>%
  translate()

# Set the workflow and include pre-specified case weights
boost_wf <- workflow() %>%
  add_model(boost_spec) %>%
  add_formula(Class ~ .) %>%
  add_case_weights(case_wts)

# Extract the parameters to be used for tuning
boost_param <- boost_spec %>% 
  extract_parameter_set_dials()

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

set.seed(6)
boost_bayes <- boost_wf %>%
  tune_bayes(resamples = cv_folds,
             iter = 10, # Set lower than other models to avoid errors in boost_bayes
             param_info = boost_param,
             metrics = bayes_metrics,
             initial = 5,
             control = bayes_control)

autoplot(boost_bayes)
show_best(boost_bayes,
          metric = 'roc_auc')

boost_bayes_butchered <- boost_bayes %>%
  # butcher(verbose = TRUE) %>%
  bundle()

saveRDS(boost_bayes_butchered,
        'data/interim/boost_bayes_tune.RDS')

