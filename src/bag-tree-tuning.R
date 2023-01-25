library(tidyverse)
library(tidymodels)
library(doParallel)
library(rpart)
library(baguette)
library(finetune)

options(tidymodels.dark = TRUE)

kdd_train2_ds_baked <- readRDS('data/interim/kdd_train2_ds_baked.RDS')

# Set cross-validation folds
set.seed(4960)
cv_folds <- vfold_cv(kdd_train2_ds_baked,
                     v = 10)


bag_spec <- bag_tree(tree_depth = tune(),
                     min_n = 2,
                     cost_complexity = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("classification") %>%
  translate()

bag_wf <- workflow() %>%
  add_model(bag_spec) %>%
  add_formula(Class ~ .) %>%
  add_case_weights(case_wts)


# Extract the parameters to be used for tuning
bag_param <- bag_spec %>% 
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
bag_bayes <- bag_wf %>%
  tune_bayes(resamples = cv_folds,
             iter = 20,
             param_info = bag_param,
             metrics = bayes_metrics,
             initial = 5,
             control = bayes_control)

stopImplicitCluster()

autoplot(bag_bayes)
show_best(bag_bayes,
          metric = 'roc_auc')

bag_best_fit_params <- select_best(bag_bayes,
                                     metric = 'roc_auc')

bag_final_wf <- bag_wf %>%
  finalize_workflow(bag_best_fit_params)

bag_final_fit <- bag_final_wf %>%
  fit(kdd_train2_ds_baked)

saveRDS(bag_final_fit, 'models/tuning/bag_fit.RDS')

rm(list = ls())