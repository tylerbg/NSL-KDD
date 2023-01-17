library(tidyverse)
library(tidymodels)
library(doParallel)
library(kknn)
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

knn_spec <- nearest_neighbor(neighbors = tune(),
                             weight_func = tune(),
                             dist_power = tune()) %>% 
  set_engine("kknn") %>% 
  set_mode("classification") %>%
  translate()

knn_wf <- workflow() %>%
  add_model(knn_spec) %>%
  add_formula(Class ~ .)

knn_param <- knn_spec %>% 
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
knn_bayes <- knn_wf %>%
  tune_bayes(resamples = cv_folds,
             iter = 50,
             param_info = knn_param,
             metrics = bayes_metrics,
             initial = 5,
             control = bayes_control)

stopImplicitCluster()

autoplot(knn_bayes)
show_best(knn_bayes,
          metric = 'roc_auc')

knn_bayes_bundled <- knn_bayes %>%
  bundle()

saveRDS(knn_bayes_bundled,
        'models/tuning/knn_bayes_tune.RDS')


