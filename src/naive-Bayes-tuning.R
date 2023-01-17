library(tidyverse)
library(tidymodels)
library(doParallel)
library(naivebayes)
library(discrim)
library(finetune)
library(butcher)
library(bundle)

options(tidymodels.dark = TRUE)

setwd("~/NSL-KDD")

kdd_train_baked <- readRDS('data/interim/kdd_train_baked.RDS')

# Set cross-validation folds
set.seed(4960)
cv_folds <- vfold_cv(kdd_train2_ds_baked,
                     v = 10)

nbay_spec <- naive_Bayes(Laplace = tune(),
                         smoothness = tune()) %>% 
  set_engine("naivebayes") %>% 
  set_mode("classification") %>%
  translate()

nbay_wf <- workflow() %>%
  add_model(nbay_spec) %>%
  add_formula(Class ~ .)

nbay_param <- nbay_spec %>% 
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
nbay_bayes <- nbay_wf %>%
  tune_bayes(resamples = cv_folds,
             iter = 50,
             param_info = nbay_param,
             metrics = bayes_metrics,
             initial = 5,
             control = bayes_control)

stopImplicitCluster()

autoplot(nbay_bayes)
show_best(nbay_bayes,
          metric = 'roc_auc')

nbay_bayes_bundled <- nbay_bayes%>%
  bundle()

saveRDS(nbay_bayes_bundled,
        'models/tuning/nbay_bayes_tune.RDS')


