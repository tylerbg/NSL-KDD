#' ---
#' title: "C5.0 rule-based model"
#' author: "Tyler B. Garner, tbgarner5023@psu.edu"
#' output:
#'   html_document:
#'     toc: true
#'     toc_float: true
#'     collapsed: false
#'     theme: united
#'     highlight: tango
#' ---

#+ setup, include = FALSE
knitr::opts_chunk$set(echo = TRUE,
                      eval = TRUE)
knitr::opts_knit$set(root.dir = '../')
options(width=100)

#' ## Introduction
#' 
#' C5.0 is a decision tree-based machine learning algorithm that uses a set of if-then rules to make predictions for classification tasks. It is an improved version of the C4.5 algorithm and is designed to handle large datasets and continuous as well as categorical variables. C5.0 generates a set of if-then rules that can be used to make predictions on new data. It is a rule-based model because it generates a set of rules that can be used to classify new data.
#' 
#' #### Advantages
#' - Relatively easy to interpret and understand, making them useful for explaining the reasoning behind a model's predictions.
#' - Handles both continuous and categorical variables, making them versatile for a wide range of data types.
#' - Relatively fast to train and use, making them suitable for large datasets.
#' 
#' ### Disadvantages
#' 
#' - Prone to overfitting, especially when the tree is deep and complex.
#' - Sensitive to small changes in the data, which can lead to instability in the model.
#' - May not work well with high-dimensional data, as the tree can become too complex to interpret.
#' - May not be able to capture complex relationships between variables, which can lead to lower accuracy.
#' - May not be as accurate as other machine learning algorithms such as Random Forest or Neural Networks in certain applications.
#' 
#' ## Set-up

#+ libs
source('src/scripts/do_packages.R')

libs <- c('rules', 'doParallel', 'tidymodels', 'finetune')
do_packages(libs)

options(tidymodels.dark = TRUE)

kdd_train_ds_baked <- readRDS('data/interim/kdd_train_ds_baked.RDS')

# Set cross-validation folds
set.seed(4960)
cv_folds <- vfold_cv(kdd_train_ds_baked,
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
  fit(kdd_train_ds_baked)

saveRDS(C5_final_fit, 'models/tuning/C5_fit.RDS')

rm(list = ls())
gc()