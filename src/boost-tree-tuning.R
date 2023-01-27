#' ---
#' title: "Boosted decision trees"
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
#' Boosted Decision Trees are a type of ensemble learning method in which multiple decision trees are combined to create a more accurate and robust model. The basic idea behind boosting is to train weak models, such as shallow decision trees, iteratively and combine their predictions to create a strong model. The weak models are trained on subsets of the data, with the subsets chosen in a way that focuses on the examples that the previous weak models had difficulty with. This process is repeated until a desired level of accuracy is reached.
#' 
#' ### Advantages
#' 
#' - Can significantly improve the accuracy of the model by combining multiple weak models into a strong one.
#' - Handles large and complex datasets with high dimensionality, and are particularly useful when the data is noisy or has missing values.
#' - Captures non-linear relationships in the data, making it useful for a wide range of applications.
#' - Well-suited for handling imbalanced datasets where the classes are not represented equally.
#' 
#' ### Disadvantages
#' 
#' - Computationally expensive, particularly when training large models on large datasets.
#' - Can lead to overfitting if the number of iterations is too high or if the weak models are too complex.
#' - Lacks interpretability as the final model is a combination of many weak models, making it hard to understand how the model is making its predictions.
#' - Requires fine tuning of the hyperparameters for optimal results.
#' 
#' ## Set-up

#+ libs
source('src/scripts/do_packages.R')

libs <- c('xgboost', 'doParallel', 'tidymodels', 'finetune')
do_packages(libs)

options(tidymodels.dark = TRUE)

kdd_train_ds_baked <- readRDS('data/interim/kdd_train_ds_baked.RDS')


# Set cross-validation folds
set.seed(4960)
cv_folds <- vfold_cv(kdd_train_ds_baked,
                     v = 10)

boost_spec <- boost_tree(tree_depth = tune(),
                         trees = tune(),
                         learn_rate = tune(),
                         mtry = tune(),
                         min_n = 2, # Low min_n predicts U2R better without affecting other classes
                         loss_reduction = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification") %>%
  translate()

# Set the workflow and include pre-specified case weights
boost_wf <- workflow() %>%
  add_model(boost_spec) %>%
  add_formula(Class ~ .) %>%
  add_case_weights(case_wts)

# Extract the parameters to be used for tuning
boost_param <- boost_spec %>% 
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(1, ncol(kdd_train_ds_baked) - 2)))

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

n_cores <- parallel::detectCores()

cl <- makeForkCluster(n_cores)
registerDoParallel(cl)

set.seed(6)
boost_bayes <- boost_wf %>%
  tune_bayes(resamples = cv_folds,
             iter = 10, # Set lower than other models to avoid errors in boost_bayes
             param_info = boost_param,
             metrics = bayes_metrics,
             initial = 6,
             control = bayes_control)

stopImplicitCluster()

autoplot(boost_bayes)
show_best(boost_bayes,
          metric = 'roc_auc')

boost_best_fit_params <- select_best(boost_bayes,
                                     metric = 'roc_auc')

boost_final_wf <- boost_wf %>%
  finalize_workflow(boost_best_fit_params)

boost_final_fit <- boost_final_wf %>%
  fit(kdd_train_ds_baked)

saveRDS(boost_final_fit, 'models/tuning/boost_fit.RDS')

rm(list = ls())