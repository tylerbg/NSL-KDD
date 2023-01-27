#' ---
#' title: "Bagged trees"
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
                      eval = FALSE)
knitr::opts_knit$set(root.dir = '../')
options(width=100)

#' ## Introduction
#' 
#' Bagged tree classification is an ensemble method that combines multiple decision tree models to improve the overall performance and reduce the variance of a single decision tree model. The method involves training multiple decision trees on different subsets of the training data, which are obtained by randomly sampling the data with replacement. The final prediction is made by averaging the predictions of all the individual trees. This can help to reduce overfitting and improve the generalization performance of the model.
#' 
#' Bagged tree models are typically used when the goal is to improve the performance and robustness of a decision tree model. They are often used in situations where there is a high degree of randomness or noise in the data, or where the goal is to reduce the variance of the model.
#' 
#' #### Advantages
#' 
#' - Can reduce the variance of a single decision tree model and improve the generalization performance by averaging the predictions of multiple decision trees.
#' - Less sensitive to noise and outliers in the data than a single decision tree model.
#' - Easy to implement and does not require any complex tuning of parameters.
#' 
#' #### Disadvantages
#' 
#' - Requires training multiple decision trees, which can be computationally expensive, especially with large datasets.
#' - Can be difficult to interpret the invidual contributions of each tree as the final predictions are made by averaging the predictions of multiple decision trees.
#' - Can only be used with decision tree models, which may not be the best choice for all types of data or problems.
#' 
#' 
#' 
#' ## Set-up
#'
#' The following set of libraries will be loaded to tune and fit a bagged tree classification model on the NSL-KDD data set using `tidymodels`.  Specifically, the `rpart` and `baguette` libraries are needed for the engine to fit the bagged tree classification model. 

#+ set-up
library(tidymodels)
library(doParallel)
library(rpart)
library(baguette)
library(finetune)

options(tidymodels.dark = TRUE)

setwd("~/NSL-KDD")

kdd_train_ds_baked <- readRDS('data/interim/kdd_train_ds_baked.RDS')

#' ### Cross-validation folds
#'
#' With the data loaded in we can set cross-validation folds, which are subsets of data used in the process of cross validation. Cross validation will be used to assess the performance of the tuned models by training them on different subsets of the data and evaluating their performance on the remaining subset of data. The data will be split into 10 folds, with each fold representing a unique subset of the data. These folds are then used in a rotation to train and test the model, with each fold being used as the test set in turn. This allows for a more robust evaluation of the model's performance and ideally reduce bias, as it is tested on multiple subsets of the data rather than just one.

#+ cv-folds
set.seed(4960)
cv_folds <- vfold_cv(kdd_train_ds_baked,
                     v = 10)

#' ### Model specifications
#' 
#' Next the parameters to be tuned on, the engine to fit the model, and the mode will be set and translated for the bagged tree algorithm in `tidymodels`.  The two parameters that will be tuned on are:
#' 
#' - `penalty` - the regularization parameter, &lambda;.
#' - `mixture` - the proportion of regularization penalty to be applied, &alpha;. A value of 1 specifies a lasso model, 0 a ridge regression model, and values between an elastic net model that interpolates lasso and ridge penalties.

bag_spec <- bag_tree(tree_depth = tune(),
                     min_n = 2,
                     cost_complexity = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("classification") %>%
  translate()

#' ### Model workflow
#' 
#' To set up the model workflow, a workflow object will be created and the model specifications and formula will be added. Additionally, the `glmnet` engine can handle case weights, so they will be added to the model.

bag_wf <- workflow() %>%
  add_model(bag_spec) %>%
  add_formula(Class ~ .) %>%
  add_case_weights(case_wts)

#' ### Tuning controls
#' 
#' The elements for the paramters that were set for tuning above will be extracted into a new object.  These will be used by the tuning algorithm to select viable values for those parameters.
#' 
#' A control argument for the tuning algorithm will also be set so that results are logged into the consol while the algorithm is running (although some of the verbage will be hidden due to the parallel back-end that will be set up prior to tuning). The tuning algorithm will also be set to end early if there are 10 consecutive iterations that do not find improvements in the model.
#' 
#' The area under the curve (AUC) of the Receiver Operating Characteristic (ROC), `roc_auc`, will be set as the metric to assess the fits of the tuned models. The AUC of the ROC is a measure of the model's classification performance. The AUC ranges between 0 and 1, with a value of 1 indicating perfect classification and a value of 0.5 indicating a classifier that is no better than random guessing. The AUC is a useful metric for the NSL-KDD data set because it is insensitive to the class distribution and is a good way to compare classifiers as it summarizes the trade-off between the true positive rate and the false positive rate in a single number.

bag_param <- bag_spec %>%
  extract_parameter_set_dials()

bayes_control <- control_bayes(verbose = TRUE,
                               verbose_iter = TRUE,
                               no_improve = 10)

bayes_metrics <- metric_set(roc_auc)

#' ## Model fitting
#' 
#' ### Bayesian tuning
#' 
#' To tune the parameters of the bagged tree model, a Bayesian optimization of the hyperparameters will be employed. Bayesian optimization is a method for global optimization of a function, it is particularly useful for optimizing the hyperparameters of a model because it can find the optimal set of parameters by iteratively exploring the parameter space. The `tune_bayes()` function uses a Gaussian process model to approximate the function that maps from the hyperparameter space to the performance metric, and it uses this model to guide the search for the optimal set of hyperparameters.
#' 
#' After setting a parallel back-end using all of the available cores, the Bayesian tuning algorithm will run over the cross-validation folds.  The `tune_bayes()` algorithm will first gather 5 initial results, then run up to 50 iterations.

n_cores <- parallel::detectCores()

cl <- makeForkCluster(n_cores)
registerDoParallel(cl)

set.seed(4960)
bag_bayes <- bayes_wf %>%
  tune_bayes(resamples = cv_folds,
             iter = 50,
             param_info = bag_param,
             metrics = bayes_metrics,
             initial = 5,
             control = bayes_control)

stopImplicitCluster()

#' ### Tuning results

autoplot(bayes_bayes)
show_best(bayes_bayes,
          metric = 'roc_auc')

#' ### Select and fit final model
#'
#' The model with the highest `roc_auc` will be selected as the best model and saved as an `R` object for ensembling.

bayes_best_fit_params <- select_best(bayes_bayes,
                                        metric = 'roc_auc')

bayes_final_wf <- bayes_wf %>%
  finalize_workflow(bayes_best_fit_params)

bayes_final_fit <- bayes_final_wf %>%
  fit(kdd_train_ds_baked)

saveRDS(bayes_final_fit, 'models/tuning/bayes_fit.RDS')

#' After saving the model, the `R` environment will be cleaned followed by garbage collection to free unused memory.

rm(list = ls())
gc()