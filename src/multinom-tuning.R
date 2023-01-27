#' ---
#' title: "Multinomial regression"
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
#' Multinomial logistic regression is a type of generalized linear model (GLM) that is used to predict a categorical variable with more than two categories. It can be used to model nominal and ordinal variables and is an extension of binomial logistic regression, which is used to model binary outcomes.  Some of the advantages and disadvantages of multinomial regression include:
#' 
#' ### Advantages
#' 
#' - Useful to predict nominal or ordinal dependent variables with more than two categories.
#' - Able to test hypotheses about the relationship between the predictor variables and the dependent variable.
#' - Can be used to estimate the relative odds of different categories, which is useful for comparing the likelihood of different outcomes.
#' 
#' ### Disadvantages
#' 
#' - Sensitive to missing data, outliers, and multicollinearity, which can affect the estimated probabilities and coefficients.
#' - Assumes the observations are independent, which may not be the case in certain types of data such as time series data.
#' - Can have high variance if the response categories are highly imbalanced.
#' - Does not work well when the sample size is small in relation to the number of possible categories.
#' 
#' The main issue found with the NSL-KDD data set is that even though the sample data was significantly reduced by downsampling, the multinomial regression algorithm will still be slow compared to many other models.  However, the prediction metrics are quite good so that multinomial is a very useful model for intrusion detection.
#' 
#' ## Set-up
#'
#' The following set of libraries will be loaded to tune and fit a multinomial regression model on the NSL-KDD data set using `tidymodels`.  Specifically, the `glmnet` library is needed for the engine to fit the multinomial model. 

#+ set-up
source('src/scripts/do_packages.R')

libs <- c('glmnet', 'doParallel', 'tidymodels', 'finetune')
do_packages(libs)

kdd <- readRDS('data/interim/kdd.RDS')

options(tidymodels.dark = TRUE)

setwd("~/NSL-KDD")

#' ### Cross-validation folds
#'
#' With the data loaded in we can set cross-validation folds, which are subsets of data used in the process of cross validation. Cross validation will be used to assess the performance of the tuned models by training them on different subsets of the data and evaluating their performance on the remaining subset of data. The data will be split into 10 folds, with each fold representing a unique subset of the data. These folds are then used in a rotation to train and test the model, with each fold being used as the test set in turn. This allows for a more robust evaluation of the model's performance and ideally reduce bias, as it is tested on multiple subsets of the data rather than just one.

#+ cv-folds
kdd_train_ds_baked <- readRDS('data/interim/kdd_train_ds_baked.RDS')

set.seed(4960)
cv_folds <- vfold_cv(kdd_train_ds_baked,
                     v = 10)

#' ### Model specifications
#' 
#' Next the parameters to be tuned on, the engine to fit the model, and the mode will be set and translated for the multinomial regression algorithm in `tidymodels`.  The two parameters that will be tuned on are:
#' 
#' - `penalty` - the regularization parameter, &lambda;.
#' - `mixture` - the proportion of regularization penalty to be applied, &alpha;. A value of 1 specifies a lasso model, 0 a ridge regression model, and values between an elastic net model that interpolates lasso and ridge penalties.

#+ specs
multinom_spec <- multinom_reg(penalty = tune(),
                              mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification") %>%
  translate()

#' ### Model workflow
#' 
#' To set up the model workflow, a workflow object will be created and the model specifications and formula will be added. Additionally, the `glmnet` engine can handle case weights, so they will be added to the model.

#+ wf
multinom_wf <- workflow() %>%
  add_model(multinom_spec) %>%
  add_formula(Class ~ .) %>%
  add_case_weights(case_wts)

#' ### Tuning controls
#' 
#' The elements for the paramters that were set for tuning above will be extracted into a new object.  These will be used by the tuning algorithm to select viable values for those parameters.
#' 
#' A control argument for the tuning algorithm will also be set so that results are logged into the consol while the algorithm is running (although some of the verbage will be hidden due to the parallel back-end that will be set up prior to tuning). The tuning algorithm will also be set to end early if there are 10 consecutive iterations that do not find improvements in the model.
#' 
#' The area under the curve (AUC) of the Receiver Operating Characteristic (ROC), `roc_auc`, will be set as the metric to assess the fits of the tuned models. The AUC of the ROC is a measure of the model's classification performance. The AUC ranges between 0 and 1, with a value of 1 indicating perfect classification and a value of 0.5 indicating a classifier that is no better than random guessing. The AUC is a useful metric for the NSL-KDD data set because it is insensitive to the class distribution and is a good way to compare classifiers as it summarizes the trade-off between the true positive rate and the false positive rate in a single number.

#+ tune-controls
multinom_param <- multinom_spec %>%
  extract_parameter_set_dials()

bayes_control <- control_bayes(verbose = TRUE,
                               verbose_iter = TRUE,
                               no_improve = 10)

bayes_metrics <- metric_set(roc_auc)

#' ## Model fitting
#' 
#' ### Bayesian tuning
#' 
#' To tune the parameters of the multinomial model, a Bayesian optimization of the hyperparameters will be employed. Bayesian optimization is a method for global optimization of a function, it is particularly useful for optimizing the hyperparameters of a model because it can find the optimal set of parameters by iteratively exploring the parameter space. The `tune_bayes()` function uses a Gaussian process model to approximate the function that maps from the hyperparameter space to the performance metric, and it uses this model to guide the search for the optimal set of hyperparameters.
#' 
#' After setting a parallel back-end using all of the available cores, the Bayesian tuning algorithm will run over the cross-validation folds.  The `tune_bayes()` algorithm will first gather 5 initial results, then run up to 50 iterations.

#+ bayes-tune
n_cores <- parallel::detectCores()

cl <- makeForkCluster(n_cores)
registerDoParallel(cl)

set.seed(4960)
multinom_bayes <- multinom_wf %>%
  tune_bayes(resamples = cv_folds,
             iter = 50,
             param_info = multinom_param,
             metrics = bayes_metrics,
             initial = 5,
             control = bayes_control)

stopImplicitCluster()

#' ### Tuning results

autoplot(multinom_bayes)
show_best(multinom_bayes,
          metric = 'roc_auc')

#' ### Select and fit final model
#'
#' The model with the highest `roc_auc` will be selected as the best model and saved as an `R` object file for ensembling.

multinom_best_fit_params <- select_best(multinom_bayes,
                                        metric = 'roc_auc')

multinom_final_wf <- multinom_wf %>%
  finalize_workflow(multinom_best_fit_params)

multinom_final_fit <- multinom_final_wf %>%
  fit(kdd_train_ds_baked)

saveRDS(multinom_final_fit, 'models/tuning/multinom_fit.RDS')

#' After saving the model, the `R` environment will be cleaned followed by garbage collection to free unused memory.

rm(list = ls())
gc()