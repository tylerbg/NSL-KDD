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
                      eval = FALSE)
knitr::opts_knit$set(root.dir = '../')
options(width=100)

#' ## Introduction
#' 
#' C5.0 is a decision tree-based machine learning algorithm that uses a set of if-then rules to make predictions for classification tasks. It is an improved version of the C4.5 algorithm and is designed to handle large datasets and continuous as well as categorical variables. C5.0 generates a set of if-then rules that can be used to make predictions on new data. It is a rule-based model because it generates a set of rules that can be used to classify new data.
#' 
#' ### Advantages
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
#' First, all of the `R` packages required for the following code will be installed if not already installed and loaded. The `rules` package has the "C5.0" engine that is required to fit the C5.0 rules fit model with `tidymodels`. The `doParallel` package will provide tools to create a parallel backend and the `finetune` package provides tools for model tuning.

#+ libs
source('src/scripts/do_packages.R')

libs <- c('rules', 'doParallel', 'tidymodels', 'finetune')
do_packages(libs)

options(tidymodels.dark = TRUE)

#' ### Cross-validation folds
#'
#' Using the training set generated in 'feat-eng.R', cross-validation (CV) folds will be set. CV folds are subsets of data used to assess the performance of the tuned models by training them on different subsets of the data and evaluating their performance on the remaining subset of data. The data will be split into 10 folds, with each fold representing a unique subset of the data. These folds are then used in a rotation to train and test the model, with each fold being used as the test set in turn. This allows for a more robust evaluation of the model's performance and ideally reduce bias, as it is tested on multiple subsets of the data rather than just one.

#+ cv-folds
kdd_train_ds_baked <- readRDS('data/interim/kdd_train_ds_baked.RDS')

set.seed(4960)
cv_folds <- vfold_cv(kdd_train_ds_baked,
                     v = 10)

#' ### Model specifications
#' 
#' Next the parameters to be tuned on, the engine to fit the model, and the mode will be set and translated for the multinomial regression algorithm in `tidymodels`.  The two parameters that will be tuned on are:
#' 
#' - `trees` - the number of trees to ensemble.
#' - `min_n` -  the minimum number of data points in a node required for further splitting.

#+ specs
C5_spec <- C5_rules(trees = tune(),
                    min_n = tune()) %>% 
  set_engine("C5.0") %>% 
  set_mode("classification") %>%
  translate()

#' ### Model workflow
#' 
#' To set up the model workflow, a workflow object will be created and the model specifications and formula will be added. Additionally, the "C5.0" engine can handle case weights, so they will be added to the model.

#+ wf
C5_wf <- workflow() %>%
  add_model(C5_spec) %>%
  add_formula(Class ~ .) %>%
  add_case_weights(case_wts)

#' ### Tuning controls
#' 
#' The elements for the paramters that were set for tuning above will be extracted into a new object.  These will be used by the tuning algorithm to select viable values for those parameters.
#' 
#' A control argument for the tuning algorithm will also be set so that results are logged into the console while the algorithm is running (although some of the verbage will be hidden due to the parallel back-end that will be set up prior to tuning). The tuning algorithm will also be set to end early if there are 10 consecutive iterations that do not find improvements in the model.
#' 
#' The area under the curve (AUC) of the Receiver Operating Characteristic (ROC), `roc_auc`, will be set as the metric to assess the fits of the tuned models. The AUC of the ROC is a measure of the model's classification performance. The AUC ranges between 0 and 1, with a value of 1 indicating perfect classification and a value of 0.5 indicating a classifier that is no better than random guessing. The AUC is a useful metric for the NSL-KDD data set because it is insensitive to the class distribution and is a good way to compare classifiers as it summarizes the trade-off between the true positive rate and the false positive rate in a single number.

#+ tune-controls
C5_param <- C5_spec %>% 
  extract_parameter_set_dials()

bayes_control <- control_bayes(verbose = TRUE,
                               verbose_iter = TRUE,
                               no_improve = 10)

bayes_metrics <- metric_set(roc_auc,
                            accuracy,
                            recall,
                            precision)

#' ## Modeling
#' 
#' ### Bayesian tuning
#' 
#' To tune the parameters of the multinomial model, a Bayesian optimization of the hyperparameters will be employed. Bayesian optimization is a method for global optimization of a function, it is particularly useful for optimizing the hyperparameters of a model because it can find the optimal set of parameters by iteratively exploring the parameter space. The `tune_bayes()` function uses a Gaussian process model to approximate the function that maps from the hyperparameter space to the performance metric, and it uses this model to guide the search for the optimal set of hyperparameters.
#' 
#' After setting a parallel back-end using all of the available cores, the Bayesian tuning algorithm will run over the cross-validation folds.  The `tune_bayes()` algorithm will first gather 5 initial results, then run up to 50 iterations.

#+ bayes-tune
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

#' ### Tuning assessment
#' 
#' To assess the tuning procedure, a plot of the tuning results for the hyperparameters and the top models by ROC AUC will be printed.

#+ tune-assess
png('results/figures/C5-tuning-results.png')
autoplot(C5_bayes) +
  theme_bw()
dev.off()

show_best(C5_bayes,
          metric = 'roc_auc')

#' In the plot, each of the metrics were all reasonably high overall. Notably, the model with the highest ROC AUC was built during the initial hyperparameter optimization, with a mean of 0.993. It may be useful to reduce the number of iterations without improvement to increase tuning speed.
#'
#' ## Final fit
#' 
#' The final model used 42 trees with a node size of 26 and will be fit and saved as an `R` object file for ensembling. The `R` environment will then be cleaned.

#+ fit-final
C5_best_fit_params <- select_best(C5_bayes,
                                     metric = 'roc_auc')

C5_final_wf <- C5_wf %>%
  finalize_workflow(C5_best_fit_params)

C5_final_fit <- C5_final_wf %>%
  fit(kdd_train_ds_baked)

saveRDS(C5_final_fit, 'models/tuning/C5_fit.RDS')

rm(list = ls())
gc()