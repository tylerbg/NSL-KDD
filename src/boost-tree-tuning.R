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
#' First, all of the `R` packages required for the following code will be installed if not already installed and loaded. The `xgboost` package has the "xgboost" engine that is required to fit the boosted tree model with `tidymodels`. The `doParallel` package will provide tools to create a parallel backend and the `finetune` package provides tools for model tuning.

#+ libs
source('src/scripts/do_packages.R')

libs <- c('xgboost', 'doParallel', 'tidymodels', 'finetune')
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
#' - `mtry` - number of predictors that will be randomly sampled at each split.
#' - `trees` - number of trees contained in the ensemble.
#' - `min_n` -  the minimum number of data points in a node required for further splitting.
#' - `tree_depth` - the maximum depth of the tree.
#' - `learn_rate` - shrinkage parameter, or the rate at which the boosting algorithm adapts from iteration-to-iteration.
#' - `loss_reduction` - reduction in the loss function required to split further .

#+ specs
boost_spec <- boost_tree(mtry = tune(),
                         trees = tune(),
                         min_n = tune(),
                         tree_depth = tune(),
                         learn_rate = tune(),
                         loss_reduction = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification") %>%
  translate()

#' ### Model workflow
#' 
#' To set up the model workflow, a workflow object will be created and the model specifications and formula will be added. Additionally, the "xgboost" engine can handle case weights, so they will be added to the model.

#+ wf
boost_wf <- workflow() %>%
  add_model(boost_spec) %>%
  add_formula(Class ~ .) %>%
  add_case_weights(case_wts)

#' ### Tuning controls
#' 
#' The elements for the hyperparameters that were set for tuning above will be extracted into a new object.  These will be used by the tuning algorithm to select viable values for those parameters. The `mtry` hyperparameter must be updated to have a maximum value, which will be set to the total number of predictor variables.
#' 
#' A control argument for the tuning algorithm will also be set so that results are logged into the console while the algorithm is running (although some of the verbage will be hidden due to the parallel back-end that will be set up prior to tuning). The tuning algorithm will also be set to end early if there are 10 consecutive iterations that do not find improvements in the model.
#' 
#' The area under the curve (AUC) of the Receiver Operating Characteristic (ROC), `roc_auc`, will be set as the metric to assess the fits of the tuned models. The AUC of the ROC is a measure of the model's classification performance. The AUC ranges between 0 and 1, with a value of 1 indicating perfect classification and a value of 0.5 indicating a classifier that is no better than random guessing. The AUC is a useful metric for the NSL-KDD data set because it is insensitive to the class distribution and is a good way to compare classifiers as it summarizes the trade-off between the true positive rate and the false positive rate in a single number.
#' 
#' Three other metrics will also be collected to assess after tuning. These are:
#' 
#' - **Accuracy** - The proportion of correct predictions made by the model out of all the predictions made. It is calculated by dividing the number of correct predictions by the total number of predictions.
#' - **Precision** - The proportion of true positive predictions made by the model out of all positive predictions made by the model. It is important to note that precision is sensitive to the number of false positives, so a model with high precision will have a low false positive rate.
#' - **Recall** - The proportion of true positive predictions made by the model out of all actual positive instances. Also known as sensitivity or true positive rate. It is important to note that recall is sensitive to the number of false negatives, so a model with high recall will have a low false negative rate.

#+ tune-controls
boost_param <- boost_spec %>% 
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(1, ncol(kdd_train_ds_baked) - 2)))

bayes_control <- control_bayes(verbose = TRUE,
                               verbose_iter = TRUE,
                               no_improve = 10)

bayes_metrics <- metric_set(roc_auc,
                            recall,
                            accuracy,
                            precision)

#' ## Modeling
#' 
#' ### Bayesian tuning
#' 
#' To tune the parameters of the boosted decision tree model, a Bayesian optimization of the hyperparameters will be employed. Bayesian optimization is a method for global optimization of a function, it is particularly useful for optimizing the hyperparameters of a model because it can find the optimal set of parameters by iteratively exploring the parameter space. The `tune_bayes()` function uses a Gaussian process model to approximate the function that maps from the hyperparameter space to the performance metric, and it uses this model to guide the search for the optimal set of hyperparameters.
#' 
#' After setting a parallel back-end using all of the available cores, the Bayesian tuning algorithm will run over the cross-validation folds.  The `tune_bayes()` algorithm will first gather 7 initial results, then run up to 50 iterations. More initial results are gathered for this model compared to other models as there are more hyperparameters being tuned. 

#+ bayes-tune
n_cores <- parallel::detectCores()

cl <- makeForkCluster(n_cores)
registerDoParallel(cl)

set.seed(6)
boost_bayes <- boost_wf %>%
  tune_bayes(resamples = cv_folds,
             iter = 10, # Set lower than other models to avoid errors in boost_bayes
             param_info = boost_param,
             metrics = bayes_metrics,
             initial = 7,
             control = bayes_control)

stopImplicitCluster()

#' ### Tuning assessment
#' 
#' To assess the tuning procedure, a plot of the tuning results for the hyperparameters and the top models by each of the four parameters, ROC AUC, accuracy, recall, and precision, will be printed.

#+ tune-assess
png('results/figures/boost-tree-tuning-results.png')
autoplot(boost_bayes) +
  theme_bw()
dev.off()

show_best(boost_bayes,
          metric = 'roc_auc')

lapply(c('roc_auc', 'accuracy', 'recall', 'precision'),
       function (x) show_best(boost_bayes,
                              metric = x))

boost_best_fit_params <- select_best(boost_bayes,
                                     metric = 'precision')

#' In the plot, there is a fairly large range of how well some of the hyperparameter sets perform. Overall, the best model as defined by the ROC AUC, *Preprocessor1_Model7*, has a mean value = 0.998. It also ranks high for accuracy and recall, but is not in the top 5 of precision. A second model, *Preprocessor1_Model3*, ranks in the top 5 of all four metrics, including having the highest precision while being on par with the other metrics. Therefore, it will be selected for the ensemble model.
#'
#' ## Final fit
#' 
#' The selected model had the hyperparameters:
#' 
#' - `mtry` = 54
#' - `trees` = 1520
#' - `min_n` = 29
#' - `tree_depth` = 9
#' - `learn_rate` = 0.0805
#' - `loss_reduction` = 0.217
#' 
#'  In the next chunk this model will be fit and saved as an `R` object file for ensembling. The `R` environment will then be cleaned.

#+ fit-final
boost_final_wf <- boost_wf %>%
  finalize_workflow(boost_best_fit_params)

boost_final_fit <- boost_final_wf %>%
  fit(kdd_train_ds_baked)

saveRDS(boost_final_fit, 'models/tuning/boost_fit.RDS')

rm(list = ls())
gc()
