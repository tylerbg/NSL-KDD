#' ---
#' title: "Naive Bayes"
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
#' Naive Bayes is a statistical method of classification based on Bayes theorem and is used for binary and multiclass classification problems. It assumes independence between the features, meaning that the presence of a particular feature in a class is unrelated to the presence of any other feature. This allows for efficient computation of probabilities and class predictions, making Naive Bayes a fast and simple algorithm.
#' 
#' ### Advantages
#' 
#' - Simple and easy to implement.
#' - Works well with a large dataset.
#' - High accuracy for binary and multiclass classification problems.
#' - Can handle irrelevant or partially missing data.
#' - Fast prediction times.
#' 
#' ### Disadvantages
#' 
#' - Assumes independence between features, which is not always accurate.
#' - Can be biased towards features with many outcomes.
#' - Performs poorly with a small dataset.
#' - Can produce incorrect results if a class has a very low probability.
#' - Sensitive to irrelevant features, so feature selection is important.
#' 
#' For the NSL-KDD dataset specifically, while the naive Bayes method is fast with handling large datasets, the assumption of independence between features does not hold as many of the selected variables from feature selection have the same origins (are splines and/or interaction terms of a handful of variables). Additionally the significantly imbalanced nature of the attack types can cause issues with predictions. Regardless, due to the high speed of the algorithm it is worth testing.
#' 
#' ## Set-up
#' First, all of the `R` packages required for the following code will be installed if not already installed and loaded. The `naivebayes` and `discrim` packages have the "naivebayes" engine that is required to fit the boosted tree model with `tidymodels`. The `doParallel` package will provide tools to create a parallel backend and the `finetune` package provides tools for model tuning.

#+ libs
source('src/scripts/do_packages.R')

libs <- c('naivebayes', 'discrim', 'doParallel', 'tidymodels', 'finetune')
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
#' Next the parameters to be tuned on, the engine that will fit the model, and the model type will be set and translated for the naive Bayes algorithm in `tidymodels`.  The two parameters that will be tuned on are:
#' 
#' - `Laplace` -  non-negative number representing the the relative smoothness of the class boundary. Smaller values result in more flexible boundaries while larger values generate class boundaries that are less adaptable.
#' - `smoothness` - number of trees contained in the ensemble.

#+ specs
nbay_spec <- naive_Bayes(Laplace = tune(),
                         smoothness = tune()) %>% 
  set_engine("naivebayes") %>% 
  set_mode("classification") %>%
  translate()

#' ### Model workflow
#' 
#' To set up the model workflow, a workflow object will be created and the model specifications and formula will be added. The *naivebayes* engine does not handle case weights, so they will be excluded..

#+ wf
nbay_wf <- workflow() %>%
  add_model(nbay_spec) %>%
  add_formula(Class ~ .)

#' ### Tuning controls
#' 
#' The elements for the two parameters that were set for tuning above will be extracted into a new object.  This will be used by the tuning algorithm to select viable values for those parameters.
#' 
#' A control argument for the tuning algorithm will also be set so that results are logged into the console while the algorithm is running (although some of the verbage will be hidden due to the parallel back-end that will be set up prior to tuning). The tuning algorithm will also be set to end early if there are 10 consecutive iterations that do not find improvements in the model.
#' 
#' The area under the curve (AUC) of the Receiver Operating Characteristic (ROC), `roc_auc`, will be set as the metric to assess the fits of the tuned models. The ROC-AUC is a measure of the model's classification performance. The AUC ranges between 0 and 1, with a value of 1 indicating perfect classification and a value of 0.5 indicating a classifier that is no better than random guessing. The ROC-AUC is a useful metric for the NSL-KDD data set because it is insensitive to the class distribution and is a good way to compare classifiers as it summarizes the trade-off between the true positive rate and the false positive rate in a single number.
#' 
#' Three other metrics will also be collected to assess the tuned models. These are:
#' 
#' - **Accuracy** - The proportion of correct predictions made by the model out of all the predictions made. It is calculated by dividing the number of correct predictions by the total number of predictions.
#' - **Precision** - The proportion of true positive predictions made by the model out of all positive predictions made by the model. It is important to note that precision is sensitive to the number of false positives, so a model with high precision will have a low false positive rate. In the context of the NSL-KDD dataset, the precision will go down as more *Normal* connections are classified as one of the intrusion types.
#' - **Recall** - The proportion of true positive predictions made by the model out of all actual positive instances. Also known as sensitivity or true positive rate. It is important to note that recall is sensitive to the number of false negatives, so a model with high recall will have a low false negative rate. In the context of the NSL-KDD dataset, the recall will go down as more attack connections are classified as *Normal*.
#' 
#' Note that the precision and recall can also be affected by other mis-classifications among the attack types. For example, a *DoS* attack predicted to be a *Probe* connection will reduce the overall model precision.

#+ tune-controls
nbay_param <- nbay_spec %>% 
  extract_parameter_set_dials()

bayes_control <- control_bayes(verbose = TRUE,
                               verbose_iter = TRUE,
                               no_improve = 10)

bayes_metrics <- metric_set(roc_auc,
                            accuracy,
                            precision,
                            recall)

#' ## Tuning
#' 
#' To tune the parameters of the boosted decision tree model, a Bayesian optimization of the parameters will be employed. Bayesian optimization is a method for global optimization of a function, it is particularly useful for optimizing the parameters of a model because it can find the optimal set of parameters by iteratively exploring the parameter space. The `tune_bayes()` function uses a Gaussian process model to approximate the function that maps from the parameters space to the performance metric, and it uses this model to guide the search for the optimal set of parameters
#' 
#' After setting a parallel back-end using all of the available cores, the Bayesian tuning algorithm will run over the CV folds.  The `tune_bayes()` algorithm will first gather 5 initial results, then run up to a maximum of 50 iterations. The tuning algorithm is set to end early if there is no improvement in the ROC-AUC score after 10 iterations.

#+ bayes-tune
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

#' ### Tuning assessment
#' 
#' To assess the tuning procedure, a plot of the tuning results for the hyperparameters and the top models by each of the four metrics, ROC AUC, accuracy, precision, and recall, will be printed.

#+ tune-assess
png('results/figures/nbay-tuning-results.png')
autoplot(nbay_bayes) +
  theme_bw()
dev.off()

lapply(c('roc_auc', 'accuracy', 'recall', 'precision'),
       function (x) show_best(nbay_bayes,
                              metric = x))

boost_best_fit_params <- select_best(nbay_bayes,
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
nbay_final_wf <- nbay_wf %>%
  finalize_workflow(nbay_best_fit_params)

nbay_final_fit <- nbay_final_wf %>%
  fit(kdd_train2_ds_baked)

saveRDS(nbay_final_fit, 'models/tuning/nbay_fit.RDS')

rm(list = ls())
gc()
