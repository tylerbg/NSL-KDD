#' ---
#' title: "Feature engineering"
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
                      eval = TRUE,
                      messages = FALSE,
                      warnings = FALSE)
knitr::opts_knit$set(root.dir = '../')
options(width=100)

#' ## Introduction
#' 
#' Feature engineering is the process of creating new features or transforming existing features from raw data to improve the performance of machine learning models. It typically involves selecting, combining, and transforming variables to create new features that better represent the underlying problem and increase the model's ability to learn from the data. This can include techniques like one-hot encoding, normalization, and feature scaling.
#' 
#' Due to the large size of the NSL-KDD dataset and the significant imbalance of connection class types, downsampling and case weights will be used to increase the speed and accuracy of the models. Multiple feature engineering approaches will then be employed which include centering and scaling, creating dummy variables, creating splines, and adding interaction terms. These methods will greatly increase the total number of variables in the dataset, so a random forest will be used to rank variables by importance, and a subset of 'important' variables will be selected for modeling.
#' 
#' ## Set-up
#' 
#' First, all of the `R` packages required for the following code will be installed if not already installed and loaded. The NSL-KDD `R` object created in the 'set-up.R' source file will be then be loaded.

#+ libs
source('src/scripts/do_packages.R')

libs <- c('xgboost', 'tidyverse', 'tidymodels', 'vip')
do_packages(libs)

kdd <- readRDS('data/interim/kdd.RDS')

#' ### Split data
#'
#' Prior to any feature engineering steps, the full NSL-KDD dataset will be split into new training and testing sets that each contain 50% of the data. A validation set will also be split from the training set at a 20:80 ratio which will be used to assess the random forest model during feature selection. This set will be reintegrated after feature selection so that the whole (but still downsampled) training dataset is used for modeling. Once the features are selected, the training and testing sets will be modified to contain those predictor variables.

#+ split
# Split the training and testing sets for modeling
set.seed(4960)
kdd_train_test_split <- initial_split(kdd,
                                      prop = 0.5)

kdd_train_val <- training(kdd_train_test_split)
kdd_test <- testing(kdd_train_test_split)

# Split the training and validation sets for feature selection
set.seed(4960)
kdd_train_val_split <- initial_split(kdd_train_val,
                                     prop = 4/5)

kdd_train <- training(kdd_train_val_split)
kdd_val <- testing(kdd_train_val_split)

#' ## Downsampling
#' 
#' Downsampling is a technique used in machine learning to balance the class distribution of a dataset when there is a significant difference in the number of observations between classes. This can be important because many machine learning algorithms assume that classes are balanced, and when they are not, the algorithm may be biased towards the majority class, resulting in poor performance on the minority class. Additionally, it can be used when modeling algorithms run inefficiently due to large data sets.
#' 
#' Downsampling involves randomly selecting a subset of the majority class observations to match the number of observations in the minority class. This can help to balance the class distribution, allowing the machine learning algorithm to learn from a more representative sample of the data. This can lead to improved performance and reduced overfitting on the majority class, since it reduces the number of samples.
#' 
#' The training set will be downsampled based on the *Class* variable so that no variable has a greater than 50:1 ratio with the *U2R* attack classification. After downsampling, downsampling factors will be calculated that will be used to modify the weights of those variables during modeling.

#+ down-sample
samp_factr <- 50

set.seed(4960)
kdd_train_ds <- kdd_train[-sample(which(kdd_train$Class == 'Normal'),
                                  sum(kdd_train$Class == 'Normal') -
                                    samp_factr * sum(kdd_train$Class == 'U2R')), ]
kdd_train_ds <- kdd_train_ds[-sample(which(kdd_train_ds$Class == 'DoS'),
                                     sum(kdd_train$Class == 'DoS') -
                                       samp_factr * sum(kdd_train$Class == 'U2R')), ]
kdd_train_ds <- kdd_train_ds[-sample(which(kdd_train_ds$Class == 'Probe'),
                                     sum(kdd_train$Class == 'Probe') -
                                       samp_factr * sum(kdd_train$Class == 'U2R')), ]

# Calculate downsampling factors
ds_fac_Normal <- sum(kdd_train$Class == 'Normal') / sum(kdd_train_ds$Class == 'Normal')
ds_fac_DoS <- sum(kdd_train$Class == 'DoS') / sum(kdd_train_ds$Class == 'DoS')
ds_fac_Probe <- sum(kdd_train$Class == 'Probe') / sum(kdd_train_ds$Class == 'Probe')

# Calculate case weights
# Add weight to the downsample classes so that:
# Example weight = original weight * downsampling factor
n_samples <- nrow(kdd_train_ds)
n_classes <- length(unique(kdd_train_ds$Class))
case_wts <- kdd_train_ds %>%
  group_by(Class) %>%
  summarize(n_samples_j = n(),
            case_wts = n_samples / (n_classes * n_samples_j))

case_wts$case_wts[1] <- case_wts$case_wts[1] * ds_fac_Normal
case_wts$case_wts[2] <- case_wts$case_wts[2] * ds_fac_DoS
case_wts$case_wts[3] <- case_wts$case_wts[3] * ds_fac_Probe

# Join case weights to downsampled training set and set as importance_weights type
kdd_train_ds <- kdd_train_ds %>%
  left_join(case_wts %>%
              select(Class, case_wts),
            by = 'Class') %>%
  mutate(case_wts = importance_weights(case_wts))

#' ## Boosted random forest variable selection
#' 
#' Variable selection is the process of selecting a subset of relevant variables for use in building a predictive model. The goal is to select a subset of variables that maximizes the model's performance while minimizing the complexity of the model. This can be done using various techniques such as stepwise selection, LASSO, Ridge regression, etc. For this analysis, a boosted random forest model will be used to select features based on their 'importance'.
#' 
#' In the context of random forests, feature importance refers to the measure of how much each feature contributes to the prediction of the target variable. Random forests provide two ways to measure feature importance:
#' 
#' 1. Mean Decrease in Impurity (MDI): It measures the decrease in node impurities (e.g. Gini impurity) when a feature is used to split the data at each tree node. The higher the MDI, the more important the feature is.
#' 2. Mean Decrease in Accuracy (MDA): It measures the decrease in accuracy when a feature is permuted (randomly re-ordered) in out-of-bag samples. The higher the MDA, the more important the feature is.
#' 
#' Both MDI and MDA are computed for each feature and then the feature importance score is the average of the MDI or MDA across all the decision trees in the forest. The features with an importance score of at least 0.1% will be selected for modeling.
#' 
#' ### Feature engineering
#' 
#' With the training set semi-balanced it is set for selecting features. First, a recipe will be written for the random forest model to perform all of the feature engineering steps. These steps are:
#' 
#' 1. Removing variables that will not be used in modeling
#' 2. Pooling values that occur less than 1% in factor variables
#' 3. Creating natural spline expansions for select continuous variables up to a degree of 10
#' 4. Making dummy variables for factor variables
#' 5. Adding 2-way interactions between all variables
#' 6. Removing variables with zero variance
#' 7. Performing a Yeo-Johnson transformation to normalize the predictor 

#+ recipe
kdd_featEng_recipe <- kdd_train_ds %>%
  recipe(Class ~ .) %>%
  # Remove non-predictor vars
  step_rm('Difficulty.Level', 'Subclass', 'Type') %>%
  # Pool infrequently occurring values in nominal pred vars
  step_other(all_nominal_predictors(),
             threshold = 0.01,
             other = 'step_other') %>%
  # Create natural spline expansions of select vars with a df of 10
  step_ns(Duration, Src.Bytes, Dst.Bytes, Num.Compromised, Num.Root, Count, Srv.Count,
          deg_free = 10) %>%
  # Make dummy vars for the nominal preds but keep the original vars
  step_dummy(all_nominal_predictors(),
             one_hot = TRUE) %>%
  # Add interactions for all preds
  step_interact(~ all_predictors():all_predictors(),
                sep = ':') %>%
  # Remove preds with zero variance
  step_zv(all_numeric_predictors()) %>%
  # Transform select vars using Yeo-Johnson transform
  step_YeoJohnson(all_numeric_predictors())

#' ### Boosted tree classification
#' 
#' Boosted tree classification is an ensemble machine learning technique that combines multiple decision trees to improve the accuracy of the model. The technique uses a technique called boosting, which iteratively trains decision trees and adjusts the weights of the data points so that the model focuses more on the misclassified points in the previous tree. The final output of the model is a combination of the predictions of all the trees, where each tree is assigned a weight based on its accuracy.
#' 
#' The boosted tree model used for this variable selection has been 'soft-tuned', in that it was not be tuned over a grid or using a specific algorithm but instead a few hyperparameters were tested and the best selected based on some criteria. For example, the `min_n` hyperparameter, which indicates the maximum number of data points in a node required for splitting, is set to `2` as it performed the best for correctly classifying *U2R* intrusions.

#+ boosted-tree
# Get the number of available cores to use in parallel with xgboost
n_cores <- parallel::detectCores()

bt_model <- boost_tree(mtry = 300,
                       trees = 200,
                       min_n = 2,
                       tree_depth = 10,
                       learn_rate = 0.2) %>%
  set_engine('xgboost',
             nthread = n_cores) %>%
  set_mode('classification') %>%
  translate()

bt_wf <- workflow() %>%
  add_model(bt_model) %>%
  add_recipe(kdd_featEng_recipe) %>%
  add_case_weights(case_wts)

set.seed(4960)
bt_fit <- bt_wf %>%
  fit(kdd_train_ds)

#' ### Model assessment
#' 
#' To check that the boosted tree model had a good fit, predictions on the validation set will be made. The confusions matrix and statistics on the prediction accuracy, recall, and precision will be printed.

#+ bt-preds
bt_probs <- predict(bt_fit,
                    new_data = kdd_val,
                    type = 'prob')

bt_preds <- names(bt_probs)[max.col(bt_probs,
                                    ties.method = "first")] %>%
  str_remove('\\.pred_')

# Set class levels so that 'Normal' is first
class_levels <- c("Normal", "DoS", "Probe", "R2L", "U2R")

bt_results <- bind_cols(bt_probs,
                        pred = bt_preds,
                        obs = kdd_val$Class) %>%
  mutate(pred = fct_relevel(pred, class_levels),
         obs = fct_relevel(obs, class_levels))

bt_confmat <- table(observed = bt_results$obs,
                    predicted = bt_results$pred)

bt_confmat

accuracy_vec(bt_results$obs,
             bt_results$pred)

recall_vec(bt_results$obs,
             bt_results$pred)

precision_vec(bt_results$obs,
             bt_results$pred)

#' Overall the model did well predicting on the validation set, with an accuracy, recall, and precision of 0.989, 0.900, and 0.875, respectively. Additionally, the confusion matrix shows that the model was decent with its predictions for each of the 5 connection types. Therefore, the features gathered from the model are likely to be related to the connections types and useful to include in other models.
#' 
#' ### Variable selection
#' 
#' To get an idea of what the important variables look like a plot of the top 20 variables will be printed.

#+ vip
bt_fit %>%
  extract_fit_parsnip() %>%
  vip(num_features = 20) +
  theme_classic()

#' Most of the top variables are interaction terms, of which most include one of the splines of *Src.Bytes* and *Dst.Bytes*. The *Dst.Host.Diff.Srv* also shows up commonly in the top interaction terms. This suggests that these variables are important predictors of connection classes. However, these variables are likely to be highly correlated as they come from the same sources, which we may need to take into account for models adversely affected by multicollinearity.
#' 
#' Next, the variables scored to have at least 0.1% importance will be collected. The variables that are interaction terms will be put into a formula to be used when creating the interactions in future recipes. This will greatly increase the speed of the recipes as only the select interaction terms will be generated rather than all possible terms.
#' 
#' Important variables that are not interaction terms will be separately collected and used to filter out unselected variables at the end of the recipe.

#+ vi-select
# Select vars with at least 0.1% importance
imp_vars <- bt_fit %>%
  extract_fit_parsnip() %>%
  vi() %>%
  filter(Importance > 0.001)

# Identify the significant interactions and create a formula
imp_vars_ints <- imp_vars$Variable[imp_vars$Variable %>%
                                     str_detect(':')]

imp_vars_ints_formula <- as.formula(paste('~ ', paste(imp_vars_ints, collapse = '+')))

nonint_vars <- imp_vars$Variable[!imp_vars$Variable %>%
                                   str_detect(':')]

#' To note, there are 143 variables selected as having at least 0.1% importance in the boosted tree model.
#' 
#' ## Engineering and selecting final features
#'
#' Next, the full training, which includes the training and validation sets for the boosted tree model above, will similarly be downsampled, which will provide more *R2L* and *U2R* observations.

# Downsample Normal, DoS, and Probe vars so that they are 50:1 to U2R
samp_factr <- 50

set.seed(4960)
kdd_train_val_ds <- kdd_train_val[-sample(which(kdd_train_val$Class == 'Normal'),
                                  sum(kdd_train_val$Class == 'Normal') -
                                    samp_factr * sum(kdd_train_val$Class == 'U2R')), ]
kdd_train_val_ds <- kdd_train_val_ds[-sample(which(kdd_train_val_ds$Class == 'DoS'),
                                     sum(kdd_train_val$Class == 'DoS') -
                                       samp_factr * sum(kdd_train_val$Class == 'U2R')), ]
kdd_train_val_ds <- kdd_train_val_ds[-sample(which(kdd_train_val_ds$Class == 'Probe'),
                                     sum(kdd_train_val$Class == 'Probe') -
                                       samp_factr * sum(kdd_train_val$Class == 'U2R')), ]

# Calculate new case weights
n_samples <- nrow(kdd_train_val_ds)
n_classes <- length(unique(kdd_train_val_ds$Class))
case_wts <- kdd_train_val_ds %>%
  group_by(Class) %>%
  summarize(n_samples_j = n(),
            case_wts = n_samples / (n_classes * n_samples_j))

# Calculate downsampling factors
ds_fac_Normal <- sum(kdd_train_val$Class == 'Normal') / sum(kdd_train_val_ds$Class == 'Normal')
ds_fac_DoS <- sum(kdd_train_val$Class == 'DoS') / sum(kdd_train_val_ds$Class == 'DoS')
ds_fac_Probe <- sum(kdd_train_val$Class == 'Probe') / sum(kdd_train_val_ds$Class == 'Probe')

case_wts$case_wts[1] <- case_wts$case_wts[1] * ds_fac_Normal
case_wts$case_wts[2] <- case_wts$case_wts[2] * ds_fac_DoS
case_wts$case_wts[3] <- case_wts$case_wts[3] * ds_fac_Probe

# Join case weights to downsampled training set and set as importance_weights type
kdd_train_val_ds <- kdd_train_val_ds %>%
  left_join(case_wts %>%
              select(Class, case_wts),
            by = 'Class') %>%
  mutate(case_wts = importance_weights(case_wts))

#' The downsampled training set will then be engineered to contain the variables selected from the boosted tree model. The training set used for the boosted tree model will be used to create the recipe so that the same transformations that were modeled are applied. The testing set will similarly be engineered.

#+ final-recipe
kdd_model_recipe <- kdd_train_ds %>%
  recipe(Class ~ .) %>%
  # Remove non-predictor vars
  step_rm('Difficulty.Level', 'Subclass', 'Type') %>%
  # Pool infrequently occurring values in nominal pred vars
  step_other(all_nominal_predictors(),
             threshold = 0.01,
             other = 'step_other') %>%
  # Create natural spline expansions of select vars with a df of 5
  step_ns(Duration, Src.Bytes, Dst.Bytes, Num.Compromised, Num.Root, Count, Srv.Count,
          deg_free = 10) %>%
  # Make dummy vars for the nominal preds but keep the original vars
  step_dummy(all_nominal_predictors(),
             one_hot = TRUE) %>%
  # Add interactions for all preds
  step_interact(imp_vars_ints_formula,
                sep = ':') %>%
  # Remove preds with zero variance
  step_zv(all_numeric_predictors()) %>%
  # Transform select vars using Yeo-Johnson transform
  step_YeoJohnson(all_numeric_predictors())

# Apply the recipe and select only important variables for the training then testing sets
kdd_train_val_ds_baked <- kdd_model_recipe %>%
  prep(training = kdd_train_ds,
       verbose = TRUE) %>%
  bake(new_data = kdd_train_val_ds,
       contains('Class'),
       contains(':'),
       all_of(nonint_vars),
       case_wts) %>%
  mutate(Class = fct_relevel(Class, c('Normal', 'DoS', 'Probe', 'R2L', 'U2R')))

kdd_test_baked <- kdd_model_recipe %>%
  prep(training = kdd_train_ds) %>%
  bake(new_data = kdd_test,
       contains('Class'),
       contains(':'),
       all_of(nonint_vars)) %>%
  mutate(Class = fct_relevel(Class, c('Normal', 'DoS', 'Probe', 'R2L', 'U2R')),
         case_wts = 1,
         case_wts = importance_weights(case_wts))

#' Some functions have specific restraints on the characters used in variable names, in this case the colons (':') can cause issues. So, the variable names will be modified, then both the training and testing sets will be saved as `R` objects and are ready for modeling. The environment will then be cleaned to free up space.

#+ save
# Make happy column names for future processes
colnames(kdd_train_val_ds_baked) <- make.names(colnames(kdd_train_val_ds_baked))
colnames(kdd_test_baked) <- make.names(colnames(kdd_test_baked))

saveRDS(kdd_train_val_ds_baked, 'data/interim/kdd_train_ds_baked.RDS')
saveRDS(kdd_test_baked, 'data/interim/kdd_test_baked.RDS')

rm(list = ls())
gc()