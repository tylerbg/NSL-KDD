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
#' 
#' ## Introduction
#' 
#' Feature engineering is the process of creating new features or transforming existing features from raw data to improve the performance of machine learning models. It typically involves selecting, combining, and transforming variables to create new features that better represent the underlying problem and increase the model's ability to learn from the data. This can include techniques like one-hot encoding, normalization, and feature scaling.
#' 
#' Due to the large size of the NSL-KDD dataset, downsampling and case weights will be used to increase the speed and accuracy of the models. After, multiple feature engineering approaches will be employed that include centering and scaling, creating dummy variables, splining, and adding interaction variables. These methods will greatly increase the total number of variables in the dataset, so a random forest will be used to rank variables by importance, and a subset of 'important' variables will be selected for modeling.

#+ setup, include = FALSE
knitr::opts_chunk$set(echo = TRUE,
                      eval = TRUE)
knitr::opts_knit$set(root.dir = '../')
options(width=100)


#' ## Set-up

#+ libs
source('src/scripts/do_packages.R')

libs <- c('tidyverse', 'tidymodels', 'vip')
do_packages(libs)

kdd <- readRDS('data/interim/kdd.RDS')

#' ### Split data
#'
#' Prior to any feature engineering steps, the full KDD dataset will be split into new training and testing sets that each contain 50% of the data. A validation set will also be split from the training set at a 20:80 ratio which will be used to assess the random forest model during feature selection. This set will be reintegrated after feature selection so that the whole (but still downsampled) training dataset is used for modeling. Once the features are selected, the training and testing sets will be modified to contain those predictor variables.

#+ split
set.seed(4960)
kdd_train_test_split <- initial_split(kdd,
                                      prop = 0.5)

kdd_train_val <- training(kdd_train_test_split)
kdd_test <- testing(kdd_train_test_split)

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

kdd_train_ds_wts <- kdd_train_ds %>%
  select(Class) %>%
  mutate(case_wts = case_when(Class == 'Normal' ~ case_wts$case_wts[1] * ds_fac_Normal,
                              Class == 'DoS' ~ case_wts$case_wts[2] * ds_fac_DoS,
                              Class == 'Probe' ~ case_wts$case_wts[3] * ds_fac_Probe,
                              Class == 'R2L' ~ case_wts$case_wts[4],
                              Class == 'U2R' ~ case_wts$case_wts[5]),
         case_wts = importance_weights(case_wts))

kdd_train_ds <- kdd_train_ds %>%
  cbind(kdd_train_ds_wts %>%
          select(case_wts))

# Create recipe
# Create dummy vars for all factors
# Add interactions for all predictors
# Remove predictors with zero variance
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

## Boosted Trees -----------------------------------------------------------------------------------
n_cores <- parallel::detectCores()

bt_model <- boost_tree(mode = 'classification',
                       mtry = 300,
                       trees = 200,
                       min_n = 2,
                       tree_depth = 10,
                       learn_rate = 0.2) %>%
  set_engine('xgboost',
             nthread = n_cores) %>%
  translate()

bt_wf <- workflow() %>%
  add_model(bt_model) %>%
  add_recipe(kdd_featEng_recipe) %>%
  add_case_weights(case_wts)

set.seed(4960)
bt_fit <- bt_wf %>%
  fit(kdd_train_ds)

bt_probs <- predict(bt_fit,
                    new_data = kdd_val,
                    type = 'prob')

bt_preds <- names(bt_probs)[max.col(bt_probs,
                                    ties.method = "first")] %>%
  str_remove('\\.pred_')

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

bt_fit %>%
  extract_fit_parsnip() %>%
  vip(num_features = 20)

# Select vars with at least 0.1% importance
imp_vars <- bt_fit %>%
  extract_fit_parsnip() %>%
  vi() %>%
  filter(Importance > 0.001)


# Identify the significant interactions and create a formula
# This will be used to reduce the processing of baking the test30 and train70 data sets by limiting
# the number of interactions

imp_vars_ints <- imp_vars$Variable[imp_vars$Variable %>%
                                     str_detect(':')]

imp_vars_ints_formula <- as.formula(paste('~ ', paste(imp_vars_ints, collapse = '+')))

nonint_vars <- imp_vars$Variable[!imp_vars$Variable %>% str_detect(':')]

# Write a model formula predicting Class with all of the important variables
# This will be used to select only the important features to include in the upcoming model
imp_vars_formula <- as.formula(paste('Class ~', paste(imp_vars$Variable, collapse = ' + ')))

# Merge training and validation sets
kdd_train2 <- bind_rows(kdd_train,
                        kdd_val)

# Downsample Normal, DoS, and Probe vars so that they are 50:1 to U2R
samp_factr <- 50

set.seed(4960)
kdd_train2_ds <- kdd_train2[-sample(which(kdd_train2$Class == 'Normal'),
                                  sum(kdd_train2$Class == 'Normal') -
                                    samp_factr * sum(kdd_train2$Class == 'U2R')), ]
kdd_train2_ds <- kdd_train2_ds[-sample(which(kdd_train2_ds$Class == 'DoS'),
                                     sum(kdd_train2$Class == 'DoS') -
                                       samp_factr * sum(kdd_train2$Class == 'U2R')), ]
kdd_train2_ds <- kdd_train2_ds[-sample(which(kdd_train2_ds$Class == 'Probe'),
                                     sum(kdd_train2$Class == 'Probe') -
                                       samp_factr * sum(kdd_train2$Class == 'U2R')), ]

# Calculate new case weights
n_samples <- nrow(kdd_train2_ds)
n_classes <- length(unique(kdd_train2_ds$Class))
case_wts <- kdd_train2_ds %>%
  group_by(Class) %>%
  summarize(n_samples_j = n(),
            case_wts = n_samples / (n_classes * n_samples_j))

# Calculate downsampling factors
ds_fac_Normal <- sum(kdd_train2$Class == 'Normal') / sum(kdd_train2_ds$Class == 'Normal')
ds_fac_DoS <- sum(kdd_train2$Class == 'DoS') / sum(kdd_train2_ds$Class == 'DoS')
ds_fac_Probe <- sum(kdd_train2$Class == 'Probe') / sum(kdd_train2_ds$Class == 'Probe')

kdd_train2_ds_wts <- kdd_train2_ds %>%
  select(Class) %>%
  mutate(case_wts = case_when(Class == 'Normal' ~ case_wts$case_wts[1] * ds_fac_Normal,
                              Class == 'DoS' ~ case_wts$case_wts[2] * ds_fac_DoS,
                              Class == 'Probe' ~ case_wts$case_wts[3] * ds_fac_Probe,
                              Class == 'R2L' ~ case_wts$case_wts[4],
                              Class == 'U2R' ~ case_wts$case_wts[5]),
         case_wts = importance_weights(case_wts))

kdd_train2_ds <- kdd_train2_ds %>%
  bind_cols(case_wts = kdd_train2_ds_wts$case_wts)

# Create new dfs with select vars ------------------------------------------------------------------
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
          deg_free = 5) %>%
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

kdd_train2_ds_baked <- kdd_model_recipe %>%
  prep(training = kdd_train_ds,
       verbose = TRUE) %>%
  bake(new_data = kdd_train2_ds,
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

# Make happy column names for future processes
colnames(kdd_train2_ds_baked) <- make.names(colnames(kdd_train2_ds_baked))
colnames(kdd_test_baked) <- make.names(colnames(kdd_test_baked))

saveRDS(kdd_train2_ds_baked, 'data/interim/kdd_train2_ds_baked.RDS')
saveRDS(kdd_test_baked, 'data/interim/kdd_test_baked.RDS')

