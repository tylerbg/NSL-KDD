library(tidyverse)
library(tidymodels)
library(doParallel)
library(ranger)
library(xgboost)
library(baguette)
library(rules)
library(kknn)
library(naivebayes)

setwd("~/NSL-KDD")

# Load the test set
kdd_test_baked <- readRDS('data/interim/kdd_test_baked.RDS')

# Split for ensemble model building and final testing
kdd_test_baked_split <- initial_split(kdd_test_baked, prop = 0.2)

kdd_test_ens <- training(kdd_test_baked_split)
kdd_test_final <- testing(kdd_test_baked_split)

# Load all of the final models
bag_fit <- readRDS('models/tuning/bag_fit.RDS')
boost_fit <- readRDS('models/tuning/boost_fit.RDS')
C5_fit <- readRDS('models/tuning/C5_fit.RDS')
knn_fit <- readRDS('models/tuning/knn_fit.RDS')
lda_fit <- readRDS('models/tuning/lda_fit.RDS')
multinom_fit <- readRDS('models/tuning/multinom_fit.RDS')
nbay_fit <- readRDS('models/tuning/nbay_fit.RDS')
rf_fit <- readRDS('models/tuning/rf_fit.RDS')

# Get predictions for each model
model_list <- list(bag_fit, boost_fit, C5_fit, knn_fit, lda_fit, multinom_fit, nbay_fit, rf_fit)

preds_df <- lapply(model_list,
                     function(x) predict(x,
                                         kdd_test_ens)) %>%
  data.frame()

colnames(preds_df) <- c('bag', 'boost', 'C5', 'knn', 'lda', 'multinom', 'nbay', 'rf')

# Predict by most common ---------------------------------------------------------------------------

most_common_pred <- apply(preds_df, 1, function(x) names(which.max(table(x)))) %>%
  as_tibble() %>%
  mutate(value = factor(value,
                        levels = c('Normal', 'DoS', 'Probe', 'R2L', 'U2R')))

pred_conf_mat <- table(kdd_test_ens$Class,
      most_common_pred$value)

pred_conf_mat

preds_model_df <- kdd_test_ens %>%
  select(Class) %>%
  bind_cols(preds_df)

preds_model_df %>%
  pivot_longer(bag:rf) %>%
  group_by(name) %>%
  summarize(prop = sum(Class == value) / n())

preds_df2 <- preds_df %>%
  select(bag, boost, C5, multinom, rf)

most_common_pred2 <- apply(preds_df2, 1, function(x) names(which.max(table(x)))) %>%
  as_tibble() %>%
  mutate(value = factor(value,
                        levels = c('Normal', 'DoS', 'Probe', 'R2L', 'U2R')))

pred_conf_mat <- table(kdd_test_ens$Class,
                       most_common_pred2$value)

pred_conf_mat

# Weigh by correct preds

# Tune a multinomial model based on all predictions

# Calculate case weights for the predictions
n_samples <- nrow(preds_model_df)
n_classes <- length(unique(preds_model_df$Class))
case_wts <- preds_model_df %>%
  group_by(Class) %>%
  summarize(n_samples_j = n(),
            case_wts = n_samples / (n_classes * n_samples_j))

preds_model_df <- preds_model_df %>%
  left_join(case_wts %>%
              select(!n_samples_j)) %>%
  mutate(case_wts = importance_weights(case_wts))

# Multinomial model --------------------------------------------------------------------------------
# Set cross-validation folds
set.seed(4960)
cv_folds <- vfold_cv(preds_model_df,
                     v = 10)

multinom_spec <- multinom_reg(penalty = tune(),
                              mixture = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification") %>%
  translate()

multinom_wf <- workflow() %>%
  add_model(multinom_spec) %>%
  add_formula(Class ~ .) %>%
  add_case_weights(case_wts)

multinom_param <- multinom_spec %>% 
  extract_parameter_set_dials()

bayes_control <- control_bayes(verbose = TRUE,
                               verbose_iter = TRUE,
                               no_improve = 10,
                               # Need both below = TRUE for stacking
                               save_pred = TRUE,
                               save_workflow = TRUE)

bayes_metrics <- metric_set(roc_auc,
                            recall,
                            accuracy,
                            precision)

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

autoplot(multinom_bayes)
show_best(multinom_bayes,
          metric = 'roc_auc')

multinom_best_fit_params <- select_best(multinom_bayes,
                                        metric = 'roc_auc')

multinom_final_wf <- multinom_wf %>%
  finalize_workflow(multinom_best_fit_params)

multinom_final_fit <- multinom_final_wf %>%
  fit(preds_model_df)

multinom_final_preds <- predict(multinom_final_fit,
                                new_data = preds_model_df)

table(preds_model_df$Class,
      multinom_final_preds$.pred_class)

accuracy_vec(preds_model_df$Class,
         multinom_final_preds$.pred_class)

recall_vec(preds_model_df$Class,
             multinom_final_preds$.pred_class)



# nnet model --------------------------------------------------------------------------------
# Set cross-validation folds
set.seed(4960)
cv_folds <- vfold_cv(preds_model_df,
                     v = 10)

nnet_spec <- mlp(hidden_units = tune(),
                 penalty = tune(),
                 epochs = tune()) %>% 
  set_engine("nnet") %>% 
  set_mode("classification") %>%
  translate()

nnet_wf <- workflow() %>%
  add_model(nnet_spec) %>%
  add_formula(Class ~ .)

nnet_param <- nnet_spec %>% 
  extract_parameter_set_dials() %>%
  update(hidden_units = hidden_units(c(1, 27)))

bayes_control <- control_bayes(verbose = TRUE,
                               verbose_iter = TRUE,
                               no_improve = 10,
                               # Need both below = TRUE for stacking
                               save_pred = TRUE,
                               save_workflow = TRUE)

bayes_metrics <- metric_set(roc_auc,
                            recall,
                            accuracy,
                            precision)

n_cores <- parallel::detectCores()

cl <- makeForkCluster(n_cores)
registerDoParallel(cl)

set.seed(4960)
nnet_bayes <- nnet_wf %>%
  tune_bayes(resamples = cv_folds,
             iter = 50,
             param_info = nnet_param,
             metrics = bayes_metrics,
             initial = 5,
             control = bayes_control)

stopImplicitCluster()

autoplot(nnet_bayes)
show_best(nnet_bayes,
          metric = 'roc_auc')

nnet_best_fit_params <- select_best(nnet_bayes,
                                        metric = 'roc_auc')

nnet_final_wf <- nnet_wf %>%
  finalize_workflow(nnet_best_fit_params)

nnet_final_fit <- nnet_final_wf %>%
  fit(preds_model_df)

nnet_final_preds <- predict(nnet_final_fit,
                                new_data = preds_model_df)

table(preds_model_df$Class,
      nnet_final_preds$.pred_class)

accuracy_vec(preds_model_df$Class,
             nnet_final_preds$.pred_class)

recall_vec(preds_model_df$Class,
           nnet_final_preds$.pred_class)

# Get final test fits

# Get predictions for each model
cl <- makeForkCluster(n_cores)

preds_df <- parLapply(cl = cl, model_list,
                   function(x) predict(x,
                                       kdd_test_final)) %>%
  data.frame()

stopImplicitCluster()

colnames(preds_df) <- c('bag', 'boost', 'C5', 'knn', 'lda', 'multinom', 'nbay', 'rf')

preds_df <- cbind(kdd_test_final$Class,
                  preds_df)

nnet_final_preds <- predict(nnet_final_fit,
                            new_data = preds_df)

table(preds_model_df$Class,
      nnet_final_preds$.pred_class)

accuracy_vec(preds_model_df$Class,
             nnet_final_preds$.pred_class)

recall_vec(preds_model_df$Class,
           nnet_final_preds$.pred_class)