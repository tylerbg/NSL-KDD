# Set Up -------------------------------------------------------------------------------------------

setwd("~/NSL-KDD")

# Install all of the packages used in this analysis if not already installed
libs <- c('tidyverse', 'tidymodels', 'butcher', 'stacks', 'bundle', 'ranger', 'xgboost', 'kknn',
          'baguette', 'finetune', 'doParallel', 'rules', 'naivebayes', 'dbarts', 'rpart', 'discrim',
          'glmnet', 'LiblineaR', 'mda')

n_cores <- parallel::detectCores()

new.packages <- libs[!(libs %in% installed.packages()[, "Package"])]
if(length(new.packages)) install.packages(new.packages,
                                          Ncpus = n_cores)

# Set up data
library(tidyverse)
library(tidymodels)

# tidymodels_prefer()
options(tidymodels.dark = TRUE)

# Load the standard NSL-KDD training and testing sets
kdd_train <- read_csv('data/raw/KDDTrain+.txt',
                      col_names = FALSE,
                      show_col_types = FALSE)

kdd_test <- read_csv('data/raw/KDDTest+.txt',
                     col_names = FALSE,
                     show_col_types = FALSE)

kdd_features <- read_csv('data/external/KDD-features.csv',
                         show_col_types = FALSE)

# Merge the train+ and test+ sets
kdd <- rbind(kdd_train, kdd_test)

# Replace whitespace in the column names with '.'
# '.' used over '_' as step_dummy() uses '_' when creating dummy vars
colnames(kdd) <- kdd_features$`Feature Name` %>%
  str_replace_all('\\s', '.')

# Get information on the data
dim(kdd)
head(kdd)
summary(kdd)

# Count the number of unique observations for each character variable
kdd %>%
  select(where(is.character)) %>%
  apply(2, function(x) length(unique(x)))

# Convert all character variables to factors and remove `Num Outbound Cmds` which is only 0's
# Convert select numeric vars to integer
kdd <- kdd %>%
  mutate(across(where(is.character), factor),
         across(c(Land:Urgent, `Num.Failed.Logins`, `Logged.In`, `Root.Shell`, `Su.Attempted`,
                  `Num.File.Creations`:`Is.Guest.Login`), as.integer)) %>%
  select(!`Num.Outbound.Cmds`)

# Create a new column that categorizes each sub-class into either DoS, Probe, U2R, or R2L
## Write labels
dos_class <- c('apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf',
               'teardrop', 'udpstorm', 'worm')
probe_class <- c('ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan')
u2r_class <- c('buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm')

# Rename the Class variable to Subclass then create a new variable that classifies connections as 
# either normal or one of the 4 attack types
kdd <- kdd %>%
  rename(Subclass = 'Class') %>%
  mutate(Class = case_when(Subclass == 'normal' ~ 'Normal',
                           Subclass %in% dos_class ~ 'DoS',
                           Subclass %in% probe_class ~ 'Probe',
                           Subclass %in% u2r_class ~ 'U2R')) %>%
  replace_na(list(Class = 'R2L')) %>%
  mutate(Class = fct_relevel(Class, 'Normal'))

# Create new variable, Type, that identifies whether the connection is normal or an attack
kdd <- kdd %>%
  mutate(Type = factor(ifelse(Class == 'Normal', 'Normal', 'Attack')))


# Split training data for feature eng/selection
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

# Downsample Normal, DoS, and Probe vars so that they are 50:1 to U2R
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

saveRDS(kdd_train, 'data/interim/kdd_train.RDS')
saveRDS(kdd_train_ds, 'data/interim/kdd_train_ds.RDS')
saveRDS(kdd_val, 'data/interim/kdd_val.RDS')
saveRDS(kdd_test, 'data/interim/kdd_test.RDS')
