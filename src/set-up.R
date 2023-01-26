#' ---
#' title: "Set-up"
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

#' 
#' The NSL-KDD (Network Security Laboratory - Knowledge Discovery and Data Mining) dataset is a dataset for intrusion detection systems (IDS). It was created by modifying the KDD Cup 1999 dataset, which was a dataset for network-based intrusion detection systems, to make it more representative of modern intrusions. The data set includes normal connections and a variety of different types of attacks that include:
#' 
#' - **Denial of Service (DoS).** These attacks are designed to make a network or system unavailable to legitimate users by overwhelming it with traffic or otherwise disrupting its normal operation.
#' - **Probe.** These attacks are designed to gather information about a network or system, such as attempting to find open ports or identify vulnerabilities.
#' - **Remote to Local (R2L).** These attacks involve unauthorized access to a system, such as guessing a password or using stolen credentials, with the goal of gaining local access to the system.
#' - **User to Root (U2R).** These attacks involve unauthorized access to a system, such as hacking or exploiting a vulnerability, with the goal of gaining full control of the system.
#' 
#' This script will set-up the NSL-KDD dataset for exploratory data analysis (EDA) by combinging the split data sets and adding missing features. 

source('src/scripts/do_packages.R')

libs <- c('tidyverse')
do_packages(libs)

#' Both the KDDTrain+ and KDDTest+ data sets will be read into `R`. These raw data files do not include the variables names, so an external data file that includes the variable names and other features will be imported.
#' 
#' The data sets will then be merged so that there is one object that includes all of the KDDTrain+ and KDDTest+ data. The variable names will be assigned to that data set after replacing white space with '.' to be more syntactically valid. The '.' is used to differentiate from later steps that create dummy variables using '_'.

#+ load-data
kdd_train <- read_csv('data/raw/KDDTrain+.txt',
                      col_names = FALSE,
                      show_col_types = FALSE)

kdd_test <- read_csv('data/raw/KDDTest+.txt',
                     col_names = FALSE,
                     show_col_types = FALSE)

kdd_features <- read_csv('data/external/KDD-features.csv',
                         show_col_types = FALSE)

kdd <- rbind(kdd_train, kdd_test)

# Replace whitespace in the column names with '.'
# '.' used over '_' as step_dummy() uses '_' when creating dummy vars
colnames(kdd) <- kdd_features$`Feature Name` %>%
  str_replace_all('\\s', '.')

#' The NSL-KDD data sets included names of the attacks, however lacks a variable to classify them as one of the four attack types as listed above, or even a variable to classify normal versus attack connections. These variables will be created here.

#+ attack-classes
# Write labels
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
  # Set 'Normal' as the first factor
  mutate(Class = fct_relevel(Class, 'Normal'))

# Create new variable, Type, that identifies whether the connection is normal or an attack
kdd <- kdd %>%
  mutate(Type = factor(ifelse(Class == 'Normal', 'Normal', 'Attack')))

#' The data set is now ready for EDA and other analyses, so will be saved as an `R` object to easily load and employ in later scripts. The `R` environment and unused memory will be cleaned.

#+ save
saveRDS(kdd, 'data/interim/kdd.RDS')

rm(list = ls())
gc()
