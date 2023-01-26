#' ---
#' title: "Exploratory data analysis"
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

#' Exploratory Data Analysis (EDA) is an approach to analyzing and understanding data sets through visual and statistical methods. The goal of EDA is to discover patterns, features, and relationships in the data that can inform further analysis and modeling. It is typically the first step in the data analysis process and is used to gain a preliminary understanding of the data before more formal statistical analysis is performed. Techniques used in EDA include visualization, descriptive statistics, and data cleaning.
#' 
#' 
#' 
#' ## Set-up and load data

#+ libs
source('src/scripts/do_packages.R')

libs <- c('tidyverse', 'corrplot')
do_packages(libs)

kdd <- readRDS('data/interim/kdd.RDS')

#' ## Dataset structure
#'
#' The next set of commands will print out information on the dataset's structure.

#+ observe-data
dim(kdd)
str(kdd)

#' The total NSL-KDD data set has 148,517 data points across 43 variables (two of the 45 variables in this data set are created in the set-up script). Notably, there are four character variables that may be more useful re-coded as factors.
#'
#' To check whether the character variables could be more useful as factors the total number of unique values in each will be calculated.

#+ unique-char
kdd %>%
  select(where(is.character)) %>%
  apply(2, function(x) length(unique(x)))

#' The dataset is much larger than any of these values, which can justify re-coding them to the factor type.\

#+ set-factors
kdd <- kdd %>%
  mutate(across(where(is.character), factor))

#' For quality control, duplicated observations and missing values will be checked.

sum(duplicated(kdd))
sum(is.na(kdd))

#' While there are no missing values there are 610 duplicated observations. As the dataset is very large and to reduce potential bias these rows will be removed.

kdd <- unique(kdd)

#' ## Descriptive Statistics
#'
#' With these variables re-coded, summary statistics for all of the variables will be generated.

#+ summary-stats

summary(kdd)

#' The total number of normal and connections are about equal.  Some of the numeric variables have very large ranges compared to other variables, and many have medians = 0.  This could suggeset that there is some right-skew in these variables and that centering and scaling, in addition to transformations, could be useful for modeling. Additionally, there is one variable, *Num.Outbound.Cmds*, that has zero-variance (all observations have the same value) and can be removed.
#' 
#' After removing *Num.Outbound.Cmds*, histograms for the remaining numeric variables will be generated to assess their distributions.

#+ histograms, fig.width = 10, fig.height = 10
kdd <- kdd %>%
  select(!Num.Outbound.Cmds)

kdd %>% 
  select(where(is.numeric)) %>%
  pivot_longer(everything()) %>%
  ggplot(aes(x = value)) +
    geom_histogram(bins = 50) +
    facet_wrap(~ name,
               scales = 'free') +
    labs(x = NULL,
         y = NULL) +
    theme_classic() +
    theme(axis.text = element_blank())

#' From the histograms there are clearly no normally distributed variables. Additionally, while some variables appeared binary from the summary statistics there are clearly a few points in between 0 and 1. Others are still clearly binary, such as *Logged.In*.
#' 
#' ## Two-variable statistics
#' 
#' To observe porential relationships between variables a correlogram of the numeric variables will be generated. The *Type* variable, which indicates whether a connection is normal or an intrusion, will also be included. Spearman's rank correlation will be used as the data are non-normal and include some binary variables. 

#+ correlogram, fig.width = 10, fig.height = 10
kdd_cors <- kdd %>%
  select(where(is.numeric), Type) %>%
  mutate(across(everything(), ~ as.numeric(.))) %>%
  cor(method = 'spearman')

corrplot(kdd_cors,
         method = 'square')

#' In the correlogram above it is obvious there are some relationships between the variables. In particular *Src.Bytes* and *Dst.Bytes* have a strong relationship together while most of the many "count" and "rate" variables are highly correlated with one another. These variables are also rank-correlated with *Type*, suggesting they could be predictive of attacks.
