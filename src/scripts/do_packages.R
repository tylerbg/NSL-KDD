# This script defines a function that takes a list of package names, checks if they are installed,
# installs them if they are not already installed, and loads them. All cores are used for
# installing packages.

do_packages <- function(libs) {
  # Get the number of available cores
  n_cores <- parallel::detectCores()
  
  # Check which packages are already installed and install if not
  new.packages <- libs[!(libs %in% installed.packages()[, "Package"])]
  if(length(new.packages)) install.packages(new.packages,
                                            Ncpus = n_cores)
  
  sapply(libs, require, character.only = TRUE)
}