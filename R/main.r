# master script for PART I

# set the path to the data, either relative or absolute
DATA_PATH = "../data/"
RESULTS_PATH = "../results"

# we load the data file and merge firm information
source('./data-final-prep.R')

# we compute the moments and assoicated standard errors
source('./data-final-step2.R')