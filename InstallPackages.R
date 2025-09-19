## Repos
options(repos = c(CRAN = "https://cloud.r-project.org"))

## Tools needed for versioned installs and GitHub
install.packages(c("remotes", "devtools"))

## CRAN packages at exact versions
remotes::install_version("tidyverse",      version = "2.0.0",  upgrade = "never")
remotes::install_version("TTR",            version = "0.24-3", upgrade = "never")
remotes::install_version("devtools",       version = "2.4.5",  upgrade = "never")
remotes::install_version("glmnet",         version = "4.1-7",  upgrade = "never")
remotes::install_version("randomForest",   version = "4.7-1.1",upgrade = "never")
remotes::install_version("rugarch",        version = "1.4-9",  upgrade = "never")
remotes::install_version("reticulate",     version = "1.31",   upgrade = "never")
remotes::install_version("writexl",        version = "1.5.1",  upgrade = "never")
remotes::install_version("rje",            version = "1.12.1", upgrade = "never")
remotes::install_version("MCS",            version = "0.1-3",  upgrade = "never")

## GitHub packages at exact versions
library(devtools)
install_github("cykbennie/fbi", ref = "v0.7.0", upgrade = "never")
install_github("gabrielrvsc/HDeconometrics") # version ref=0.1.0
