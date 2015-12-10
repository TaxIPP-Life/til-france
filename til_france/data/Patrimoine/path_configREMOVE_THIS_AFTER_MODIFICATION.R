## This file has to be modified to suit your needs and renamed to path_config.R


user <- "IFS" # sert pout les chemins

## chemin
chem_patr = switch(user,
                   IFS = "T:/data/",
                   AE_port = "M:/data/",
                   IPP = "M:/")
chem_patr = paste0(chem_patr,"Patrimoine/EP 2009-10/Stata/")
dest <-"C:/til/data/Patrimoine/"


  
