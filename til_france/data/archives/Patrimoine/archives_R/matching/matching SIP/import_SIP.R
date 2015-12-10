# TODO: retro and table anc should be with other ones in saved files

## run all
rm(list = ls()) # Clean the workspace

## options :
user <- "IFS" # sert pout les chemins

gc() # Garbage collecting (for memory efficiency)
options(scipen=50) # to prevent scientific notation in csv files
library(plyr)

## chemin
chem_patr <- paste0(chem_patr, "SIP/Stata")
dest <-"C:/til/data/Patrimoine/"
setwd(dest)
chem_patr = switch(user,
       IFS = "T:/data/",
       AE_port = "M:/data/",
       IPP_pers = "M:/")
chem_patr = paste0(chem_patr,"SIP/Stata/")
library(foreign)
ind   <- read.dta(paste0(chem_patr,"individus06.dta"))

Pluisieurs type d'information vont nous servir pour l'appariemment. 
Ce programme travaille sur ces informations pour créer une table sip_match que
l'on pourra utiliser pour matcher avec les individus de la base issue de partimoine. 
Type d'informations : 
    - type de ménage
    - revenus et activité
    - parcours professionnel

Échelle de risque : en matière de santé
handicap
cause inactivité (maladie ?) pour la carriere

menage=
  ANAISn
  MNAISn
  SEXEn
TYPLOG


individu
selection =
  ACTIVCODEA ACTIVLIBEA, CLASSIFEA, DEP, CODE_CSEA, anaisq
EAHSEM, EAMA, EANBH, EARAISTP
FANARR

carriere

EAMA
   
