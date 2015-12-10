#### liste des variables inutiles dans le modele
# plus precisement, on parle des variables qui dont l'info est 
# exploitee mais qui sont supprimees par la suite



rm(list = ls()) # Clean the workspace
gc()            # Garbage collecting (for memory efficiency)
user <- "IFS"
## AE
if (user=="AE_port"){
  chem_patr <-"M:/data/Patrimoine/EP 2009-10/Stata/"
  dest <-"M:/Myliam2/Patrimoine/"
}
if (user=="IPP_pers"){
  chem_patr <-"M:/Patrimoine/EP 2009-10/Stata/"
  dest <-"C:/Myliam2/Patrimoine/"
}
if (user=="IPP"){
  chem_patr <-"M:/Patrimoine/EP 2009-10/Stata/"
}

if (user=="IFS"){
  chem_patr <-"T:/data/Patrimoine/EP 2009-10/Stata/"
  dest <-"T:/Myliam2/Patrimoine/"
}
setwd(dest)


library(foreign)
men   <- read.dta(paste0(chem_patr,"menage.dta"))
ind   <- read.dta(paste0(chem_patr,"Individu.dta"))

 



# on retire identind qui ne sert Ãƒ  rien, et prodep qu'on a amÃƒÂ©liorÃƒÂ© dans cydeb1 et toutes les variables construite ou inutile
ind <- subset(ind, select = - c(prodep,t5age))
# on retire aussi toute les variables concernant pr et cj puisqu'on peut
# recuperer celle qu'on veut plus tard
var_pr = names(men)[grep("pr$", names(men))] 
var_pr <- var_pr[which(! var_pr %in% c("indepr","r_dcpr","r_detpr"))]
var_pr = c(var_pr, names(men)[grep("cj$", names(men))] )
men = subset(men, select= names(men)[which(! names(men) %in% var_pr)] )
# dans la meme veine, on retire les variables diplomes
dipl = names(men)[grep("^diplom", names(men))] 
men = subset(men, select= names(men)[which(! names(men) %in% dipl)] )
# on peut maintenant retirer toutes les variables concernant les enfants a l exterieur du menage
hors_foyer = names(men)[grep("^hod", names(men))] 
# length(ToRemove) 157 variables tout de mÃƒÂªme
men = subset(men, select= names(men)[which(! names(men) %in% hors_foyer)] )
# on passe de 856 variable a 611
names(men)

rev = names(men)[grep("^rf", names(men))] 
r_ = names(men)[grep("^r_", names(men))] 
zr = names(men)[grep("zr", names(men))] 
cj = names(men)[grep("cj", names(men))] 

### variables ind

cj = names(ind)[grep("_i$", names(ind))]

list_ind = names(ind)
past_activ = names(ind)[grep("^cy", names(ind))] 
list_ind = setdiff(list_ind,past_activ)
detention = union( names(ind)[grep("^p0", names(ind))] , names(ind)[grep("^p1", names(ind))] )
list_ind = setdiff(list_ind,detention)
rind = names(ind)[grep("^rind", names(ind))] 
antilles = c("rsalmtcd", "rsdurd", "rsprimmtcd", "rindbmtcd", "rinddmtcd", 
             "chomtcd", "rretmtcd", "rmalmtcd")
list_ind = setdiff(list_ind,antilles)

var retro 
  indiv 
ANFINETU Année de fin des études initiales
ANARRIV Année d'arrivée en France
NAIS7 Lieu de naissance
DURCHO Nombre de mois passés au chômage au cours des 12 derniers mois
DUREE Durée totale d'activité effective (en années)
EDUC Détention de rentes éducation
JEQUIT Année de départ du foyer parental


names(men)[grep("^for", names(men))]
names(ind)[grep("^for", names(ind))]