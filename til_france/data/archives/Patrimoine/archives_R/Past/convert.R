library(foreign)
sal   <- read.dta(paste0(dest,"past/tables_matchees.dta"))
ind_t   <- read.dta(paste0(chem_patr,"Individu.dta"))
men_t   <- read.dta(paste0(chem_patr,"menage.dta"))

##### traduction nouvel identifiant

# on supprime aussi les antilles
antilles = men_t$identmen[which(men_t$zeat == "0")]
#on retire les antilles et on ne garde que les ident
men_t = subset(men_t, ! identmen %in% antilles,select=identmen)
ind_t = subset(ind_t, ! identmen %in% antilles,select=c(identmen,noi))
rm(antilles)
#on refait l'ident de depart
ind_t$identmen <- sprintf("%06d",match(ind_t$identmen,men_t$identmen))   #on laisse un z?ro devant, toujours pratique 
ind_t$identind <- paste(ind_t$identmen,ind_t$noi)
#ident d'arrivee
ind_t$id   <- seq(nrow(ind_t))
ind_t = subset(ind_t,select=c(identind,id))
#on merge pour avoir le bon id

sal = merge(ind_t,sal)
sal = subset(sal,select= -c(identind))


#reshape
sal_period = reshape(sal, idvar= "id", v.names = "salaire_imposable", varying = list(paste0("sal_net",1976:2001)), 
                times= 1976:2001, timevar = "period", direction = "long")

# carriere= as.data.frame(carriere)
# carriere_period = reshape(carr, idvar= "id", varying = list(paste0("Td_",1999:2000)), 
#                           times= 1999:2000, timevar = "period", direction = "long")
