# TODO: retro and table anc should be with other ones in saved files

## run all
rm(list = ls()) # Clean the workspace

## options :
user <- "IPP" # sert pout les chemins

option_run = "oui" # if run=="oui" then calcule les etapes depuis le debut avec import
option_expand = "non" # if expand=="oui" then calcule avec la version etandue
option_retro = "non" # if retro=="oui" then calcule le retro sinon part de retro.csv
# attention option_lien = "oui" quand "option_expand="oui" en general
option_lien = "oui" # if lien=="oui" then calcule les liens pere et mere en +, sinon non


gc() # Garbage collecting (for memory efficiency)
options(scipen=50) # to prevent scientific notation in csv files
library(plyr)

## path configuration : modify the file path_configREMOVE_THIS_AFTER_MODIFICATION.R
## to suit your needs

source("./path_config.R")
setwd(dest)


if (option_run == "oui") {
  # note : on retire les antilles
  print("import")
  source("Import/import.R") # modifie les tables ind et men
  print("declar")
  source("Import/declar.R") # modifie les tables ind et men
  print("lien_parenf")
  source("lien_parent_enfant/lien_parenf.R") # cree look_child et look_parent
  ### selection des variables
  person = subset(ind, select=c(id,period,res,quires,age,age_en_mois,sexe,conj,mere,pere,findet,diplome,
                                anc,xpr,workstate,
                                salaire_imposable,choi,alr,rsti,zrentes_i,
                                zrag_i, zric_i, zrnc_i,
                                foy,quifoy))
  sapply(person,class)
  
  menage = subset(men, select=c(res,pref,period,loyer,tu,zeat,surface,resage,restyp,reshlm,pond) )
  menage = as.data.frame(lapply(menage,as.integer))
  menage = rename(menage,c("res"="id"))
  
  if (option_retro == "oui") {
    print("convert")
    source("Past/convert.R") # incorpore les resultats du matching
    print("import_retro")
    source("Import/import_retro.R") # modifie les tables ind et men
  }
  
  save(person,menage,look_child,look_parent,retro,declar,file='ici.Rdata')
}

# ### optionnel : expand
if (option_expand == "oui") {
  load("ici.rdata")
  source("import/expand.R")
  save(person,menage,look_child,look_parent,retro,declar,file='ici_exp.Rdata')
}


if (option_lien == "oui") {
  
  ifelse(option_expand == "oui", load('ici_exp.Rdata'), load("ici.rdata") )
  
  lien = rbind.fill.matrix(as.matrix(look_parent),as.matrix(look_child))
  taille.lien = nrow(lien)
  colnames(lien)[1] <- "id_origin"
  id = seq(1:taille.lien); period = rep(200901, taille.lien); parent = is.na(lien[,1])
  lien = cbind(id,period,parent,lien)
  lien = as.data.frame(apply(lien, 2,as.integer))
  lien = replace(lien, is.na(lien), as.integer(0))
  lien = subset( lien, select = c(id,period,parent,id_origin,pere,mere,sexe,anais,couple,dip6,nb_enf,situa,classif,pond,
                                  mer1e,per1e))

  save(lien,file="lien_parent_enfant/lien.Rdata")
  
  method_link = "score"
  # le bon python (64bit) doit ?tre dans le path system
  system('python lien_parent_enfant/run_lien.py')
  print("Le warning est normal: ZeroDivisionError: integer division or modulo by zero")
  # Remarque si on veut changer le yaml qu'on lance, ouvrir, le run_lien.py avec n'importe quel ?diteur et
  # modifier le chemin du fichier appeler.
  # Remarque : si on veut ajouter une variable.
  # i) travailler dans le fichier lien_parent_enfant/lien_parenf.R (qui pourrait ?tre dans import d'ailleurs)
  # ii) ouvrir lien_parent_enfant/import.yml puis ajouter les variables avec le bon type
  # iii) ouvrir lien_parent_enfant/match_XXX.yml puis ajouter les variables avec le bon type dans la partie entities (
  # un copier coller de l'?tape ii fait l'affaire)
  source("lien_parent_enfant/merge.R")
  
  
  # ### optionnel : expand
  
  if(option_expand == "oui"){
    save(person,menage,declar,retro,file='final_exp.Rdata')
  }
  if(option_expand != "oui"){
    save(person,menage,declar,retro,file='final.Rdata')
  }
}


if (option_lien != "oui") {
  ifelse(option_expand == "oui", load('final_exp.Rdata'), load("final.rdata") )
}


person = replace(person, is.na(person), as.integer(0))
menage = replace(menage, is.na(menage), as.integer(0))
declar = replace(declar, is.na(declar), as.integer(0))
retro  = replace(retro , is.na(retro ), as.integer(0))

person = rename(person, c("res"="men", "quires"="quimen"))
retro = rename(retro, c("res"="men", "quires"="quimen"))
declar = rename(declar, c("res"="men"))

save(person,menage,retro,declar,file='to_import.Rdata') 

load('to_import.Rdata')


# # subset
load('to_import.Rdata')

list = sample(person$id,50)
sub2 = subset(person,id %in% list)
sub1 = data.frame()
while (nrow(sub2) != nrow(sub1)) {
  sub1 = subset(person,id %in% unique(c(sub2$pere,sub2$mere,sub2$id)))
  sub2 = subset(person, men %in% sub1$men)
}
person = sub2
menage = subset(menage, id %in% sub2$men)
declar = subset(declar, id %in% sub2$foy)
save(person,menage,declar,file='subset.Rdata')
load('to_import.Rdata')


