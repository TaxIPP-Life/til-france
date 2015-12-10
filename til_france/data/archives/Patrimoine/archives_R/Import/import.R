
# lecture des données de l'enquêtre patrimoine
# selection des valeurs utiles pour le matching.
# construction des valeurs dans certains cas.

gc()            # Garbage collecting (for memory efficiency)

library(foreign)
men   <- read.dta(paste0(chem_patr,"menage.dta"))
ind   <- read.dta(paste0(chem_patr,"Individu.dta"))


################################ correction sur la base   ################################
# le pourquoi de ces corrections se trouve dans les programmes de verif

ind$cydeb1 <- ind$prodep  #= pmax(anfinetu,jeactif)
ind$modif  <- "" #on crée une varaible MODIF qui retient que c'est un ménage qui a subi une modif.
ind$cydeb1[c(6723,7137,10641,21847,30072,31545,33382)] <- ind$anais[c(6723,7137,10641,21847,30072,31545,33382)]+20
ind$cydeb1[15207] <- 1963 ; ind$cydeb1[27801] <- 1999
ind$modif[c(15207,27801,6723,7137,10641,21847,30072,31545,33382)] <- "cydeb1 manq"
ind$cyact3[10834] <- "04"
ind$cyact2[23585] <- "11"
ind$cyact3[27817] <- "05"
ind$modif[c(10834,23585,27817)] <- "cyact manq"

moulinette <- function(vecteur) { #fonction qui décale tous les événements de 1 vers la gauche, utile quand il y a un trou
  for (cas in 1:15) {
    ind[vecteur,c(paste0("cyact",cas),paste0("cydeb",cas),paste0("cycaus",cas),paste0("cytpto",cas) )    ]  <<- 
      ind[vecteur, c(paste0("cyact",cas+1),paste0("cydeb",cas+1),paste0("cycaus",cas+1),paste0("cytpto",cas+1))]
    print(paste0('on traite le cas ',cas,' sur 15'))
  }
  ind$modif[vecteur] <<- "decal act"
}
moulinette( c(8298,which(ind$cyact2!="" & ind$cyact1=="" & (ind$cydeb1==ind$cydeb2 | ind$cydeb1>ind$cydeb2 | ind$cydeb1==(ind$cydeb2-1) )))) 

#toute les activités
list.prob.date <- which(ind$cyact2!="" & ind$cyact1==""  & !(ind$cydeb1==ind$cydeb2 | ind$cydeb1>ind$cydeb2 | ind$cydeb1==(ind$cydeb2-1) ))
#on va les mettre à 04 sauf si, leur cyact2==04 et là on met à chomage. 
ind$cyact1[intersect(list.prob.date, which(ind$cyact2!="04"))] <- "04"
ind$cyact1[intersect(list.prob.date, which(ind$cyact2=="04"))] <- "02"
ind$modif[list.prob.date] <- "cyact1 manq"
ind$modif[which(is.na(ind$cydeb1)& (ind$cyact1!=""|ind$cyact2!=""))]  <- "jeact ou anfinetu manq"
ind$cydeb1[which(is.na(ind$cydeb1)& (ind$cyact1!=""|ind$cyact2!=""))] <- pmax(ind$jeactif[which(is.na(ind$cydeb1)&(ind$cyact1!=""|ind$cyact2!=""))],
                                                                              ind$anfinetu[which(is.na(ind$cydeb1)&(ind$cyact1!=""|ind$cyact2!=""))], na.rm = TRUE)
#quand l'ordre des dates n'est pas le bon on fait l'hypothèse que c'est la première date entre anfinetu et jeactif qu'il faut prendre en non pas l'autre
list.prob.deb <- which(ind$cydeb1>ind$cydeb2)
ind$cydeb1[list.prob.deb]  <- pmin(ind$anfinetu[list.prob.deb],ind$jeactif[list.prob.deb])
ind$cydeb1[which(ind$cyact1=="")] <- NA

rm(moulinette,list.prob.date,list.prob.deb)
gc()

############### on supprime les antilles ##################
# Pourquoi ? - elles n'ont pas les memes variables ni l'appariemment
# Pourquoi pas plus tot ? - Des raison historiques : si on remonte ca casse les numeros de
# lignes utilises plus haut
antilles = men$identmen[which(men$zeat == "0")]
men = subset(men, ! identmen %in% antilles)
ind = subset(ind, ! identmen %in% antilles)
rm(antilles)


taille = nrow(ind)
taille.m = nrow(men)

################################  utiles   ################################

## on modifie les identifiants ménage pour que ça ait une tête sympa; on changera éventuellement à la fin
# pour que le numéro soient simplifiés mais pour l'instant cette forme peut être plus évidente en cas de merge
# (encore que si on a une table de passage...)##
ind$res <- as.integer(sprintf("%06d",match(ind$identmen,men$identmen)))  #on laisse un zéro devant, toujours pratique 
men$res <- as.integer(sprintf("%06d",rep(1:length(unique(men$identmen))))) # en cas de duplication, etc
ind$idi  <- paste(ind$res,ind$noi)
ind$id   <- seq(nrow(ind))

# on retire identind qui ne sert à rien, et prodep qu'on a amélioré dans cydeb1 et toutes les variables construite ou inutile
ind <- subset(ind, select = - c(prodep,t5age))
# on retire aussi toute les variables concernant pr et cj puisqu'on peut
# recuperer celle qu'on veut plus tard
names(men)
grep("^pr", names(men))
ToRemove = names(men)[grep("pr$", names(men))] 
ToRemove <- ToRemove[which(! ToRemove %in% c("indepr","r_dcpr","r_detpr"))]
ToRemove = c(ToRemove, names(men)[grep("cj$", names(men))] )
men = subset(men, select= names(men)[which(! names(men) %in% ToRemove)] )
# dans la meme veine, on retire les variables diplomes
ToRemove = names(men)[grep("^diplom", names(men))] 
men = subset(men, select= names(men)[which(! names(men) %in% ToRemove)] )



attach(men)
attach(ind) #note bien : l'ordre est important entre les deux attach, on veut que res renvoie aux res de ind.

############################## conjoint ################################
conj <- integer(taille)
ConjMiss <- c()
ConjToo  <- c()


for (i in which(couple==1)) {
  if(length(which(res==res[i] & couple==1 & id != i))==1){
    conj[i]=which(res==res[i] & couple==1 & id != i)
  }
  else if(length(which(res==res[i] & couple==1 & id != i))>1){
    
    if (ind[i,"lienpref"] %in% c("00","01")) {
      conj[i]=which(res==res[i] & couple==1 & id != i & ind[,"lienpref"] %in% c("00","01"))
    }
    else if (ind[i,"lienpref"] %in% c("02","31") & i != "13582") { # sans ce truc ad-hoc, il faut verifier la longueur 
      # à chaque fois, j'ai pas le courage même si une petite macro pour faire ça serait bien utile pour plein de chose
      conj[i]=which(res==res[i] & couple==1 & id != i & ind[,"lienpref"] %in% c("02","31"))
    }    
    else if (ind[i,"lienpref"] %in% c("03")) {
      conj[i]=which(res==res[i] & couple==1 & id != i & ind[,"lienpref"] %in% c("03"))
    }
    else if (ind[i,"lienpref"] %in% c("32")) {
      conj[i]=which(res==res[i] & couple==1 & id != i & ind[,"lienpref"] %in% c("32"))
    }
    else {
      if (length(which(res==res[i] & couple==1 & id != i & couple==1 & !(ind[,"lienpref"] %in% c("00","01","02","03","32"))))==1    ) {
        conj[i]=which(res==res[i] & couple==1 & id != i & couple==1 & !(ind[,"lienpref"] %in% c("00","01","02","03","32"))) }
      else {
        print (i,ind[i,"lienpref"] )
        ConjToo  <- c(ConjToo,i)}
      }
      
  }
  else {ConjMiss <- c(ConjMiss,i)}
}
# ind[which(res %in% ind[ConjToo,"res"]),]
# table(ind$couple)

############################## Liste des enfants ################################

#on a deux cas: les enfants cohabitant a) et les enfants décohabitant b). Ils sont enregistrés différemment. 
#pour les premiers, on regarde pour chacun qui est leur pere et qui est leur mere dans les tables parent1 et parent2
#ensuite dans une grosse étape, on crée la matrice enfant

#on récupère en même temps le nombre d'enfant
nb_enf  <- integer(taille)

## a) enfants cohabitant
# TODO : améliorer ce programme en utilisant by
parent1 <- integer(taille)
parent2 <- integer(taille)
for (i in which(enf==1)) {
  parent1[i] <- which(res==res[i] & lienpref=="00")
  if ( length(which(res==res[i] & lienpref=="01"))>0) { 
    parent2[i] <- which(res==res[i] & lienpref=="01")
  }
}

par.enf1 <- function(i) {
  if ( length(res==res[i] & lienpref=="00")>0) {parent1[i] <- which(res==res[i] & lienpref=="00")}
  if ( length(which(res==res[i] & lienpref=="01"))>0){  parent2[i] <- which(res==res[i] & lienpref=="01") }
}

for (i in which(enf==1)) {
  par.enf1(i)
}


for (i in which(enf==2)) {
  parent1[i] <-  which(res==res[i] & lienpref=="00")
}
for (i in which(enf==3)) {
  parent1[i] <-  which(res==res[i] & lienpref=="01")
}


#la fonction qui met les parents et qui sera utilisé ci-après pour les petits enfants
ajout.parent <- function() {
  if (length(ll)==1 & parent1[i] == 0) {
    parent1[i] <<- ll
    if (length( which(res==res[i] & !enf %in% c(1,2,3) & ! lienpref %in% c("00","01","21") & id != i ) ) >0 ) {
      parent2[i] <<- which(res==res[i] & !enf %in% c(1,2,3) & ! lienpref %in% c("00","01","21") & id != i )[1]
    }
  }
}


for (i in which(lienpref=="21")) {
  #liste des enfants de la PR et donc des parents potentiels
  ll <- which(res==res[i] & enf %in% c(1,2,3))
  ajout.parent()
  if (is.na(parent1[i])) {
    if (mer1e[i] =="1" & per1e[i]=="1") {
      ll <- intersect(ll, which(couple==1) )
      ajout.parent()
    }
    else if (mer1e[i] =="1" & per1e[i]!="1") {
      ll <- intersect(ll, which(sexe==2))
      ajout.parent()
      ll <- ll[1]
      ajout.parent()
    }
    else if (mer1e[i]!="1" & per1e[i]=="1") {
      ll <- intersect(ll, which(sexe==1))
      ajout.parent()
      ll <- ll[1]
      ajout.parent()
    }
  }
}

      
#On ne fait que maintenant le travail sur le sexe pour dissocier, pere et mere
# C'est peut-être inutile, on verra
pere <- integer(taille)
mere <- integer(taille)

for (i in which(parent1>0)) {
  if (sexe[parent1[i]]==1) { pere[i]= parent1[i]}
  else { mere[i]= parent1[i]}
  if (parent2[i]>0) {
    if (sexe[parent2[i]]==1) { pere[i]= parent2[i]}
    else { mere[i]= parent2[i]}
  } 
}



#c'est pas bien, parce qu'il faut voir d'où ça vient, mais on bricole directement ici
mere[24526] = 24525; pere[24526] = 0
mere[24527] = 24525; pere[24527] = 0



# on fait aussi tourner ce qui suit parce qu'on a besoin du nombre d'enfant

for (i in which(anais<1995)) { #on ne tourne que sur les plus de 14 ans
  # on définit la liste des enfants hors ménage list
  list <- NULL
  if (lienpref[i]=="00") {
    for (k in 1:12) {
      if (get(paste0("hodln",k))[i] %in% c('1','2')) {
        list <- c(list,get(paste0("hodan",k))[i])
      }
    }
  }
  if (lienpref[i]=="01") {
    for (k in 1:12) {
      if (get(paste0("hodln",k))[i] %in% c('1','3')) {
        list <- c(list,get(paste0("hodan",k))[i])
      }
    }
  }
  
  # on a tous les pour calculer le nombre d enfant
  # on prend d'abord les enfants cohabitant
  l <-  union( anais[which(parent1==i)] ,  anais[which(parent2==i)]) #nombre d'enfant cohabitant avec i
  nb_enf[i] <-  length(l)+length(list)
  if (length(list)>0) {
  }
}

rm(ConjMiss,ConjToo,ToRemove,i,k,l,list,ll,parent1,parent2,ajout.parent) # Clean the workspace

ind$conj <- as.integer(conj)
ind$mere <- as.integer(mere)
ind$pere <- as.integer(pere)



######## situation sur le marché  du travail **** 

#on travaille avec situa, puis avec statut, puis avec classif 
#pour de le precision sur le statut

workstate = integer(taille)
workstate[which(situa %in% c(1,2))] = 3
workstate[which(situa == 3)] = 11
workstate[which(situa == 4)] = 2
workstate[which(situa %in% c(5,6,7))] = 1 
workstate[which(situa == 3)] = 0

#precision inactif
workstate[which(preret == 1)] = 9
# precision AVPF
#TODO: Vous pouvez également en bénéficier si vous n'exercez aucune
#     activité professionnelle (ou seulement à temps partiel) et avez 
#     la charge d'une personne handicapée (enfant de moins de 20 ans
#     ou adulte).
#TODO: on fait ça parce que ça colle avec PensIPP pour l'instant
#mais en théorie c'est de la législation, donc on devrait le calculer
avpf.men = men$res[which(paje == 1 | complfam==1 | allocpar==1 | asf==1)]
workstate[which(res %in% avpf.men & workstate %in% c(1,2))] = 8

# public, privé, indépendant
workstate[which(statut %in% c(1,2))] = 5
workstate[which(statut == 7)] = 7

# cadre, non cadre
workstate[which(classif == 6 & workstate == 5)] = 6
workstate[which(classif == 7 & workstate == 3)] = 4
# code destinie reproduit ici
# inactif   <-  1
# chomeur   <-  2
# non_cadre <-  3
# cadre     <-  4
# fonct_a   <-  5
# fonct_s   <-  6
# indep     <-  7
# avpf      <-  8
# preret    <-  9
workstate[which(anais < 2009-64 & workstate == 1)] = 10
table(workstate)
ind$workstate = as.integer(workstate)
rm(avpf.men)

#### QUIRES ######
#TODO: ameliorer en prenant les fichier de OF-data par exemple

quires = as.integer(lienpref)
quires[quires>1] = 2
ind$quires = as.integer(quires)


ind$age =  as.integer(2009 - ind$anais)
ind$age_en_mois =  as.integer(12*ind$age + 11 -ind$mnais)
ind$period = as.integer(rep.int(200901,taille))
men$period = as.integer(rep.int(200901,taille.m))
men$pref = ind$id[which(ind$lienpref=="00")]


# cleaning
list_rename = c("dip14"="diplome", "zsalaires_i"="salaire_imposable", "zchomage_i"="choi",
                "zpenalir_i"="alr", "zretraites_i"="rsti", "agfinetu"="findet",
                "cyder"="anc", "duree"="xpr")

ind = rename(ind, list_rename)

# cleaning
ind$diplome = as.integer(ind$diplome)
ind$findet = as.integer(ind$findet)
ind$anc = as.integer(ind$anc)
ind$xpr = as.integer(ind$xpr)

rm(conj,pere,mere, quires)

