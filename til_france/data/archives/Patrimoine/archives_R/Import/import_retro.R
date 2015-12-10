
#############################   Travail sur les carrière #######################################

deb = min(cydeb1[!is.na(cydeb1)])
nb_col = as.numeric(2010-deb+1)

carriere <- matrix("",nrow(ind),nb_col +1 )
names = paste0("T",deb+seq(0,2010-deb))
names[nb_col +1] = "id"
colnames(carriere) <- names

carriere[,nb_col +1] <- ind$id


list.actif <- which(!is.na(cydeb1))
for (i in 1:length(list.actif)) {
  indiv <- list.actif[i]
  for (cas in c(1:15))  {
    if (is.na(get(paste0("cydeb",cas))[indiv])) {break}
    else {
      if (is.na(get(paste0("cydeb",cas+1))[indiv])) {
        carriere[i, (get(paste0("cydeb",cas))[indiv]-deb+1):(nb_col)] <- get(paste0("cyact",cas))[indiv]
      }
      else {
        carriere[i, (get(paste0("cydeb",cas))[indiv]-deb+1):(get(paste0("cydeb",cas+1))[indiv]-deb)] <- get(paste0("cyact",cas))[indiv]
      }
    }
  }
  if(!is.na(cydeb16[indiv])){
    carriere[i,(cydeb16[indiv] -deb+1):(nb_col)] <- cyact16[indiv]
  }
}

carriere= as.data.frame(carriere)
carriere_per=c()
#on split pour que le reshape ne dure pas des heures
for (k in 0:8){
  prem = deb+10*k
  dern = min(deb+10*k+10,2010)
  carriere_per_t = reshape(carriere[(prem-deb+1):(dern-deb+1)],
                                        idvar= "id", v.names="workstate",
                                  varying = list(paste0("T",prem:dern)),
                          times= prem:dern, timevar = "period", direction = "long")
  carriere_per_t = subset(carriere_per_t, subset= workstate!="" )
  carriere_per = rbind(carriere_per,carriere_per_t)
  rm(carriere_per_t)
}


### traduction des activite en format liam = TaxIpp-Life = GeneBios de PensIPP
# code destinie reproduit ici comme dans import
# inactif   <-  1
# chomeur   <-  2
# non_cadre <-  3
# cadre     <-  4
# fonct_a   <-  5
# fonct_s   <-  6
# indep     <-  7
# avpf      <-  8
# preret    <-  9
work = as.numeric(carriere_per$workstate)
work_trad = 1*(work==14 | work==15 | work==12) +
            2*(work==9  | work==10) +
            3*(work==1 | work==4 | work==5 | work==6 | work==11) +
            5*(work==2 | work==3 | work==13) +
            7*(work==7 | work==8)
#TODO:
#utiliser cycaus pour AVPF
#utiliser les infos invalidite
#utiliser les preretraite
#utiliser les infos sur les temps partiel
## Avant de faire ça; refléchir à la meilleure
## méthode et utiliser ce qui est fait pour le
## matching.


retro = subset(ind,select= c(id,res,anais,lienpref,
            jequit, anfinetu,anarriv)) # demenagement            
retrom = subset(men, select=c(res, emmenag,foran, forcjvie,
            forcoupv, forens, formarie,
            formcoupc, forrupt, forruptv,forseul ))
# travail sur le format des donnees
retro = as.data.frame(lapply(retro,as.integer))
retrom = as.data.frame(lapply(retrom,as.integer))

# travail sur le merge
# on supprime le res des gens qui ne sont pas lien=01 car ils ne sont pas
#concernés par les donnees du couple
retro[which(retro$lienpref != 0),"res"] = 0
retro = merge(retro,retrom, by="res", sort= FALSE, all=TRUE)
retro = retro[order(retro$id),]
retro = subset(retro, select = -c(res,lienpref))

retro = merge(retro,person, by="id")

#TODO: Il y a du travail...pour comprendre et corriger les erreurs

# en_coup = subset(test, quires==0 & conj >0)
# single = subset(test, quires==0 & conj == 0)
# prob = which(!is.na(single$foran) & (is.na(single$forrupt) &  is.na(single$forruptv) & single$formcoupc != 3 ))
# single[prob,]
# stat_veuf = which(!is.na(single$foran) & !is.na(single$forruptv))
# stat_separ = which(!is.na(single$foran) & !is.na(single$forrupt))
# count(single[stat_veuf,"forruptv"]-single[stat_veuf,"foran"])
# count(single[stat_separ,"forrupt"]-single[stat_separ,"foran"])
# which(is.na(single$formcoupc) & is.na(single$forcoupv))
# count(single[,"forseul"]-single[,"anais"])


# TODO : ajouter les infos sur les carrieres et conjoints décédés.


# peut-être supprimer MNAIS, si on le garde, il faut tout faire en mois, pourqu'il n'y
#ait pas plus de distance entre décembre et janvier (à cause du changement d'année) qu'entre mars et avril par exemple.

anc = merge(carriere_per,sal_period, all = TRUE)
anc$period = 201001+(2010-anc$period)*100
anc = apply(anc, 2,as.numeric)
anc[which(is.na(anc))]  <- 0
write.csv(anc,file=paste0(dest,"retro2009.csv"),row.names=F)
