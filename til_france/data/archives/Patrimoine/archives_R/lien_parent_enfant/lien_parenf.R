# r?cup?ration de variables pour les liens parents enfants, 
# suppression des variables inutiles ensuite

inf_pr = ind[which(ind$lienpref=="00"),c("id","res","sexe","anais","per1e","mer1e","cs42")]
inf_cj = ind[which(ind$lienpref=="01"),c("id","res","sexe","anais","per1e","mer1e","cs42")]
inf_pr$gpar = 2-( inf_pr$per1e %in% c("1","2") | inf_pr$mer1e %in% c("1","2") ) 
inf_cj$gpar = 2-( inf_cj$per1e %in% c("1","2") | inf_cj$mer1e %in% c("1","2") )
inf_pr = subset(inf_pr, select= c("id","res","sexe","anais","gpar","cs42"))
inf_cj = subset(inf_cj, select= c("id","res","sexe","anais","gpar","cs42"))
colnames(inf_pr) <- paste0(colnames(inf_pr),"pr")
colnames(inf_cj) <- paste0(colnames(inf_cj),"cj")

men = merge( men,inf_pr,by.x="res",by.y="respr")
men = merge( men,inf_cj,by.x="res",by.y="rescj",all=TRUE)


#TODO: gerer les valeurs manquante
#TODO: s'occuper de la precision sur les statuts

# ajouter des elements de matching sur le 
# patrimoine, par exemple, toutes les variabels jepro_

save_inf <- function(i){
  liste  <<- which(men[,paste0("hodln",i)] !="")
  
  hodln <- men[liste,paste0("hodln",i)]
  sexe <- character(length(liste))
  # info sur l'enfant
  sexe   <- men[liste,paste0("hodsex",i)]
  anais  = men[liste,paste0("hodan",i)]
  couple = men[liste,paste0("hodco",i)] #couple=1 et couple=2 devront etre groupes en couple=1
  couple[which(couple=="2")]<-"3"
  dip6   = men[liste,paste0("hodip",i)]
  nb_enf = men[liste,paste0("hodenf",i)]
  # son activite
  situa=  1*(men[liste,paste0("hodemp",i)]==1) +
    2*(men[liste,paste0("hodcho",i)]==3) +
    3*(men[liste,paste0("hodcho",i)]==3) +
    4*(men[liste,paste0("hodcho",i)]==1) +
    5*(men[liste,paste0("hodemp",i)]==2) +
    6*(men[liste,paste0("hodcho",i)]==2) +
    7*(men[liste,paste0("hodcho",i)]==4)
  # on verifie que hodcho est rempli seulement quand hodemp=3
  classif = ifelse( men[liste,paste0("hodpri",i)] %in% c("1","2","3","4"), 
                    men[liste,paste0("hodpri",i)],  
                    men[liste,paste0("hodniv",i)])
  
  
  # info sur les PARENTs
  info_mere    <- matrix("",length(liste),5)
  info_pere    <- matrix("",length(liste),4)
  colnames(info_mere) <- c("mere","jemnais","gparmat","jemprof","jemact")
  colnames(info_pere) <- c("pere","jepnais","gparpat","jepprof")
  
  inf_pere <- function(enfants,pers) {
    if (pers == "pr") {
      info_pere[enfants,] <<- as.matrix(men[liste[enfants],c("idpr","anaispr","gparpr","cs42pr")])
    }
    if (pers == "cj") {
      info_pere[enfants,] <<- as.matrix(men[liste[enfants],c("idcj","anaiscj","gparcj","cs42cj")])
    }
  }
  inf_mere <- function(enfants,pers) {
    if (pers == "pr") {
      info_mere[enfants,1:4] <<- as.matrix(men[liste[enfants],c("idpr","anaispr","gparpr","cs42pr")])
    }
    if (pers == "cj") {
      info_mere[enfants,1:4] <<- as.matrix(men[liste[enfants],c("idcj","anaiscj","gparcj","cs42cj")])
    }
  }
  
  inf_pere( which(men$sexepr[liste]==1 & men[liste,paste0("hodln",i)]=="1"), "pr")
  inf_mere( which(men$sexepr[liste]==1 & men[liste,paste0("hodln",i)]=="1"), "cj")
  inf_pere( which(men$sexepr[liste]==2 & men[liste,paste0("hodln",i)]=="1"), "cj")
  inf_mere( which(men$sexepr[liste]==2 & men[liste,paste0("hodln",i)]=="1"), "pr")
  inf_pere( which(men$sexepr[liste]==1 & men[liste,paste0("hodln",i)]=="2"), "pr")
  inf_mere( which(men$sexepr[liste]==2 & men[liste,paste0("hodln",i)]=="2"), "pr")
  inf_pere( which(men$sexepr[liste]==1 & men[liste,paste0("hodln",i)]=="3"), "cj")
  inf_mere( which(men$sexepr[liste]==2 & men[liste,paste0("hodln",i)]=="3"), "cj")
  
  res <- men[liste,"res"]
  pond <- men[liste,"pond"]
  print(length(liste))
  PARENT = rep.int(1,length(liste))
  ajout <- cbind(res,sexe,anais,couple,dip6,nb_enf,situa,classif,info_pere,info_mere, PARENT,hodln,pond)
}

look_child <- save_inf(1)
for (k in 2:12) {
  look_child <- rbind(look_child,save_inf(k))
}




# on peut maintenant retirer toutes les variables concernant les enfants a l exterieur du menage
# ToRemove = names(men)[grep("^hod", names(men))] 
# # length(ToRemove) 157 variables tout de même
# men = subset(men, select= names(men)[which(! names(men) %in% ToRemove)] )



#### info sur les PARENTs
look_parent <- as.matrix(subset(ind, per1e == "2" | mer1e == "2", 
                                   select= c(id,sexe,anais,couple,diplome,situa,jemnais,gparmat,jemprof,jemact,
                                             jepnais,gparpat,jepprof,per1e,mer1e,jegrave_div,classif,pond)))

# on supprime les infos quand on ne cherche pas ce PARENT
pas_pere <- which(look_parent[,"per1e"] %in% c("1","3","4") )
look_parent[pas_pere,c("jepnais","gparpat","jepprof")] <- NA
pas_mere <- which(look_parent[,"mer1e"] %in% c("1","3","4") )
look_parent[pas_mere,c("jemnais","gparmat","jemprof","jemact")] <- NA
rm(pas_pere,pas_mere)

#  hodind=substr(acti,1,1)
#   if (hodind==4 & statut!=6) {hodind=5}

dip6 = character(nrow(look_parent))
dip6[] <- "6"
dip6[which(look_parent[,"diplome"]>=30)]  <- "5"
dip6[which(look_parent[,"diplome"]>=41)]  <- "4"
dip6[which(look_parent[,"diplome"]>=43)]  <- "3"
dip6[which(look_parent[,"diplome"]>=50)]  <- "2"
dip6[which(look_parent[,"diplome"]>=60)]  <- "1"
PARENT = rep.int(0,nrow(look_parent))

look_parent[which(look_parent[,"classif"] %in% c("1","2","3")),"classif"] <- "a"
look_parent[which(look_parent[,"classif"] %in% c("4","5")),"classif"] <- "2"
look_parent[which(look_parent[,"classif"] %in% c("6","7")),"classif"] <- "1"
look_parent[which(look_parent[,"classif"] %in% c("8","9")),"classif"] <- "3"
look_parent[which(look_parent[,"classif"] %in% c("a")),"classif"] <- "4"

look_parent <- subset(look_parent,select=-c(diplome))
look_parent <- cbind(look_parent,dip6,PARENT,nb_enf[which(per1e == "2" | mer1e == "2")])
colnames(look_parent)[which(colnames(look_parent)=="")] <- "nb_enf" 

 
# # élément à utiliser si on veut matcher, not
look_child = as.data.frame(apply(look_child, 2,as.integer))   
look_parent = as.data.frame(apply(look_parent, 2,as.integer))

look_child = look_child[order(look_child$res),]
look_child = subset(look_child,  select= c(res,mere,pere,sexe,anais,couple,dip6,nb_enf,situa,classif,hodln,pond))
look_parent = subset(look_parent, select= c(id,mer1e,per1e,sexe,anais,couple,dip6,nb_enf,situa,classif,pond))

rm(inf_cj,inf_pr)


#JEGRAVE_DIV==1 alors on cherche parmi HODLN=2 ou HODLN=3
# pour l'instant, on laisse de cote la profession des PARENTs qui de toute facon ne colle pas parce que c'est dans
# la jeunesse de l'individu


## il n'y a plus qu'a faire ce matching....
# list.class = c("sexe","anais","couple","dip6","nb_enf","situa")
# lienDF=as.data.frame(lien)
# test <- split(lienDF, f=as.list(lienDF[ ,c(list.class)]), drop=TRUE)
# test[[2]]
# 
# taille.ssgroupe <- unlist(lapply(test,nrow)) # table(taille.ssgroupe.pat) ; sum(taille.ssgroupe.pat)
# 
# l.pat <- split(pat, f = as.list(pat[,c(match.exact) ]),drop=TRUE)
