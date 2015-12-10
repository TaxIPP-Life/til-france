###### creation d'une table foyer ###### 
## Tout ceci peut se faire sur la table non elargie.

# Dans l ideal et pour les initiés de l'ERFS, on pourrait relier les FIP
# avec des EE, et ça par contre, il faut le faire avec la table étendue
# cela dit, on n'a pas ces infos donc ne nous prenons pas trop la tête,
# on fait comme si et on simulera plus tard.

# On pourrait aussi penser faire tourner ça après les liens parents_enfants
# pour associer par exemple les jeunes de 19 ans hors du foyer à la declaration
# declaration.

library(plyr) # pour la fonction count

# liste des variables de l'enquete de source fiscale, en dehors des revenus
# individuel
foymen <- subset(men,select=c(res,zcsgcrds, zfoncier, zimpot,
                          zpenaliv,zpenalir,zpsocm,zrevfin,pond))


## obtain quifoy
#TODO
foyind <- subset(ind,select=c(id,res,etamatri,lienpref,age,sexe,conj,pere,mere))
foyind$quifoy <- 0
foyind$idfoy <- 0
# cette premiere version n'est pas coh?rente sur l'ensemble du mod?le
# spouse = which(foyind$lienpref=="01"  & foyind$etamatri==2)
# child = which(foyind$lienpref=="02" & foyind$etamatri==1 & foyind$age <25 )
child = which((foyind$pere>0 | foyind$mere>0)  & foyind$etamatri==1 & foyind$age <25 )
spouse = which(foyind$conj>0 & foyind$etamatri==2 & (foyind$conj< foyind$id)) #la condition sur le conj et le id est pour 
# ne prendre qu'un individu par couple, on pourrait prendre le sexe mais ?a coince sur les couples homo. 
# cela dit, pour l'instant, on va ?tre scandaleusement irrespectueux, en changeant le sexe d'un des membres du couple
# c'est mal mais comme le model est cod? comme ?a apr?s, ?a simplifie pour l'instant.

#TODO: faire de meilleures correstion table(foyind$etamatri[foyind$conj[spouse]])
foyind$etamatri[foyind$conj[spouse]] = "2"
foyind$sexe[spouse] = ifelse(foyind$sexe[foyind$conj[spouse]]==1,"2","1")


# la selection devrait se faire sur le revenu
foyind$quifoy[spouse] <- 1
foyind$quifoy[child]  <- 2
#c'est pour l'instant simpliste : TODO

vous = which(foyind$quifoy==0)
conj = which(foyind$quifoy==1)
pac = which(foyind$quifoy==2)
declar = data.frame(cbind(seq(1,length(vous)),vous ,foyind$res[vous]))
colnames(declar) = c("idfoy","vous","res")


foyind$idfoy[vous] = seq(1,length(vous))
foyind$idfoy[conj] = foyind$idfoy[foyind$conj[conj]]
#pac
pere_pac = numeric(length(pac))
mere_pac = numeric(length(pac))
pere_pac[which(foyind$pere[pac]>0)] = foyind$idfoy[foyind$pere[pac]]
mere_pac[which(foyind$mere[pac]>0)] = foyind$idfoy[foyind$mere[pac]]

foyind$idfoy[pac] = ifelse( foyind$pere[pac] >0  ,
                            pere_pac,
                            mere_pac)

# on peut verifier que tout le monde ? un idfoy

# # ancienne m?thode ? effacer si on est sur qu'on ne fera pas plus compliqu? un jour
# idfoy = foy[1,1]
# k = 0
# res = 0
# foyind$test = 0
# marqueur = 0 # ce marqueur sert à noter que l'on change de idfoy quand 
# # on change de menage mais qu'alors, il ne faut pas rechanger de idfoy au
# # premier quifoy==0 que l'on rencontre
# for (i in 1:nrow(ind)) { #ne marche que si on commence par un quifoy==0
#   if (foyind$quifoy[i] == 0 & marqueur == 0) {
#     k = k+1
#   }
#   if (foyind$quifoy[i] == 0 & marqueur == 1) {
#     marqueur = 0
#   }  
#   if ( foyind$res[i] != res & foyind$quifoy[i] != 0) {
#     k = k+1
#     marqueur = 1
#   }
#   foyind$idfoy[i] = foy[k,1]
#   res=foyind$res[i]
# # on pourrait travailler sur les grands-parents : 22
# # ou sur les gendres : 31 
#   
#   if (i %% 1000 == 0) {
#     print(paste("done ",i/1000, "over" , trunc(nrow(ind)/1000)))
#   }
# }  


## repartition des revenus menage par declaration
# TODO : tres tres primaire pour l'instant
foy_by_res = subset(foyind, quifoy==0, select = c(res,idfoy))
foymen = merge(foymen,count(foy_by_res,vars="res"))
colnames(foymen)
foymen[,2:8] = round(foymen[,2:8]/foymen[,10])


declar = merge(declar,foymen, by="res", all = TRUE)
declar = subset(declar, select = -c(freq))
declar$period = as.integer(rep.int(200901,nrow(declar))) 
declar$vous = ind$id[vous]
declar = rename(declar,c("idfoy"="id"))

ind$foy = as.integer(foyind$idfoy)
ind$quifoy = as.integer(foyind$quifoy)

# modif pour cohérence entre conjoint
# remarque: on perd probablement ici la cohérence entre pere et mere du coup...
ind$etamatri = as.integer(foyind$etamatri)
ind$sexe = as.integer(foyind$sexe)

