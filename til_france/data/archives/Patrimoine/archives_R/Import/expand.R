## expand

# on part pour l'instant des bases initiales, mais 
# on partira des bases avec les seules
# infos qui nous intéressent plus tard

gc()            # Garbage collecting (for memory efficiency)

library(plyr)
# TODO: faire des order, pour que ce soit sans risque

pond_unif = min(menage$pond)
pond_unif = max(150,pond_unif)
num_dup = 1+floor(menage$pond/pond_unif)
  # test surpond�ration: 
  surpond = which( min(menage$pond) < menage$pond & menage$pond < pond_unif )
  pond_unif * length(surpond)
  sum(menage$pond[surpond])
  ratio = 100*(pond_unif * length(surpond) - sum(menage$pond[surpond])) /  sum(menage$pond)
  ratio

### table ménage
taille.m = nrow(menage)
pond = menage$pond
menage$pond = menage$pond/num_dup
menage = menage[rep(1:nrow(menage), num_dup),]
menage$id_old = menage$id
menage$id = seq(1:nrow(menage))
  # test surpopulation dans Til
  100*(sum(nrow(menage)*pond_unif) - sum(menage$pond)) / sum(menage$pond)

### table foyer
#pour l'instant c'est exactement le meme format que menage
taille.f = nrow(declar)
num_declar_by_res = count(declar,vars="res")[,2]
num_dup.declar = 1+ floor(rep(pond,num_declar_by_res)/pond_unif) #pas besoin de pond dans la table declar
declar = declar[rep(1:nrow(declar), num_dup.declar),]
declar$res_old = declar$res
declar$id_old = declar$id
declar$id = seq(1:nrow(declar))

### table individu
num_ind_by_res = count(person,vars="res")[,2]
num_ind_by_foy = count(person,vars="foy")[,2]
num_declar_by_res = count(declar,vars="res_old")[,2]
num_dup.ind = rep.int(num_dup,num_ind_by_res)
person = person[rep(1:nrow(person), num_dup.ind),]

#on travaille sur les liens
name_replication = lapply(num_dup,seq)
name_replication = rep(name_replication,num_ind_by_res)
name_replication = unlist(name_replication)
id_replication = paste0(person$id,"_",name_replication)

conj2 = paste0(person$conj,"_",name_replication)
pere2 = paste0(person$pere,"_",name_replication)
mere2 = paste0(person$mere,"_",name_replication)
#nouvel id
person$id_old = person$id
person$id = seq(1:nrow(person))
#modif des liens
person$conj <- person$id[match(conj2,id_replication)]
person$pere <- person$id[match(pere2,id_replication)]
person$mere <- person$id[match(mere2,id_replication)]


# a partir de la, ca se complique
# on regarde les identifiants lies a id et res et on s'occuppe des
# tables pour les lien parents_enfants
## travail sur les res 
list_res_by_old = list()
list_res_by_old.parent = list()
list_res_old = unique(menage$id_old)
list_res_old.parent = unique(look_child$res)
num_parent = numeric(taille.m)
for (i in 1:taille.m) {
  list_res_by_old[[i]] = menage$id[which(menage$id_old == list_res_old[i])]
  if (list_res_old[i] %in% list_res_old.parent) {
    list_res_by_old.parent[[i]] = menage$id[which(menage$id_old == list_res_old[i])]  
    num_parent[i] = length( which(look_child$res == list_res_old[i] ))
  }
  if (i %% 1000 == 0) {
    print(paste("done ",i/1000, "over" , trunc(taille.m/1000)))
  }
} 
res.person = rep(list_res_by_old, num_ind_by_res)
res.person = unlist(res.person)
person$res <- res.person
res.parent = rep(list_res_by_old.parent ,num_parent[1:length(list_res_by_old.parent)])
res.parent = unlist(res.parent)

## travail sur les id
#id des enfants
list_child = list()
num_dup_child = numeric(nrow(look_parent))
for (i in 1:nrow(look_parent)) {  
  a = look_parent$id[i]
  temp= which(person$id_old == a)
  list_child[[i]] = person$id[temp]
  num_dup_child[i] = length(temp)
  if (i %% 1000 == 0) {
    print(paste("done ",i/1000, "over" , trunc(nrow(look_parent)/1000)))
  }
}
list_child = unlist(list_child)
#duplication de la table look_parent
look_parent = look_parent[rep(1:nrow(look_parent), num_dup_child),]
look_parent$id <- list_child

#id des peres et meres de la table look_child
list_meres = list()
list_peres = list()
for (i in 1:length(list_res_old.parent)) {
  pere = look_child$pere[which(look_child$res == list_res_old.parent[i])[1]]
  mere = look_child$mere[which(look_child$res == list_res_old.parent[i])[1]]
  #  print(paste(pere,mere))
  if (is.na(pere)) {
    meres = person$id[which(person$id_old == mere)] 
    list_meres[[i]] = meres      
    list_peres[[i]] = rep(NA,length(meres))
  }
  else if (is.na(mere)) {
    peres = person$id[which(person$id_old == pere)] 
    list_peres[[i]] = peres      
    list_meres[[i]] = rep(NA,length(peres))  
  }
  else {
    peres = person[ which(person$id_old == pere) , c("id","conj") ]  
    list_peres[[i]] = peres[,1]      
    list_meres[[i]] = peres[,2] 
  }
  if (i %% 1000 == 0) {
    print(paste("done ",i/1000, "over" , trunc(length(list_res_old.parent)/1000)))
  }
}
list_peres = rep(list_peres, num_parent[which(num_parent>0)])
list_peres = unlist(list_peres)
list_meres = rep(list_meres, num_parent[which(num_parent>0)])
list_meres = unlist(list_meres)

# replique les parents de look_child 
num_dup.look_child = rep(num_dup, num_parent)
# = num_dup.look_child = 1+floor(look_child$pond/pond_unif) 
look_child = look_child[rep(1:nrow(look_child), num_dup.look_child),]
look_child$res = res.parent
look_child$pere = list_peres
look_child$mere = list_meres

## travail sur les declar 

# pour foy, le plus simple est de travailler sur les vous puis d'étendre aux autres membres comme on le fait dans
# declar.R
vous = which(person$quifoy==0)
conj = which(person$quifoy==1)
pac = which(person$quifoy==2)

person$foy[vous] = seq(1,length(vous))
person$foy[conj] = person$foy[person$conj[conj]]
person$foy[pac] = ifelse( !is.na(person$pere[pac]),
                            person$foy[person$pere[pac]],
                            person$foy[person$mere[pac]])

# retour sur la table declar
declar$vous = vous
#TODO: un truc ne va pas les res de declar


# retour sur la table res
pref = person$id[which(person$quires==0)]
menage$pref = pref

# enfin la table retro est une table person
# mais pas sur d'en avoir besoin
retro = retro[rep(1:nrow(retro), num_dup.ind),]
retro$id = person$id
retro$res = person$res