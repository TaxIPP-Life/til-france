################ merge link in table person #############

library(foreign)
link <- read.csv(paste0("lien_parent_enfant/match_",method_link,".csv"))
link = subset(link, select = c(id_origin,link_pere,link_mere))

# test = merge(person,link,by.x="id",by.y="id_origin", all=TRUE)
# prob = which( (test[,"mere"]>0 & test[,"link_mere"]>0 ) |  (test[,"pere"]>0 & test[,"link_pere"]>0 ) )

person = merge(person,link,by.x="id",by.y="id_origin", all=TRUE)
new_mother =  which( person[,"link_mere"]>0 )
which(person[new_mother,"mere"]>0)
person[new_mother,"mere"] = person[new_mother,"link_mere"]
new_father =  which( person[,"link_pere"]>0 )
which( person[new_father,"pere"]>0)
person[new_father,"pere"] = person[new_father,"link_pere"]

person = subset(person, select = -c(link_pere, link_mere))

rm(link, new_mother, new_father)
gc()

