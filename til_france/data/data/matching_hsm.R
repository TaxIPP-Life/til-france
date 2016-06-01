rm(list=ls())
gc()
devtools::install_github("wesm/feather/R")
library(feather)
library(StatMatch)

# Variables AGFINETU SITUA NPERS NENFANTS

patrimoine_path <- '/home/benjello/data/til/temp/patrimoine.feather'
patrimoine <- read_feather(patrimoine_path)
patrimoine$age = round(patrimoine$age_en_mois / 12)
patrimoine = subset(patrimoine, select = c(age, sexe))
patrimoine = patrimoine[sample(nrow(patrimoine), 12000), ]
patrimoine$sexe = as.factor(patrimoine$sexe)
patrimoine$age_group = round(patrimoine$age / 10)

hsm_path <- '/home/benjello/data/til/temp/hsm.feather'
hsm <- read_feather(hsm_path)
hsm$sexe = hsm$v_sexe - 1
hsm$sexe = ifelse(hsm$sexe >= 0, hsm$sexe, 0)
hsm$sexe = as.factor(hsm$sexe)
hsm = subset(hsm, select = c(age, ident_men, sexe))
summary(hsm)
hsm = na.omit(hsm)
hsm$ident_men = as.integer(hsm$ident_men)
hsm$age_group = round(hsm$age / 10)

hsm = subset(hsm, select = c(age_group, ident_men, sexe))
patrimoine = subset(patrimoine, select = c(age_group, sexe))
summary(hsm)
summary(patrimoine)

# variables 
patrimoine = as.data.frame(patrimoine)
hsm = as.data.frame(hsm)
gc()
match_vars = "age_group"
don_class = c("sexe")
out.nnd <- NND.hotdeck(
  data.rec=patrimoine, data.don=hsm, match.vars=match_vars
  )
out.nnd
summary(out.nnd$mtc.ids)

head(out.nnd$mtc.ids, 10)
head(patrimoine, 10)

fused.nnd.m <- create.fused(
    data.rec=patrimoine, data.don=hsm,
    mtc.ids=out.nnd$mtc.ids,
    z.vars="ident_men"
    )
summary(fused.nnd.m)

