clear all
version 14.1
set more off
capture log close

log using cr_nlsw.log, replace

webuse nlswork
mdesc ln_wage

fillin id year
mdesc ln_wage

egen grid = group(id)
drop idcode
ren grid idcode

bys id (year): egen t = seq()

gen Choice = 2-!mi(ln_wage)

xtset id year

gen working = Choice==1

drop ttl_exp age

bys id (year): gen exper = sum(L.working)
bys id (year): egen yrbornA = max(birth_yr)
bys id (year): egen yrborn = mean(yrbornA)
bys id (year): egen collgradA = max(collgrad)
bys id (year): egen raceA = max(race)

drop birth_yr collgrad race
ren yrborn birth_yr
ren collgradA collgrad
ren raceA race
gen race1 = race==1
gen age = year - birth_yr

recode ln_wage (. = 999)

keep id t Choice workin ln_wage exper race age collgrad race1
outsheet using nlswlearn.csv, comma nol replace

reg   ln_wage c.exper##c.exper i.collgrad race1 if Choice==1
xtreg ln_wage c.exper##c.exper i.collgrad race1 if Choice==1, fe
xtreg ln_wage c.exper##c.exper i.collgrad race1 if Choice==1, re

log close

