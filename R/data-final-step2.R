require(foreign)
require(data.table)
require(SDMTools)

# ---- loading data -----

NREPS = 100
load("data-jmp-step1.dat") 

rdata[,lwr:=logrealmanadslon]
rdata = rdata[female==0][age>=20][age<=50]
smoms = data.frame()
set.seed(8972354)

# --- quarterly mobility -----
setkey(rdata,wid,aret,quarter)
rdata[,time := quarter + 4*(aret - min(aret))]
setkey(rdata,wid,time)


# compute a probability
mprob <- function(D,C,name) {
  p = sum(D & C)/sum(C)
  r = data.frame( moment = 'prob' , value=p,sd=p*(1-p)/sqrt(sum(C)))
  r$name = name
  r$N = length(D)
  return(r)
}

mv <- function(v,name) {
  sfit1 = summary(lm(v ~1))
  dev = abs(v - sfit1$coef[1,1])^2
  sfit2 = summary(lm(dev ~1))
  r = data.frame( moment = 'mean' , value=sfit1$coef[1,1],sd=sfit1$coef[1,2])
  r = rbind(r,data.frame( moment = 'var' ,  value=sfit2$coef[1,1],sd=sfit2$coef[1,2]))
  r$name=name
  return(r)
}

pr <- function(A,B) sum(A & B)/sum(B)


# ----- computing first set of moments ------

ff0 <- function(rdata) {
  rdata[,list( 
    u    = pr(from %in% c(4,5), from %in% c(1,2,3,4,5)),
    u2e  = pr(from==3,from%in%c(3,4)),
    j2j  = pr(from==2,from%in%c(1,2,5)),
    e2u  = pr(from==5,from%in%c(1,2,5)),
    na2a = pr(is.na(from),rep(1,.N))
  )]
}


mval = ff0(rdata)

# re-sample at the individual level
rep_all = data.frame()
for (i in 1:NREPS) {
  wids = rdata[,unique(wid)]
  wids = sample(wids,length(wids),replace=T)
  rdata_rs = data.table(wid=1:length(wids),wid_true=wids)
  setkey(rdata_rs,wid_true)
  setkey(rdata,wid)
  rdata_rs = rdata[,list(from,wid)][rdata_rs]
  rep_all = rbind(rep_all,data.frame(ff0(rdata_rs)))
} 
rep_all = data.table(melt(rep_all))
rep_stat = rep_all[, list(m=mean(value),sd=sd(value)),list(variable)]
rep_stat$main = as.numeric(as.list(mval)[rep_stat$variable])
smoms = rbind(smoms,rep_stat)


# ----- constructing yearly data --------
# keep only workers fully employed workers in each year
yspells = rdata[,list(.N,nm=sum(monthsworked),nm2=sum(monthsworked,na.rm=T),ne=sum(employed),nf=length(unique(fid)),fid=fid[1]),list(wid,aret)]

# attach lag month worked
setkey(yspells,wid,aret)
yspells[,nm.l1:=yspells[J(wid,aret-1),nm2]]
yspells = yspells[!is.na(nm)][nm==12][nf==1][,list(wid,fid,aret,nm.l1)]

# nf=1 for sure because nm==12!
setkey(yspells,wid,fid,aret)
setkey(rdata,wid,fid,aret)
datay   = rdata[quarter==2][yspells] # take the second quarter, they should all be indentical
datay[,lw:=lwr] # using residual wages

# append lag wages
setkey(datay,wid,aret)
datay[,fid.l1:=datay[J(wid,aret-1),fid]]
datay[,lw.l1 :=datay[J(wid,aret-1),lw]]
datay[,fid.l2:=datay[J(wid,aret-2),fid]]
datay[,lw.l2 :=datay[J(wid,aret-2),lw]]
datay[,fid.l3:=datay[J(wid,aret-3),fid]]
datay[,lw.l3 :=datay[J(wid,aret-3),lw]]
datay[,fid.l4:=datay[J(wid,aret-4),fid]]
datay[,lw.l4 :=datay[J(wid,aret-4),lw]]

# append lag productity
datay[,lp :=lprodr]
datay[,lp.l1:=datay[J(wid,aret-1),lprodr]]
datay[,lp.l2:=datay[J(wid,aret-2),lprodr]]
datay[,lp.l3:=datay[J(wid,aret-3),lprodr]]
datay[,lp.l4:=datay[J(wid,aret-4),lprodr]]


ff <- function(datay) {
  smoms = list()
  # compute growth covariance matrices for stayers
  M = datay[is.finite(lw*lw.l1*lw.l2*lw.l3),var(cbind(lw-lw.l1,lw.l1 - lw.l2, lw.l2 - lw.l3))]
  
  # extract and MA-2 process
  r =  mean(M[ row(M) == col(M) +1]) /  mean(M[ row(M) == col(M) +2])
  a = - 1; b = 2 +r; c=-1
  theta   = (-b +  sqrt(b^2 - 4*a*c))/(2*a)
  var_eps =   - mean(M[ row(M) == col(M) +2]) / theta
  var_mu  = mean(diag(M)) - var_eps * (1+theta^2 + (1-theta)^2)
  
  smoms$w_var_eps = var_eps
  smoms$w_var_mu  = var_mu
  smoms$w_theta   = theta
  smoms$dw_ac_0 =  mean(M[ row(M) == col(M) +0])
  smoms$dw_ac_1 =  mean(M[ row(M) == col(M) +1])
  smoms$dw_ac_2 =  mean(M[ row(M) == col(M) +2])
  
  # wage growth
  smoms$dw_mean     = datay[is.finite(lw*lw.l1),mean(lw - lw.l1)]
  smoms$dw_mean_2   = datay[is.finite(lw*lw.l2),mean(lw - lw.l2)]
  smoms$dw_mean_pos = datay[is.finite(lw*lw.l2),mean(lw >= lw.l2)]
  
  # wage growth J2J
  smoms$dw_j2j_2   = datay[is.finite(lw*lw.l2)][fid !=fid.l2,mean(lw - lw.l2)]
  smoms$dw_j2j_pos = datay[is.finite(lw*lw.l2)][fid !=fid.l2,mean(lw >= lw.l2)]
  
  # level
  smoms$w_mean     = datay[is.finite(lw)][,mean(lw)]
  smoms$w_mean_u2e = datay[is.finite(lw)][nm.l1==0][,mean(lw)]
  smoms$w_u2e_ee_gap = smoms$w_mean - smoms$w_mean_u2e
  smoms$w_var_u2e  = datay[is.finite(lw)][nm.l1==0][,var(lw)]
  
  # long run covariance in levels
  # datay[is.finite(lw*lw.l1*lw.l2*lw.l3*lw.l4),var(cbind(lw,lw.l1,lw.l2,lw.l3,lw.l4))]
  M = datay[is.finite(lw*lw.l4),var(cbind(lw,lw.l4))]
  smoms$w_long_ac   = M[1,2]
  smoms$w_var       = datay[is.finite(lw),var(lw)]
  
  return(smoms)
}

# get value on data
main = ff(datay)

# resample at the worker level
df_all = data.frame()
for (i in 1:NREPS) {
  wids = datay[,unique(wid)]
  wids = sample(wids,length(wids),replace=T)
  datay_rs = data.table(wid=1:length(wids),wid_true=wids)
  setkey(datay_rs,wid_true)
  setkey(datay,wid)
  datay_rs = datay[datay_rs]
  df = data.frame(ff(datay_rs))
  df_all = rbind(df_all,df)
}
df_all_m = data.table(melt(df_all))
df_stat = df_all_m[,list(m=mean(value),sd=sd(value)),list(variable)]
df_stat$main = as.numeric(as.list(main)[df_stat$variable ])
smoms = rbind(smoms,df_stat)

# ------- pass-through analysis ------

# get stayers
datays = datay[fid==fid.l1][fid==fid.l2][fid==fid.l3][fid==fid.l4]

ff2 <- function(datays) {
  smoms = list()
  M = datays[is.finite(lp*lp.l1*lp.l2*lp.l3*lp.l4),var(cbind(lp-lp.l1,lp.l1 - lp.l2, lp.l2 - lp.l3, lp.l3 - lp.l4))]
  
  r =  mean(M[ row(M) == col(M) +1]) /  mean(M[ row(M) == col(M) +2])
  a = - 1; b = 2 +r; c=-1
  theta   = (-b +  sqrt(b^2 - 4*a*c))/(2*a)
  var_eps =   - mean(M[ row(M) == col(M) +2]) / theta
  var_mu  = mean(diag(M)) - var_eps * (1+theta^2 + (1-theta)^2)
  
  smoms$dy_ac_0 =  mean(M[ row(M) == col(M) +0])
  smoms$dy_ac_1 =  mean(M[ row(M) == col(M) +1])
  smoms$dy_ac_2 =  mean(M[ row(M) == col(M) +2])
  smoms$y_var_eps = var_eps
  smoms$y_var_mu  = var_mu
  smoms$y_theta   = theta
  
  # covariance
  S1 = datays[is.finite(lp*lp.l1*lp.l2*lp.l3*lp.l4), cov(cbind( lp - lp.l1, lw - lw.l1  ))]
  smoms$dwdy_cov  = S1[1,2]
  
  # MA(0) IV
  S1 = datays[is.finite(lp*lp.l1*lp.l2*lp.l3*lp.l4), cov(cbind( lp.l1 - lp.l2, lp - lp.l3  ))]
  S2 = datays[is.finite(lp*lp.l1*lp.l2*lp.l3*lp.l4), cov(cbind( lp.l1 - lp.l2, lw - lw.l3  ))]
  smoms$gamma_iv_ma0 = S2[1,2]/S1[1,2]
  
  return(smoms)
}

main = ff2(datay)

df_all = data.frame()
for (i in 1:NREPS) {
  fids = datays[,unique(fid)]
  fids = sample(fids,length(fids),replace=T)
  datays_rs = data.table(fid=1:length(fids),fid_true=fids)
  setkey(datays_rs,fid_true)
  setkey(datays,fid)
  datays_rs = datays[datays_rs,allow.cartesian=TRUE]
  df = data.frame(ff2(datays_rs))
  df_all = rbind(df_all,df)
}
df_all_m = data.table(melt(df_all))
df_stat = df_all_m[,list(m=mean(value),sd=sd(value)),list(variable)]
df_stat$main = as.numeric( as.list(main) [df_stat$variable]  )
smoms = rbind(smoms,df_stat)

# ----------- extract next period mobility ----------

setkey(rdata,wid,time)
rdata[, from.f1 := rdata[J(wid,time-1),from]]
dd = rdata[, list(.N,e2u = sum(from.f1 == 5,na.rm=T), j2j= sum(from.f1 == 2,na.rm=T), lprodr = lprodr[1]),list(fid,aret) ]
dd[is.finite(lprodr),cov(cbind(lprodr, e2u,j2j))]

# create growth measure
setkey(dd,fid,aret)
dd[, dlprodr := lprodr -  dd[J(fid,aret-1), lprodr]]
dd[, dle2u := log(e2u/N) -  dd[J(fid,aret-1), log(e2u/N)]]
dd[, dlj2j := log(j2j/N) -  dd[J(fid,aret-1), log(j2j/N)]]
dd[, dlsep := log((j2j +e2u)/N) -  dd[J(fid,aret-1), log((j2j +e2u)/N)]]
dd = dd[fid!=""]

# bootstrap at the firm level
ffsep <- function(ddl) {
  cc = ddl[is.finite(dlprodr*dlj2j*dle2u),cov(cbind(dlprodr, dle2u,dlj2j,dlsep))]
  smoms = list()
  smoms$cov_dydlsep = cc[1,4]
  smoms$cov_dydle2u = cc[1,2]
  smoms$cov_dydlj2j = cc[1,3]
  smoms
}

main = ffsep(dd)

df_all = data.frame()
for (i in 1:NREPS) {
  fids = dd[,unique(fid)]
  fids = sample(fids,length(fids),replace=T)
  dd_rs = data.table(fid=1:length(fids),fid_true=fids)
  setkey(dd_rs,fid_true)
  setkey(dd,fid)
  dd_rs = dd[dd_rs,allow.cartesian=TRUE]
  df = data.frame(ffsep(dd_rs))
  df_all = rbind(df_all,df)
  cat(sprintf("done with %i\n",i))
}
df_all_m = data.table(melt(df_all))
df_stat = df_all_m[,list(m=mean(value),sd=sd(value)),list(variable)]
df_stat$main = as.numeric( as.list(main) [df_stat$variable]  )


smoms = rbind(smoms,df_stat)

# ---------- final save -------

write.csv(smoms, file= paste(RESULTS_PATH,'moments-jmp-2022.csv',sep='') )



# ---- DATA STATS ------
# get yearly data from rdata

rr = list()
rr$rdata_nwid     = rdata[,length(unique(wid))]   # unique worker
rr$rdata_nfid     = rdata[fid!="",length(unique(fid))] 
rr$rdata_employed = rdata[,mean(employed==1)]
rr$rdata_firm_median_reported_size_worker = rdata[,median(ant_anst,na.rm=T)]
rdata[,asize := .N,list(fid,aret,quarter)]
rr$rdata_firm_median_actual_size_worker = rdata[,median(asize,na.rm=T)]

rr$rdata_employed_month = rdata[,sum(monthsworked,na.rm=T)]/(3*rdata[,.N])
rr$worker_share_age_0to30   = rdata[,mean(age<=30)] 
rr$worker_share_age_31to40  = rdata[,mean(age>=31 & age<=40)] 
rr$worker_share_age_41toInf = rdata[,mean(age>41)] 

rr$worker_share_ind_manu     = rdata[,mean(industry=="Manufacturing",na.rm=T)]
rr$worker_share_ind_serv     = rdata[,mean(industry=="Services",na.rm=T)]
rr$worker_share_ind_retail   = rdata[,mean(industry=="Retail trade",na.rm=T)]
rr$worker_share_ind_cons     = rdata[,mean(industry=="Construction etc.",na.rm=T)]

rr$worker_share_higheduc =rdata[,mean(educ==3)] 


# aggregate 
yeardata = rdata[, list(mw = sum(monthsworked),
                        educ = max(educ),
                        age=min(age),
                        industry=industry[1],
                        lw = mean(logrealmanadslon), 
                        valueadded=mean(valueadded), 
                        employed=mean(employed)), list(wid,fid,aret)]

rr$ydata_nwid     = yeardata[,length(unique(wid))]   # unique worker
rr$ydata_nfid     = yeardata[fid!="",length(unique(fid))] 
rr$ydata_nyearobs = yeardata[,.N]
rr$ydata_nyearobs_ft = yeardata[mw==12,.N]

rr$worker_share_higheduc_ft =yeardata[mw==12,mean(educ==3)] 

rr$ydata_mean_wage_ft = yeardata[mw==12,mean(lw)]
rr$ydata_sd_wage_ft = yeardata[mw==12,sd(lw)]

for (i in 1:12) {
  rr[paste0("ydata_nyearobs_mw",i)] = yeardata[mw==i,.N]
}

write.csv(smoms, file= paste(RESULTS_PATH,'data-stats.csv',sep='') )
