require(foreign)
require(data.table)
require(futile.logger)

#' loads the data for given gender 
#' and education. Reformat variables, 
#' and merge in the firm data.
loadData <- function(gender='male',educ=3,cached=TRUE,path="") {
  
  gender = ifelse(gender=='male',0,1)
  
  if (cached) {
    load(paste(path,'/selectedf',gender,'educ',educ,'.dat',sep=''))
    return(data)
  } 
  
  flog.info('data needs to be reloaded') 
  flog.info('loading file %s',paste(path,'/selectedf',gender,'educ',educ,'.dta',sep=''))
  data  <- data.table(read.dta(paste(path,'/selectedf',gender,'educ',educ,'.dta',sep='')))
  flog.info('data loaded')  
  
  # some renaming
  # --------------------
  data[ , fid := peorgnrdnr2008025]
  data[ , wid := dnr2008025]
  data[ , lw  := logrealmanadslon]
  flog.info(" nrows=%i nwids=%i nfids=%i", data[,.N] , data[,length(unique(wid))], data[,length(unique(fid))])
  
  # Merging firm data
  # -----------------
  # todo: count if some years-firm are missing, not just firms
  flog.info('merging firm information')  
  data2  <- data.table(read.dta(paste(path,'/selectedfirms9708.dta',sep='')))
  flog.info("total firms: %i",data2[,length(unique(peorgnrdnr2008025))])
  data2  <- subset(data2,valueadded>0 & ant_anst>0) 
  flog.info("total firms after: %i",data2[,length(unique(peorgnrdnr2008025))])
  data2[,prod  := valueadded/ant_anst]
  data2[,lprod := log(prod)]
  data2 = data2[is.finite(lprod)]
  flog.info("total firms after removing negative VA: %i",data2[,length(unique(peorgnrdnr2008025))])
  fit <- lm(lprod~factor(aret)*industry,data2,na.action=na.exclude)
  data2[ , lprodr := residuals(fit)]
  # removing time effects from prod
  data2[ , fid:= peorgnrdnr2008025]
  setkey(data2,fid,aret)
  
  # merging industry and prod data to worker data
  setkey(data,fid,aret)
  data2=data2[,list(fid,aret,prod,lprodr,valueadded,industry,ant_anst)] 
  setkey(data2,fid,aret)
  data = data2[data]
  #unique(data[employed==1][is.na(industry)][,list(fid,aret)])
  flog.info("share of observations employed without firm info %4.4f",data[employed==1][,mean(is.na(industry))])
  
  # preparing dummy to show if transition within year
  # compute dummy that states if worker is changing job within the
  # year
  flog.info('preparing some worker dummies')  
  setkey(data,wid,aret)
  dd.w = data[,list(jchange = length(unique(fid))>1),list(wid,aret)]
  setkey(dd.w,wid,aret)
  data = dd.w[data]
  data = data[,list(wid,fid,aret,quarter,birthyear,age,educ,female,from,jobmobility,jchange,kapink,monthsworked,employed,logrealmanadslon,industry,prod,lprodr,valueadded,ant_anst)]
  
  rm(data2)
  flog.info(" nrows=%i nwids=%i nfids=%i", data[,.N] , data[,length(unique(wid))], data[,length(unique(fid))])
  cat('saving the file for later use')    
  save(data,file=paste(path,'/selectedf',gender,'educ',educ,'.dat',sep=''))
  
  flog.info("done")
  
  return(data)
  #save(data,file='qtrdata2012/worker_process.dat')
}

# load stata files, merge them
rdata1 = loadData(gender="male",educ=1,path=DATA_PATH,cached=F)
rdata2 = loadData(gender="male",educ=2,path=DATA_PATH,cached=F)
rdata3 = loadData(gender="male",educ=3,path=DATA_PATH,cached=F)
rdata = rbind(rdata1,rdata2,rdata3)
rdata= rdata[aret %in% c(2001,2002,2003,2004,2005,2006)]
save(rdata,file = "data-jmp-step1.dat") 

