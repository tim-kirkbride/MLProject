ks <- read_csv("ks-projects-201801.csv")
ks_filt<-ks%>%filter(state %in% c("successful","failed"))
ks_filt$ID<-NULL
ks_filt$name<-NULL
ks_filt$`usd pledged`<-NULL
ks_filt$pledged<-NULL
ks_filt$goal<-NULL
ks_filt$country <- ifelse(ks_filt$country == "N,0\"", NA, ks_filt$country)
sum(complete.cases(ks_filt))/nrow(ks_filt)
ks_filt<-ks_filt%>%na.omit()
ks_filt$launched <- format(as.POSIXct(ks_filt$launched,format='%Y/%m/%d %H:%M:%S'),format='%Y/%m/%d')
ks_filt$launched<-ks_filt$launched%>%as.Date()
ks_filt$duration<-as.numeric(ks_filt$deadline-ks_filt$launched)
ks_filt$launched_month<-as.factor(month(ymd(ks_filt$launched)))
ks_filt$launched_year<-as.factor(year(ymd(ks_filt$launched)))
ks_filt$launched<-NULL
ks_filt$deadline<-NULL
ks_filt$category<-NULL
ks_filt[sapply(ks_filt, is.character)] <- lapply(ks_filt[sapply(ks_filt, is.character)], as.factor)