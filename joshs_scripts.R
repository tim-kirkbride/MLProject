#duration
p<-ggplot(ks_filt,aes(x=duration,fill=state))
p+geom_density()

#pledged
ks_pledged<-ks_filt%>%filter(usd_pledged_real<5000)
p1<-ggplot(ks_pledged,aes(x=usd_pledged_real,fill=state))
p1+geom_density(alpha=.4)

#currency
p2<-ggplot(ks_pledged,aes(x=currency,y=usd_pledged_real,fill=state))
p2+geom_boxplot()

#time (should remove year)
p3<-ggplot(ks_filt,aes(x=launched_month,fill=state))
p3+geom_bar(position = "dodge")


#scatter plot duration x pledged
p4<-ggplot(ks_filt,aes(x=usd_pledged_real,y=duration))
p4+geom_point(aes(fill=state))

#

#proportions within categories
cat_state <- table(ks_filt$main_category,ks_filt$state, dnn = c("Category","State"))
prop.cat_state<-data.frame(prop.table(cat_state,1))
colnames(prop.cat_state) <- c("Category","State","Proportion")

p5<-ggplot(prop.cat_state,aes(x=Category,y=Proportion,fill=State))
p5+geom_bar(stat="identity",position="dodge")+
  labs(title="Proportions of Successful/Failed Projects with Project Catgeories")
