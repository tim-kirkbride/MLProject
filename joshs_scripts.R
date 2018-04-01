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

#proportions within continents
cont_state <- table(ks_filt$continent,ks_filt$state, dnn = c("Continent","State"))
prop.cont_state<-data.frame(prop.table(cont_state,1))
colnames(prop.cont_state) <- c("Continent","State","Proportion")

p6<-ggplot(prop.cont_state,aes(x=Continent,y=Proportion,fill=State))
p6+geom_bar(stat="identity",position="dodge")+
  labs(title="Proportions of Successful/Failed Projects within Project Continents")



ks_filt$duration<-as.numeric(ks_filt$deadline-ks_filt$launched)


#As the dates within the launched and deadline columns are specific to the day, they would not be appropriate for use as a categorical feature as there would be far too many levels. Also, with the addition of the duration feature to the dataset, the deadline feature may no longer be particularly valuable in a model. Considering this, it was decided to replace the launched feautre with two new columns - launched_month and launched_year - which would simply represent the month and year each project was launched, respectively. After this, the launched column would no longer be needed so it was dropped. Additionally, the deadline column would be dropped as its information would be captured by the new launched features and the duration feature.


ks_filt$launched_month<-as.factor(month(ymd(ks_filt$launched)))
ks_filt$launched_year<-as.factor(year(ymd(ks_filt$launched)))
ks_filt$launched<-NULL
ks_filt$deadline<-NULL
```
