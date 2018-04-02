#duration
p<-ggplot(ks_filt,aes(x=duration,fill=state))
p+geom_density()

#pledged
ks_pledged<-ks_filt%>%filter(usd_pledged_real<5000)
p1<-ggplot(ks_pledged,aes(x=usd_pledged_real,fill=state))
p1+geom_density(alpha=.4)



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

#backers
back_filt<-ks_filt%>%filter(backers<150)

backers <- ggplot(back_filt, aes(x = backers,fill=state))

backers + geom_density(alpha=.4) 

p7<-ggplot(back_filt,aes(x=backers))
p7+geom_histogram()

#pledged

pledge_filt<-ks_filt%>%filter(usd_pledged_real<4000)

pledged <- ggplot(pledge_filt, aes(x = usd_pledged_real,fill=state))

pledged + geom_density(alpha=.4) 

#big box
pledge_filt2<-ks_filt%>%filter(usd_pledged_real<7500)
p13<-ggplot(pledge_filt2,aes(x=continent,y=usd_pledged_real,fill=state))
p13+geom_boxplot()+
  labs(title = "Boxplot of Amount Pledged (USD) by Project State and Continent",
       x="Continent",
       y="Amount Pledged (USD)")

###

p13 <- ggplot(pledge_filt, aes(x = usd_pledged_real,fill=state))

p13 + geom_histogram() + 
  labs(title = "Density of Amount Pledged to Project (USD) by State",
       x = "Amount Pledged (USD)",
       y = "Density",
       fill = "State")



ks_filt$duration<-as.numeric(ks_filt$deadline-ks_filt$launched)


#As the dates within the launched and deadline columns are specific to the day, they would not be appropriate for use as a categorical feature as there would be far too many levels. Also, with the addition of the duration feature to the dataset, the deadline feature may no longer be particularly valuable in a model. Considering this, it was decided to replace the launched feautre with two new columns - launched_month and launched_year - which would simply represent the month and year each project was launched, respectively. After this, the launched column would no longer be needed so it was dropped. Additionally, the deadline column would be dropped as its information would be captured by the new launched features and the duration feature.


ks_filt$launched_month<-as.factor(month(ymd(ks_filt$launched)))
ks_filt$launched_year<-as.factor(year(ymd(ks_filt$launched)))
ks_filt$launched<-NULL
ks_filt$deadline<-NULL
```
