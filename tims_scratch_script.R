library(dplyr)

ks_filt <- ks_filt %>% mutate(
  pledge_per_backer = usd_pledged_real / backers
)

ppb <- ggplot(filter(ks_filt, pledge_per_backer > 0 & pledge_per_backer < 500), aes(x = pledge_per_backer))

ppb + geom_density()

students$entity <- case_when(
  students$`Broad Fund Code` == 'VIETNAM' ~ 'Vietnam',
  students$`Enrolled School Code` == "E-TRN" ~ 'Training',
  students$Group == 'Online_Grp' | students$Group == 'OUA_Grp' ~ 'Online',
  TRUE ~ 'RMIT_University'
)

countries <- ggplot(ks_filt, aes(x = country))

countries + geom_bar()

europe <- c("AT", "BE", "DE", "DK", "ES", "FR", "GB", "IE", "IT", "LU", "NL", "NO", "SE")
north_america <- c("CA", "MX", "US")
asia <- c("CH", "HK", "JP", "SG")
oceania <- c("AU", "NZ")

ks_filt$continent <- case_when(
  ks_filt$country %in% europe ~ "Europe",
  ks_filt$country %in% north_america ~ "North America",
  ks_filt$country %in% asia ~ "Asia",
  ks_filt$country %in% oceania ~ "Oceania"
)

cont <- ggplot(ks_filt, aes(x = continent))

cont + geom_bar(aes(fill = state), position = "dodge")
