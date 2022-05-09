library(ggplot2)
library(hrbrthemes)

dloss <- read.csv("~/Downloads/Archive 2 (1)/Syn_dloss_.csv")
View(dloss)
# Basic line plot with points
ggplot(data=dloss, aes(x=Step, y=Value, group=1)) +
geom_line( color="#69b3a2", size=1, alpha=0.9, linetype=1) +
theme_ipsum() +
ggtitle("Discriminator Loss")
gloss <- read.csv("~/Downloads/Archive 2 (1)/Syn_gLoss.csv")
View(gloss)
ggplot(data=gloss, aes(x=Step, y=Value, group=1)) +
geom_line( color="#69b3a2", size=1, alpha=0.9, linetype=1) +
theme_ipsum() +
ggtitle("Generator Loss")
syn_fk_discr <- read.csv("~/Downloads/Archive 2 (1)/syn_fk_discr.csv")
View(syn_fk_discr)
ggplot(data=Real_D_fk_discr, aes(x=Step, y=Value, group=1)) +
geom_line( color="#69b3a2", size=1, alpha=0.9, linetype=1) +
theme_ipsum() +
ggtitle("D(x)")
syn_g_fake <- read.csv("~/Downloads/Archive 2 (1)/syn_g_fake.csv")
View(syn_g_fake)
ggplot(data=fake_genr, aes(x=Step, y=Value, group=1)) +
geom_line( color="#69b3a2", size=1, alpha=0.9, linetype=1) +
theme_ipsum() +
ggtitle("D(G(z))")
