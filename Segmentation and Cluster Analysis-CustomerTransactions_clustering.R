# CUSTOMER TRANSACTION DATA / CLUSTERING
# Gülhan Damla Aşık

getwd()
setwd("C:/Users/user/.../2- Adult Income Dataset R_glm")

library(dplyr)
library(ggplot2)
library(gplots)
library(gapminder)
library(reshape2)
library(tidyverse)
library(moments) #Skewness and Kurtosis
library(caTools)    #Split data
library(rlang)
library(GGally)
library(ROSE) # imbalanced data
library(caret) # Variable importance test
library(cluster)

CustTransactions <-read.csv("CustomerTransactions.csv",header=T)
View(CustTransactions)
head(CustTransactions)

CustTransactions <- CustTransactions[-1]
CustTransactions <- CustTransactions[-1]
head(CustTransactions)
glimpse(CustTransactions)

is.null(CustTransactions)
# FALSE
sum(complete.cases(CustTransactions))

cormat <- round(cor(CustTransactions))
head(cormat)
melted_cormat <- melt(cormat)
head(melted_cormat)
ggplot(data = melted_cormat, aes(x =Var1 , y =Var2, fill=value )) + 
  geom_tile(color="white") + 
  scale_fill_gradient2(low = "blue", high = "red", mid = "white" ,midpoint = 0, limit = c(-1,1), space = "Lab" , name="Pearson\nCorrelation") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1)) + 
  geom_text(aes(label = value), color = "black" , size = 4) +
  coord_fixed()
# 32-20 and 24-17 and 26-24 has correlation 1. 

glimpse(CustTransactions)
# Columns: 32
# Remove correlated columns
CustTransactionsNew <- CustTransactions[c("X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11",
                                          "X12","X13","X14","X15","X16","X17","X18","X19","X20","X21",
                                          "X22","X23","X25","X26","X27","X28","X29","X30","X31")]
glimpse(CustTransactionsNew)
# Columns: 30

# Histogram for each Attribute
CustTransactionsNew %>%
  gather(Attributes, value, 1:13) %>%
  ggplot(aes(x=value, fill=Attributes)) +
  geom_histogram(colour="black", show.legend=FALSE , binwidth = 1) +
  facet_wrap(~Attributes, scales="free_x") +
  labs(x="Values", y="Frequency",
       title=" - Histograms") +
  theme_bw()

############# “elbow criterion”  METHOD 1
wssplot <- function(CustTransactionsNew, nc=15, seed=1234)
{
  wss <- (nrow(CustTransactionsNew)-1)*sum(apply(CustTransactionsNew,2,var))
  for (i in 2:nc)
  {
    set.seed(seed)
    wss[i] <- sum(kmeans(CustTransactionsNew, centers=i)$withinss) 
  }
  plot(1:nc, wss, type="b", xlab="Number of Clusters",ylab="Within groups sum of squares")
} 

wssplot(CustTransactionsNew, nc=15)
# Looks like after 4th cluster, marginal gain drops.

############# “elbow criterion”  METHOD 2
set.seed(109)
wss <- 0
for (i in 1:30) {
  clust1 <- kmeans(CustTransactionsNew, centers = i, nstart = 20)
  wss[i] <- clust1$withinss}

wss_df <- data.frame(num_cluster = 1:30, wgss = wss)

ggplot(data = wss_df, aes(x=num_cluster, y= wgss)) + 
  geom_line(color = "lightgrey", size = 2) + 
  geom_point(color = "green", size = 5) + scale_x_continuous(limits = c(0,13)) +
  geom_text(aes(label = num_cluster), color = "black" , size = 4) +
  labs(title = "Clusters", subtitle = "Selecting the point where the elbow 'bends', or where the slope of \nthe Within groups sum of squares levels out")
# Again, k=4 is a better choice for clustering


# https://www.ibm.com/support/pages/clustering-binary-data-k-means-should-be-avoided
############################################### K-Means Cluster Analysis
fitkmeans1 <- kmeans(CustTransactionsNew, 4)
# get cluster means
aggregate(CustTransactionsNew,by=list(fitkmeans1$cluster),FUN=mean)
# append cluster assignment
CustTransactionskmeans <- data.frame(CustTransactionsNew, fitkmeans1$cluster)
head(CustTransactionskmeans)

# Let's check the quality of clusters.
# Cluster Cardinality  -> number of examples per cluster
table(CustTransactionskmeans$fitkmeans1.cluster)
# 1  2  3  4 
# 35 20 24 21
# Cluster number 1 should investigated. If we had more information about variables, we could have done more research.

fitkmeans1$betweenss
# 62.23929
# In an optimal segmentation, one expects this ratio to be as higher as possible, 
# since we would like to have heterogeneous clusters.

fitkmeans1$withinss
# 52.62857 54.40000 33.37500 64.85714
# In an optimal segmentation, one expects this ratio to be as lower as possible for each cluster,
# since we would like to have homogeneity within the clusters.


ggpairs(CustTransactionskmeans, columns=1:6, aes(colour=fitkmeans1$cluster, alpha=0.5),
        lower=list(continuous="points"),
        upper=list(continuous="blank"),
        axisLabels="none", switch="both") +
  theme_bw()

################# Aproach 2
library(cluster)
install.packages("fpc")
library(fpc)

data(CustTransactionskmeans)
dat <- CustTransactionskmeans[, -31] # without known classification 
# Kmeans cluster analysis
clus <- kmeans(dat, centers=4)
plotcluster(dat, clus$cluster)
# 4 cluster doesn't look good.

data(CustTransactionskmeans)
dat <- CustTransactionskmeans[, -31] # without known classification 
# Kmeans cluster analysis
clus <- kmeans(dat, centers=3)
plotcluster(dat, clus$cluster)
# 3 is better.


############################################### Hierarchical Clustering
# If all of the cluster variables are binary, then one can employ the distance measures for binary variables that are available for the Hierarchical Cluster. # IBM.com
#Find Hierarchical clustering using Euclidean distance and wards method in matrix.
distance <- dist(CustTransactionsNew, method = "euclidean") 
H_Model <- hclust(distance, method="ward.D")

# display dendogram
plot(H_Model, main="Hierarchical") 

# cut tree into 3 clusters
Cuthier4 <- cutree(H_Model, k=4)
Cuthier3 <- cutree(H_Model, k=3)

# show groups
CustTransactionsHier4 <- data.frame(CustTransactionsNew, Cuthier4)
head(CustTransactionsHier4)
table(CustTransactionsHier4$Cuthier4)
#  1  2  3  4 
# 32 31 16 21
# draw dendogram with red borders around the 3 clusters
rect.hclust(H_Model, k=4, border="red")

CustTransactionsHier3 <- data.frame(CustTransactionsNew, Cuthier3)
head(CustTransactionsHier3)
table(CustTransactionsHier3$Cuthier3)
#  1  2  3 
# 32 52 16


data(CustTransactionsHier3)
dat <- CustTransactionsHier3[, -31] # without known classification 
# Kmeans cluster analysis
clus <- kmeans(dat, centers=3)    
plotcluster(dat, CustTransactionsHier3$Cuthier3)
# 3 is better.


# If all of the cluster variables are binary, then one can employ the distance measures for binary variables that are available for the Hierarchical Cluster. # IBM.com
#Find Hierarchical clustering using Binary distance and wards method in matrix.
distancebinary <- dist(CustTransactionsNew, method = "binary")   # jaccard distance
H_Modelbinary <- hclust(distancebinary, method="ward.D")
?hclust
# display dendogram
plot(H_Modelbinary, main="Hierarchical") 

# cut tree into clusters
Cuthierbinary4 <- cutree(H_Modelbinary, k=4)
Cuthierbinary3 <- cutree(H_Modelbinary, k=3)

# show groups
CustTransactionsHierBinary4 <- data.frame(CustTransactionsNew, Cuthierbinary4)
head(CustTransactionsHierBinary4)
table(CustTransactionsHierBinary4$Cuthierbinary4)
#  1  2  3  4 
# 27 55 13  5
# draw dendogram with red borders around the 4 clusters
rect.hclust(H_Modelbinary, k=4, border="red") 

# show groups
CustTransactionsHierBinary3 <- data.frame(CustTransactionsNew, Cuthierbinary3)
head(CustTransactionsHierBinary3)
table(CustTransactionsHierBinary3$Cuthierbinary3)
#  1  2  3
# 32 55 13
# draw dendogram with red borders around the 3 clusters
rect.hclust(H_Modelbinary, k=3, border="red") 

data(CustTransactionsHierBinary3)
dat <- CustTransactionsHierBinary3[, -31] # without known classification 
plotcluster(dat, CustTransactionsHierBinary3$Cuthierbinary3)
# Looks ok.

