import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

"""Feature Engineering"""

#a) Load the California Housing dataset using
# sklearn.datasets.fetch_california_housing() as per usual.

housing = fetch_california_housing()
X = housing["data"]
y = housing["target"]

#b) Extract polynomial features (without bias!) and interactions up to a degree
# of 2 using PolynomialFeatures(). How many features do you end up with?

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
housing_poly = poly.fit_transform(X)
poly_features = poly.fit(X)
print("Number of output features:", poly_features.n_output_features_)
# 44 output features

#c) Create a pandas.DataFrame() using the polynomials. Use the originally provided feature names to generate names for
# the polynomials .get_feature_names_out() accepts a parameter) and use them ascolumn names. Also add the target
# variable to the data frame and name the column ”y”. Finally save it as comma-separated textfile
# named ./output/polynomials.csv.

poly_feature_names = \
    poly_features.get_feature_names(input_features=housing["feature_names"])
y = housing["target"]
df = pd.DataFrame(housing_poly, columns=poly_feature_names)
df["y"] = y
df.to_csv('./output/polynomials.csv', index=False)

"""2. Principal Component Analysis"""

#(a) Read the textfile, drop score?
data = pd.read_csv("./data/olympics.csv", index_col=0)

#Dropping score could make sense given the high variance, and possible
# multicolinearity

data = data.drop(["score"], axis=1)
X = data.loc[:, :]

#(b) Scale the data such that all variables have unit variance.
# Which pandas.DataFrame() method can you #use to assert that all variables
# have unit variance?

scaler = StandardScaler()
scaler.fit(X)
X_Scaled = scaler.transform(X)
X_Scaled.std(axis=0)

#(c) Fit a plain vanilla PCA model. Store the components in a
# pandas.DataFrame() to display the loadings of #each variable. Which variables
# load most prominently on the first component? Which ones on the second?
# Which ones on the third? How would you thus interpret those components?

pca = PCA(random_state=0)
pca.fit(X_Scaled)
sns.heatmap(pca.components_,
            xticklabels=data.columns)
df = pd.DataFrame(pca.components_, columns=data.columns)

# 100m Sprint (100) and 110m Hurdles (110) load the most on the first component
# Shot Put (poid) and Discus throw (disq) load the most on the second component
# High jump (haut) loads most prominently on the third component
# The first component is related to running, while the second about strength
# and the third about jump.

df = pd.DataFrame(pca.components_, columns=data.columns)

#(d) How many components do you need to explain at least 90% of the data?
df = pd.DataFrame(pca.explained_variance_ratio_,
                  columns=["Explained Variance"])
df["Cumulative"] = df["Explained Variance"].cumsum()
df.plot(kind="bar")

#It needs at least 6 components to explain at least 90% of the data

"""3.Clustering"""

#(a) Load the iris dataset using sklearn.datasets.load_iris().
# The data is on classifying flowers.

iris = load_iris()
X = iris["data"]
y = iris["target"]

#(b) Scale the data such that each variable has unit variance.

scaler = StandardScaler()
scaler.fit(X)
X_Scaled = scaler.transform(X)
X_Scaled.std(axis=0)

#(c) Assume there are three clusters. Fit a K-Means model, an Agglomerative
# Model and a DBSCAN model #(with min sample equal to 2 and epsilon equal to
# 1) with Euclidean distance. Store only the cluster assignments #in a
# new pandas.DataFrame().

    #K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_Scaled)
cluster = pd.DataFrame(kmeans.labels_, columns=["kmeansclustering"])

    # Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=2)
agg.fit(X_Scaled)
cluster["aggclustering"] = agg.labels_

    # DBSCAN
dbscan = DBSCAN(eps=1, min_samples=2)
dbscan.fit(X_Scaled)
cluster["dbscn"] = dbscan.labels_

#(d) Compute the silhouette scores using sklearn.metrics.silhouette_score()
# for each cluster algorithm from c). Why do you have to treat noise
# assignments from DBSCAN differently? Which model has the #highest Silhouette
# score?

cluster["dbscn"].value_counts()
# have one noisy observation (-1) which Python would think is an additional
# cluster. Thus, we drop it.
cclean = cluster.replace(-1, np.nan)
cclean["dbscn"].value_counts()

# Silhouette score (0 - +1)
print(silhouette_score(X, kmeans.labels_))
print(silhouette_score(X, agg.labels_))
print(silhouette_score(X, dbscan.labels_))

#K-Means has the greatest score 0.6867350732769776

#(e) Add variables ”sepal width” and ”petal length” including the corresponding
# column names to the pandas.DataFrame()that contains the cluster assignments.
# (Beware of the dimensionality!)

nv = X[:, 1:3]
columns = ["sepal width", "petal length"]
df = pd.DataFrame(nv, columns=columns)
dff = pd.concat([df, cclean], axis=1, sort=False)

#(f ) Rename noise assignments to ”Noise”.

df_melt = pd.melt(dff, id_vars=["petal length", "sepal width"],
                  value_vars=["kmeansclustering", "aggclustering", "dbscn"],
                  value_name="Cluster",
                  var_name="Cluster_Type")

#(g) Plot a three-fold scatter plot using ”sepal width” as x-variable and
# ”petal length” as y-variable, with dots #colored by the cluster assignment
# and facets by cluster algorithm. ( Melt the pandas.DataFrame() with above
# variables as ID variables.) Save the plot as ./output/cluster_petal.pdf.
# Does the noise #assignment make sense intuitively?

g = sns.FacetGrid(df_melt, col="Cluster_Type",  col_wrap=2, hue="Cluster")
g = g.map(plt.scatter, "petal length", "sepal width")
g.savefig("./output/cluster_petal.pdf")


