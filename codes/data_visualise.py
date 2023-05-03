import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler,PowerTransformer
import add_steps
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN




class data_:
	
	
	
	def read_file(self,filepath):
		data = arff.loadarff(str(filepath))
		df = pd.DataFrame(data[0])
		le = LabelEncoder()
		for col in df.select_dtypes(['object', 'category']).columns:
			df[col] = le.fit_transform(df[col])
		return pd.DataFrame(data[0])

	def convert_category(self,df,column_name):

		le=LabelEncoder()
		df[column_name] =le.fit_transform(df[column_name])
		return df[column_name],"LabelEncoder()"
	
	def get_column_list(self,df):

		column_list=[]

		for i in df.columns:
			column_list.append(i)
		return column_list

	def get_empty_list(self,df):

		empty_list=[]

		for i in df.columns:
			if(df[i].isnull().values.any()==True):
				empty_list.append(i)
		return empty_list

	def get_shape(self,df):
		return df.shape 


	def fillna(self,df,column):

		
		df[column].fillna("Uknown",inplace=True)
		return df[column]

	def fillmean(self,df,column):
		
		
		df[column].fillna(df[column].mean(),inplace=True)
		return df[column]

	def drop_columns(self,df,column):
		return df.drop(column,axis=1)

	def get_numeric(self,df):
		numeric_col=[]
		for i in df.columns:
			if(df[i].dtype!='object'):
				numeric_col.append(i)
		return numeric_col
	def get_cat(self,df):
		cat_col=[]
		for i in df.columns:
			if(df[i].dtype=='object'):
				cat_col.append(i)
		return cat_col

	def get_describe(self,df):

		return str(df.describe())
	
	def StandardScale(self,df):
		
		scaler = StandardScaler()
		data1 = df.copy()
		data1[df.select_dtypes(include="number").columns] = scaler.fit_transform(df.select_dtypes(include="number"))
		data = data1[df.select_dtypes(include="number").columns]
		
		return data,"StandardScaler()"

	def MinMaxScale(self,df):
		
		scaler = MinMaxScaler()
		data1 = df.copy()
		data1[df.select_dtypes(include="number").columns] = scaler.fit_transform(df.select_dtypes(include="number"))
		data = data1[df.select_dtypes(include="number").columns]
		return data,"MinMaxScaler()"
		
	def PowerScale(self,df,target):
		
		sc=PowerTransformer()
		x=df.drop(target,axis=1)
		scaled_features=sc.fit_transform(x)
		scaled_features_df = pd.DataFrame(scaled_features, index=x.index, columns=x.columns)
		scaled_features_df[target]=df[target]
		return scaled_features_df,"PowerTransformer()"

	def elbow_(self,data):
            # Initialiser une liste vide pour stocker les sommes des carrés des distances
			sse = []

            # Itérer sur différents nombres de clusters
			for k in range(1, 11):
				kmeans = KMeans(n_clusters=k, max_iter=1000)
				kmeans.fit(data)
				sse.append(kmeans.inertia_)

			# Tracer la courbe d'Elbow
			plt.plot(range(1, 11), sse)
			plt.title("Courbe d'Elbow")
			plt.xlabel("Nombre de clusters")
			plt.ylabel("Somme des carrés des distances")
			plt.show()
	    
	def elbow_2(self,data):
        # define a range of number of clusters to test
		k_range = range(2, 10)

		# initialize an empty list to store the sum of squared distances for each k
		ssd = []

		# iterate over the range of k values and compute the sum of squared distances for each k
		for k in k_range:
			kmedoids = KMedoids(n_clusters=k, metric='euclidean', random_state=42)
			kmedoids.fit(data)
			ssd.append(kmedoids.inertia_)

		# plot the sum of squared distances as a function of the number of clusters
		plt.plot(k_range, ssd, 'bx-')
		plt.xlabel('Number of clusters (k)')
		plt.ylabel('Sum of squared distances (SSD)')
		plt.title('Elbow Method for K-Medoids')
		plt.show()


	def plot_heatmap(self,df):
		plt.figure()
		x=df.corr()
		mask = np.triu(np.ones_like(x, dtype=np.bool))
		sns.heatmap(x,annot=True,mask=mask,vmin=-1,vmax=1)
		plt.show()

	def kmeans_plot(self,df,x,y,k,marker):
		# create the k-means object
		kmeans = KMeans(n_clusters=int(k), init='k-means++', random_state=42)

		# fit the model to the data
		kmeans.fit(df)

		# retrieve the labels for each data point
		labels = kmeans.labels_
		plt.figure()
		#plt.scatter(df[x],df[y],c=c,marker=marker)
		plt.scatter(df[x], df[y], c=kmeans.labels_ ,marker=marker)
		plt.xlabel(x)
		plt.ylabel(y)
		plt.title(y + " vs "+ x)
		plt.show()

	def kmedoid_plot(self,df,x,y,k,marker):
		# create the k-means object
		kmedoids = KMedoids(n_clusters=int(k), metric='euclidean', random_state=42)

		# fit the model to the data
		kmedoids.fit(df)

		# retrieve the labels for each data point
		labels = kmedoids.labels_
		plt.figure()
		#plt.scatter(df[x],df[y],c=c,marker=marker)
		plt.scatter(df[x], df[y], c=kmedoids.labels_ ,marker=marker)
		plt.xlabel(x)
		plt.ylabel(y)
		plt.title(y + " vs "+ x)
		plt.show()



	def agnes_plot(self,df,k):
		agg = AgglomerativeClustering(n_clusters=int(k), linkage='ward')
		agg.fit(df)
		agg_labels = agg.labels_
		agg_dist = shc.distance.pdist(df, metric='euclidean')
		agg_linkage = shc.linkage(agg_dist, method='ward')
		# Create the dendrogram for Agnes
		plt.figure(figsize=(10, 7))
		plt.title("Dendrogramme d'AGNES")
		dend = shc.dendrogram(agg_linkage, labels=agg_labels)
		plt.show()

	def diana_plot(self,df,k):
		diana = AgglomerativeClustering(n_clusters=8, linkage='single')
		diana.fit(df)
		diana_labels = diana.labels_
		# Calculate the distance matrix using complete linkage
		diana_dist = shc.distance.pdist(df, metric='euclidean')
		diana_linkage = shc.linkage(diana_dist, method='complete')
		# Create the dendrogram for DIANA
		plt.figure(figsize=(10, 7))
		plt.title("Dendrogramme de DIANA")
		dend = shc.dendrogram(diana_linkage, labels=diana_labels)
		plt.show()

	def btn_1(self,df,k):
		# create the k-means object
		kmeans = KMeans(n_clusters=int(k), init='k-means++', random_state=42)
		# fit the model to the data
		kmeans.fit(df)
		# calculate the inter-cluster distance
		centroids = kmeans.cluster_centers_
		inter_dist = pairwise_distances(centroids).sum()
		return int(kmeans.inertia_), int(inter_dist)
	
	def btn_2(self,df,k):
		# create the k-means object
		kmedoids = KMedoids(n_clusters=int(k), metric='euclidean', random_state=42)
		# fit the model to the data
		kmedoids.fit(df)
		# retrieve the labels for each data point
		labels = kmedoids.labels_
		# calculate the inter-cluster distance
		centroids = kmedoids.cluster_centers_
		inter_dist = pairwise_distances(centroids).sum()

		# print the intra- and inter-cluster distances
		return int(kmedoids.inertia_), int(inter_dist)

	def btn_3(self,df,k):
		agg = AgglomerativeClustering(n_clusters=int(k), linkage='ward')
		agg.fit(df)
		agg_labels = agg.labels_
		agg_dist = shc.distance.pdist(df, metric='euclidean')
		agg_linkage = shc.linkage(agg_dist, method='ward')


		clusters = np.unique(agg_labels)
		intra_distances = []
		for cluster in clusters:
			cluster_points = df[agg_labels == cluster]
			centroid = np.mean(cluster_points, axis=0)
			distance = np.sum((cluster_points - centroid)**2)
			intra_distances.append(distance)

		# calculate the interclass distance between each pair of clusters
		# Calculate the centroids of each cluster
		centroids = []
		for label in set(agg_labels):
			cluster_points = df[agg_labels == label]
			centroid = cluster_points.mean(axis=0)
			centroids.append(centroid)

		# Calculate the pairwise distances between centroids
		interclass_distances = []
		for i in range(len(centroids)):
			for j in range(i+1, len(centroids)):
				distance = pairwise_distances([centroids[i]], [centroids[j]])[0,0]
				interclass_distances.append(distance)

		# Calculate the total interclass distance
		total_interclass_distance = sum(interclass_distances)


		return int(sum(sum(intra_distances))), int(total_interclass_distance)
	

	def btn_4(self,df,k):
		diana = AgglomerativeClustering(n_clusters=int(k), linkage='single')
		diana.fit(df)
		diana_labels = diana.labels_
		clusters = np.unique(diana_labels)
		# Calculate the distance matrix using complete linkage
		diana_dist = shc.distance.pdist(df, metric='euclidean')
		diana_linkage = shc.linkage(diana_dist, method='complete')

		intra_distances = []
		for cluster in clusters:
			cluster_points = df[diana_labels == cluster]
			centroid = np.mean(cluster_points, axis=0)
			distance = np.sum((cluster_points - centroid)**2)
			intra_distances.append(distance)

		# calculate the interclass distance between each pair of clusters
		# Calculate the centroids of each cluster
		centroids = []
		for label in set(diana_labels):
			cluster_points = df[diana_labels == label]
			centroid = cluster_points.mean(axis=0)
			centroids.append(centroid)

		# Calculate the pairwise distances between centroids
		interclass_distances = []
		for i in range(len(centroids)):
			for j in range(i+1, len(centroids)):
				distance = pairwise_distances([centroids[i]], [centroids[j]])[0,0]
				interclass_distances.append(distance)

		# Calculate the total interclass distance
		total_interclass_distance = sum(interclass_distances)

		return int(sum(sum(intra_distances))), int(total_interclass_distance)
	



	def perf_plot(self,df, min_pts, epsilon, range1):
		df = df.iloc[:, :-1].values
		min_pts = range(1, int(min_pts))		
		epsilon_value = float(epsilon)
		epsilon = np.linspace(0.05, epsilon_value, int(range1))
		def cluster_performances(df, min_pts, epsilon):
			performances = np.zeros((len(min_pts), len(epsilon)))
			for i, min_pts in enumerate(min_pts):
				for j, eps in enumerate(epsilon):
					dbscan = DBSCAN(eps, min_samples=min_pts)
					dbscan.fit(df)
					performances[i, j] = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
			return performances
		# Compute the cluster performances
		performances = cluster_performances(df, min_pts, epsilon)
		# Plot the cluster performances
		fig, ax = plt.subplots()
		cax = ax.imshow(performances, interpolation='nearest', cmap='inferno')
		ax.set_xticks(np.arange(len(epsilon)))
		ax.set_yticks(np.arange(len(min_pts)))
		ax.set_xticklabels(epsilon)
		ax.set_yticklabels(min_pts)
		plt.xlabel('Epsilon')
		plt.ylabel('MinPts')
		cbar = fig.colorbar(cax)
		cbar.ax.set_ylabel('Number of clusters', rotation=270)
		plt.title('the cluster performances for different values of min_pts and epsilon')
		plt.show()




	def dbscan_plot(self,df, min_pts, epsilon, range1):
		df = df.iloc[:, :-1].values
		min_pts = range(1, int(min_pts))		
		epsilon_value = float(epsilon)
		epsilon = np.linspace(0.05, epsilon_value, int(range1))
		def cluster_performances(df, min_pts, epsilon):
			performances = np.zeros((len(min_pts), len(epsilon)))
			for i, min_pts in enumerate(min_pts):
				for j, eps in enumerate(epsilon):
					dbscan = DBSCAN(eps, min_samples=min_pts)
					dbscan.fit(df)
					performances[i, j] = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
			return performances
		# Compute the cluster performances
		performances = cluster_performances(df, min_pts, epsilon)
		
		# Choose the best hyperparameters based on the performances
		min_pts_idx, eps_idx = np.unravel_index(np.argmax(performances), performances.shape)
		best_min_pts = min_pts[min_pts_idx]
		best_eps = epsilon[eps_idx]
		# Cluster the data using the best hyperparameters and plot the resulting clusters
		dbscan = DBSCAN(eps=best_eps, min_samples=best_min_pts)
		dbscan.fit(df)
		unique_labels = set(dbscan.labels_)
		colors = [plt.cm.Spectral(each)
				for each in np.linspace(0, 1, len(unique_labels))]
		for k, col in zip(unique_labels, colors):
			if k == -1:
				# Black used for noise.
				col = [0, 0, 0, 1]

			class_member_mask = (dbscan.labels_ == k)

			xy = df[class_member_mask]
			plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
					markeredgecolor='k', markersize=6)

		plt.title('DBSCAN clustering with best hyperparameters (eps={}, min_pts={})'.format(best_eps, best_min_pts))
		plt.show()

