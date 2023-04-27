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
	    
	def plot_histogram(self,df,column):
		
		df.hist(column=column)
		plt.show()

	def plot_heatmap(self,df):
		plt.figure()
		x=df.corr()
		mask = np.triu(np.ones_like(x, dtype=np.bool))
		sns.heatmap(x,annot=True,mask=mask,vmin=-1,vmax=1)
		plt.show()

	def scatter_plot(self,df,x,y,c,marker):
		plt.figure()
		plt.scatter(df[x],df[y],c=c,marker=marker)
		plt.xlabel(x)
		plt.ylabel(y)
		plt.title(y + " vs "+ x)
		plt.show()

	def line_plot(self,df,x,y,c,marker):
		plt.figure()
		df=df.sort_values(by=[x])
		plt.plot(df[x],df[y],c=c,marker=marker)
		plt.xlabel(x)
		plt.ylabel(y)
		plt.title(y + " vs "+ x)
		plt.show()