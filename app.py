import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

model = joblib.load("model.joblib")
df = pd.read_csv('news.csv')
vectorizer = joblib.load("vec.joblib")


cluster_0 = []
cluster_1= []
cluster_2=[]
cluster_3=[]
cluster_4=[]

for i in df.index:
  content = df["Content"][i]
  y = vectorizer.transform([content])
  prediction = model.predict(y)
  if prediction[0] == 0:
    cluster_0.append([df["Title"][i], df["Url"][i]])
  elif prediction[0] == 1:
    cluster_1.append([df["Title"][i], df["Url"][i]])
  elif prediction[0] == 2:
    cluster_2.append([df["Title"][i], df["Url"][i]])
  elif prediction[0] == 3:
    cluster_3.append([df["Title"][i], df["Url"][i]])
  else:
    cluster_4.append([df["Title"][i], df["Url"][i]])


st.title('News Article Clustering')

st.write('A collection of news articles scraped from New Times Rwanda(https://www.newtimes.co.rw/rwanda) and clustered using MiniBatchKMeans Clustering into 5 different clusters according to the similarity of their content')

st.header('Cluster 0')

for i in cluster_0:
  st.write(':blue[[',i[0],'](',i[1],')]')

st.header('Cluster 1')

for i in cluster_1:
  st.write(':blue[[',i[0],'](',i[1],')]')

st.header('Cluster 2')

for i in cluster_2:
  st.write(':blue[[',i[0],'](',i[1],')]')

st.header('Cluster 3')

for i in cluster_3:
  st.write(':blue[[',i[0],'](',i[1],')]')

st.header('Cluster 4')

for i in cluster_4:
  st.write(':blue[[',i[0],'](',i[1],')]')