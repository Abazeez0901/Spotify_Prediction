import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


data = pd.read_csv(r"C:\Users\Dell\Downloads\spotify_tracks.csv")


data = data[['year', 'popularity', 'language']]


encoder = LabelEncoder()
data['language'] = encoder.fit_transform(data['language'])


x = data.drop(columns=['popularity'])
y = data['popularity']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)


pkl.dump(model, open("spotify_model.pkl", "wb"))
pkl.dump(encoder, open("encoder.pkl", "wb"))

print("✅ Model and Encoder saved successfully!")