import numpy as np
import pandas as pd

#.csv 형식을 읽어들이겠다
data = pd.read_csv('gpascore.csv')
#print(data)

#NAN/빈값있는 행 지워줌
data = data.dropna()
#NAN/빈값을 100으로 채워줌
#data.fillna(100)

y데이터 = data['admit'].values
x데이터 = []

for i, rows in data.iterrows():
    x데이터.append([rows['gre'], rows['gpa'], rows['rank']])



import tensorflow as tf

#딥러닝 모델 만들기
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit( np.array(x데이터),  np.array(y데이터), epochs=10)

# 예측
model.predict( [],[] )