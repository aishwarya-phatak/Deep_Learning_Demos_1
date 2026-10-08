#deep learning - 2

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#OR operation
input_values = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=int)
print(np.shape(input_values))
output_values_and = np.array([0,1,1,1],dtype=int)

model = Sequential()

model.add(Dense(4,input_dim=2,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(input_values,output_values_and,epochs=950)

#checking on input shapes
arr_1 = np.array([0,0])
print("Shape of arr_1 is  : ",np.shape(arr_1))

arr_2 = np.array([[0,0]])
print("Shape of arr_2 is : ",np.shape(arr_2))

print(model.predict(np.array([[0,0]],dtype=int)))