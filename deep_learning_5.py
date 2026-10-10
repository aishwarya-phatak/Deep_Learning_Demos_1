import numpy as np
from keras.models import Sequential
from keras.layers import Dense

input_values_not = np.array([0,1],dtype=int)
print(input_values_not)
print(input_values_not.shape)

output_values_not = np.array([1,0],dtype=int)
print(output_values_not)
print(output_values_not.shape)

model = Sequential()

model.add(Dense(4,input_shape=(1,),activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(input_values_not,output_values_not,epochs=900)

print(model.predict(np.array([0],dtype=int)))
print(model.predict(np.array([1],dtype=int)))