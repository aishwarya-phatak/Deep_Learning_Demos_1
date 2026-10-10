import numpy as np
from keras.models import Sequential
from keras.layers import Dense

input_values_xor = np.array([[0,0],[0,1],[1,0],[1,1]],
                            dtype=int)
print("--------------")
print(input_values_xor)
print("-------Shape of input-------")
print(np.shape(input_values_xor))

output_values_xor = np.array([0,1,1,0],dtype=int)
print("-------------")
print(output_values_xor)
print("------Shape of output-------")
print(np.shape(output_values_xor))

model = Sequential()

model.add(Dense(4, input_dim=2, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(input_values_xor, output_values_xor, epochs=900)

print(model.predict(np.array([[0,0]],dtype=int)))
print(model.predict(np.array([[0,1]],dtype=int)))
print(model.predict(np.array([[1,0]],dtype=int)))
print(model.predict(np.array([[1,1]],dtype=int)))