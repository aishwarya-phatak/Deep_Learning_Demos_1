#deep learning -- 1
import numpy as np
from tensorflow.keras.models import Sequential
from keras.layers import Dense

#input
hours_of_study = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])
print(np.shape(hours_of_study))

result_obtained = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1])
print(result_obtained)
print(np.shape(result_obtained))

#define model
model = Sequential()

#create layers
model.add(Dense(4,input_dim=1,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

#compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#model -- fit
model.fit(hours_of_study,
          result_obtained,
          epochs=850)

#model -- prediction
print(model.predict(np.array([13.33])))
print(model.predict(np.array([22])))
print(model.predict(np.array([8.5])))
print(model.predict(np.array([4.432])))
print(model.predict(np.array([11.9])))