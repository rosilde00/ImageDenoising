from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.models import Model
import pandas as pd
import numpy as np
from keras.optimizers import Adam

#Creazione del training set e normalizzazione delle immagini
train = pd.read_csv("./fashion-mnist_train.csv")
train = train.drop('label', axis=1)
clean_train = train[list(train.columns)].values
clean_train = clean_train / 255
clean_train = clean_train.reshape(-1, 784)

#Applicazione del noise alle immagini per ottenere le immagini del train
noisy_train= []
for i in range(0, len(clean_train)):
    x = np.random.rand(784)
    noise = np.random.normal(x, 0.2*x)
    noisy_img = clean_train[i] + noise
    noisy_train.append(noisy_img)
noisy_train = np.asarray(noisy_train)

#Creazione del test set e normalizzazione delle immagini
test = pd.read_csv("./fashion-mnist_test.csv")
test = test.drop('label', axis=1)
clean_test = test[list(test.columns)].values
clean_test = clean_test / 255
clean_test = clean_test.reshape(-1, 784)

#Applicazione del noise alle immagini per ottenere le immagini del test
noisy_test = []
for i in range(0, len(clean_test)):
    x = np.random.rand(784)
    noise = np.random.normal(x, 0.2*x)
    noisy_img = clean_test[i] + noise
    noisy_test.append(noisy_img)
noisy_test = np.asarray(noisy_test)

#Livello di input
input_layer = Input(shape=(784,))

#Livelli di codifica interni
encode_layer1 = Dense(600, activation='relu')(input_layer)
encode_layer2 = Dense(100, activation='relu')(encode_layer1)

#Livello latente
latent_view   = Dense(20, activation='relu')(encode_layer2)

#Livelli di decodifica interni
decode_layer1 = Dense(100, activation='relu')(latent_view)
decode_layer2 = Dense(600, activation='relu')(decode_layer1)

#Livello di output finale
output_layer  = Dense(784)(decode_layer2)

model = Model(input_layer, output_layer)
model.summary()

#Train del modello con early stopping
adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss='mse', metrics=['mean_absolute_error'])
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=3, verbose=1, mode='auto')
model.fit(noisy_train, clean_train, epochs=30, batch_size=2048, validation_data=(noisy_test, clean_test), callbacks=[early_stopping])
scores = model.evaluate(noisy_test, clean_test, verbose=1)
print('MSE e MAE sul validation set')
print(f'MSE: {scores[0]}')
print(f'MAE: {scores[1]}')

#Salvataggio del modello
model.save_weights('./results')
