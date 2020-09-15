# Import necessary packages
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras import backend as K
from numpy import mean
import scipy.io
import sklearn
from sklearn.model_selection import train_test_split

# Define training parameters
EPOCHS = 5
BS = 10
LATENT = 4
(length, width, depth) = (128, 20, 1)   # Dimensions of the DDMs

# Import data from .mat files
test_data = scipy.io.loadmat('DDMtest.mat')
train_data = scipy.io.loadmat('DDMtrain.mat')

# Extract data
x_train = train_data["DDM"]
x_test = test_data["DDM"]
input_img = Input(shape = (length, width, depth))

# Preprocess data
x_train = x_train.astype('float32') / 65535
x_test = x_test.astype('float32') / 65535
x_train = x_train.reshape(-1, length, width, depth)
x_test = x_test.reshape(-1, length, width, depth)

"""#Autoencoder Model"""

# Define Convolutional Autoencoder Model
def autoencoder(x):
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    units = x.shape[1]
    x = Dense(LATENT, name="latent")(x)
    x = Dense(units)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((32, 5, 8))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(1, (3, 3), activation='tanh', padding='same')(x)
    return x

# Define model training hyperparameters
autoencoder_train = Model(input_img, autoencoder(input_img))
autoencoder_train.compile(loss='mean_squared_error', optimizer = Adam())
autoencoder_train.summary()

# Create a file to save parameter weights
weightname = 'model_weights.h5'
model_saver = tf.keras.callbacks.ModelCheckpoint(weightname ,monitor='val_loss', verbose=1,save_best_only=True, save_weights_only=False, mode='auto')

# Train the Autoencoder
autoencoder_train.load_weights(weightname)
H = autoencoder_train.fit(x_train, x_train, validation_data=(x_test, x_test), epochs=EPOCHS, batch_size=BS, callbacks = [model_saver])

"""#Plotting original and extracted images"""

# Make a prediction using the validation set
extract = autoencoder_train.predict(x_test)
loss = H.history['loss']
print(H.history.keys())

# summarize history for training losses
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
scipy.io.savemat('History.mat', H.history)

# Plot a side-by-side comparison of the training model outcome
w=10
h=10
fig=plt.figure(figsize=(10, 20))
columns = 2
rows = 5
j = 0
k = 0
for i in range(1, columns*rows +1):
    if i%2 == 0:
        img = extract[j,...,0]
        j+=1
    else:
        img = x_test[k,...,0]
        k+=1
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, interpolation='nearest', aspect='auto')
plt.show()
fig.savefig('output.png')

# Save extracted data model
mdic = {"extract": extract}
scipy.io.savemat('DDMextract.mat', mdic)