from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from skimage.color import rgb2lab, gray2rgb
import numpy as np
import tensorflow as tf
import glob
import cv2
import pandas as pd

#Välj epok och dropout
epochs              =   160

vggmodel = VGG16()
newmodel = Sequential() 
#num = 0
for i, layer in enumerate(vggmodel.layers):
    if i<19:          #Only up to 19th layer to include feature extraction only
        newmodel.add(layer)

for layer in newmodel.layers:
    layer.trainable=False   #We don't want to train these layers again, so False. 


#Decoder
model = Sequential()

model.add(Conv2D(256, (3,3), activation='relu', padding='same', input_shape=(7,7,512)))
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
#model.summary()


model.compile(optimizer=Adam(learning_rate=0.001), loss='mse' , metrics=['accuracy'])
 

def preprocess(img):
    lab =   rgb2lab(img)
    X   =   lab[:,:,0]
    Y   =   (lab[:,:,1:] / 128) #A and B values range from -127 to 128, 
    X   =   np.array(X)
    Y   =   np.array(Y)
    X   =   X.reshape(X.shape+(1,)) #dimensions to be the same for X and Y
    
    sample = gray2rgb(X)
    sample = sample.reshape((1,224,224,3))
    prediction = newmodel.predict(sample)
    prediction = prediction.reshape((7,7,512))

    return prediction, Y


def gen_train():
    batch_X     =   []
    batch_Y     =   []
    for epoch in range(epochs):
        for fil in glob.glob("training_data/*.jpg"):
            if len(batch_X)==20:
                batch_X     =   []
                batch_Y     =   []
            image = cv2.imread(fil)
            x, y = preprocess(image)
            batch_X.append(x)
            batch_Y.append(y)
            if len(batch_X)==20:
                X   =   np.array(batch_X)
                Y   =   np.array(batch_Y)
                yield (X, Y)
def gen_valid():
    batch_X     =   []
    batch_Y     =   []
    for epoch in range(epochs):    
        for fil in glob.glob("validation_data/*.jpg"):
            if len(batch_X)==20:
                batch_X     =   []
                batch_Y     =   []
            image = cv2.imread(fil)
            x, y = preprocess(image)
            batch_X.append(x)
            batch_Y.append(y)
            if len(batch_X)==20:
                X   =   np.array(batch_X)
                Y   =   np.array(batch_Y)
                yield (X, Y)

dataset_train =     tf.data.Dataset.from_generator(gen_train, (tf.float32, tf.float32), output_shapes=(tf.TensorShape((None, None, None, None)), tf.TensorShape((None, None, None, None))))
dataset_valid =     tf.data.Dataset.from_generator(gen_valid, (tf.float32, tf.float32), output_shapes=(tf.TensorShape((None, None, None, None)), tf.TensorShape((None, None, None, None))))

lr_reducer    =   ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=25, verbose=1, min_lr=0.5e-9)
checkpoint    =   ModelCheckpoint(filepath='saved_models/model_at_{epoch:03d}.h5', monitor='val_loss', verbose=1, save_best_only=True)

history_model =   model.fit(dataset_train, validation_data=dataset_valid, validation_steps=25, steps_per_epoch=430, epochs=epochs, verbose=1, callbacks = [lr_reducer, checkpoint])
model.save('saved_models/final_model.h5')


pd.DataFrame.from_dict(history_model.history).to_csv('history.csv',index=False)