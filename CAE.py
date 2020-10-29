# TensorFlow and skimage
from skimage.color import lab2rgb, rgb2lab
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, BatchNormalization, Dropout, ELU
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
#from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import Model

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

config      =   {
                    "strides"       :   2,
                    "kernel_size"   :   3,
                    "activation"    :   ELU(),
                    "padding"       :   'same',
                    "optimizer"     :   'adam',
                    "loss"          :   'mse',
                    "metrics"       :   ['accuracy'],
                    "batch_size"    :   32,
                    "epochs"        :   10,
                    "dropout"       :   0.5,
                    "show_summary"  :   False,
                    "filters"       :   [32, 64, 128, 256],
                    "batch_norm"    :   True
                }
def main():
    #Load and transform data
    (train_images, _), (test_images, _) = cifar10.load_data()
    train_images                        =   train_images / 255.0
    test_images                         =   test_images / 255.0
    l_train, ab_train                   =   rgb_to_lab(train_images)
    l_test, ab_test                     =   rgb_to_lab(test_images)
    _, rows, cols, channels             =   l_train.shape

    #Pick out validation data
    l_valid     =   l_train[45000:]
    ab_valid    =   ab_train[45000:]
    l_train     =   l_train[:45000]
    ab_train    =   ab_train[:45000]

    #Model Input
    model   =   [Input(shape=(rows, cols, 1))]
    #Model Encoder
    for filter in config['filters']: model.append(LayerDown(filter, config))
    #Model Decoder
    for filter in reversed(config['filters']): model.append(LayerUp(filter, config))
    #Model Output
    model.append(Conv2DTranspose(filters = 2, kernel_size=config["kernel_size"], strides=1, activation='sigmoid', padding=config["padding"]))
    #Save model
    model = Sequential(model)

    #Show the model in the terminal
    if config["show_summary"]   :   model.summary()

    #Train the model
    model.compile(optimizer=config['optimizer'], loss=config['loss'], metrics=config['metrics'])
    model.fit(l_train, ab_train, validation_data=(l_valid, ab_valid), batch_size = config["batch_size"], epochs=config["epochs"])

    #Get test accuracy
    test_loss, test_acc = model.evaluate(l_test,  ab_test, verbose=2)

    #Plot a sample from input-output
    guess_test      =   model.predict(l_test[0:10])
    images_guess    =   lab_to_rgb(l_test[0:10],guess_test)
    imgs = images_guess[:9]
    imgs = imgs.reshape((3, 3, rows, cols, 3))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    plt.figure()
    plt.axis('off')
    plt.title('Colorized test images (Predicted)')
    plt.imshow(imgs, interpolation='none')
    plt.show()

class LayerDown(Model):
    def __init__(self, filter_size, config):
        super(LayerDown, self).__init__()
        model   =   []
        model.append(Conv2D(filters = filter_size, kernel_size=config["kernel_size"], strides=config["strides"], padding=config["padding"] ))
        model.append(config["activation"])
        if config["batch_norm"]:    model.append(BatchNormalization())
        if config["dropout"] > 0:   model.append(Dropout(config["dropout"]))
        self.model  =   Sequential(model)
    def call(self, x):
        return self.model(x)

class LayerUp(Model):
    def __init__(self, filter_size, config):
        super(LayerUp, self).__init__()
        model   =   []
        model.append(Conv2DTranspose(filters = filter_size, kernel_size=config["kernel_size"], strides=config["strides"], padding=config["padding"] ))
        model.append(config["activation"])
        if config["batch_norm"]:    model.append(BatchNormalization())
        if config["dropout"] > 0:
            model.append(Dropout(config["dropout"]))
        self.model  =   Sequential(model)
    def call(self, x):
        return self.model(x)

def rgb_to_lab(imgs):
    #Takes from range [0,1] and replies with [0,1]
    lab     =   rgb2lab(imgs)
    l       =   lab[...,0].reshape(imgs.shape[0],imgs.shape[1],imgs.shape[2],1)
    ab      =   lab[...,1:]
    return l/100, (ab+128)/255

def lab_to_rgb(l, ab):
    #Takes from range [0,1] and replies with [0,1]
    lab     =   np.concatenate([l*100,(ab*255)-128], axis=3)
    imgs    =   np.zeros(lab.shape)
    for i in range(len(imgs)):
        imgs[i]     =   lab2rgb(lab[i])
    return imgs

if __name__=='__main__':
    main()