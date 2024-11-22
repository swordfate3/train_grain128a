import numpy as np
from pickle import dump
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import SeparableConv1D, Dense, Conv1D, Input, Reshape, Cropping1D, Concatenate,Permute, Add, Flatten, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop,SGD


def CConv1D(filters, kernel_size, strides=1, activation='linear', padding='valid',
            kernel_initializer='glorot_uniform', kernel_regularizer=None):

    def CConv1D_inner(x):
        in_width = int(x.get_shape()[1])

        if in_width % strides == 0:
            pad_along_width = max(kernel_size - strides, 0)
        else:
            pad_along_width = max(kernel_size - (in_width % strides), 0)

        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        # left and right side for padding
        pad_left = Cropping1D(cropping=(in_width-pad_left, 0))(x)
        pad_right = Cropping1D(cropping=(0, in_width-pad_right))(x)

        # add padding to incoming image
        conc = Concatenate(axis=1)([pad_left, x, pad_right])

        # perform the circular convolution
        cconv1d = Conv1D(filters=filters, kernel_size=kernel_size,
                         strides=strides, activation=activation,
                         padding='valid',
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer)(conc)

        # return circular convolution layer
        return cconv1d

    return CConv1D_inner

def make_resnet(
        num_blocks=32, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=8, ks=3, depth=5, dilation_rate=1, reg_param=0.0001,
        final_activation='sigmoid', cconv=False
):
    Conv = CConv1D if cconv else SeparableConv1D  # Check if we use circular convolutions
    # Input and preprocessing layers
    inp = Input(shape=(num_blocks * word_size,))
    rs = Reshape((num_blocks, word_size))(inp)
    perm = Permute((2, 1))(rs)
    # add a single residual layer that will expand the data to num_filters channels
    # this is a bit-sliced layer
    conv0 = Conv(num_filters, kernel_size=1, dilation_rate=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)
    # add residual blocks
    shortcut = conv0
    for i in range(depth):
        conv1 = Conv(num_filters, kernel_size=ks, padding='same', dilation_rate=2, kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv(num_filters, kernel_size=ks, padding='same', dilation_rate=1, kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
    # add prediction head
    flat1 = Flatten()(shortcut)
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    model.summary()
    return model

def cyclic_lr(num_epochs, high_lr, low_lr):
    return lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)
best_hyperparameters = {
    'n_filters': 4,
    'depth': 17,
    'kernel_size': 11,
    'n_neurons': 550,
    'batch_size': 100,
    'reg_param': 6.486746864806685e-05,
    'lr_high': 0.0030607453354945208,
    'lr_low': 4.150501701826967e-05
}

def mlp_random(classes, number_of_samples, activation, neurons, layers, learning_rate):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(number_of_samples,)))
    for l_i in range(layers):
        model.add(Dense(neurons, activation=activation, kernel_initializer='he_uniform', bias_initializer='zeros'))
    model.add(Dense(classes, activation='softmax'))
    # model.add(Dense(classes))
    model.summary()
    optimizer = RMSprop(learning_rate=learning_rate)#categorical_crossentropy
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def run_mlp(X_profiling, Y_profiling, X_validation, Y_validation,classes):
    mini_batch = 50 #random.randrange(500, 1000, 100)
    learning_rate = 0.000180006094109679
    activation = 'tanh'
    layers = 2
    neurons = 100

    model = mlp_random(classes, len(X_profiling[0]), activation, neurons, layers, learning_rate)
    es = EarlyStopping(monitor='val_accuracy',mode='max',patience=30,restore_best_weights=True)
    his = model.fit(
        x=X_profiling,
        y=Y_profiling,
        batch_size=mini_batch,
        verbose=2,
        epochs=200,
        shuffle=True,
        validation_data=(X_validation, Y_validation),)
        # callbacks=[es])

    # prediction = model.predict(X_validation)
    # prediction = prediction.reshape(-1)
    # Y_validation = Y_validation.reshape(-1)
    # corr = np.corrcoef(Y_validation,prediction)
    # K.clear_session()
    # return prediction



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

if __name__ == "__main__":
    n_epochs = 300
    # net = make_resnet(
    #     num_filters=best_hyperparameters['n_filters'],
    #     depth=best_hyperparameters['depth'],
    #     ks=best_hyperparameters['kernel_size'],
    #     d1=best_hyperparameters['n_neurons'],
    #     d2=best_hyperparameters['n_neurons'],
    #     reg_param=best_hyperparameters['reg_param'],
    # )
    # net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    # lr = LearningRateScheduler(cyclic_lr(n_epochs, best_hyperparameters['lr_high'], best_hyperparameters['lr_low']))

    X = np.loadtxt('data.txt',dtype=np.uint8)[:50000]
    print(X.shape)
    X_val = np.loadtxt('data.txt',dtype=np.uint8)[50000:60000]

    Y = np.loadtxt('label.txt',dtype=np.uint8)[:50000]
    Y = Y.reshape(-1,1)
    Y = to_categorical(Y,num_classes=2)
    Y_val = np.loadtxt('label.txt',dtype=np.uint8)[50000:60000]
    Y_val = Y_val.reshape(-1,1)
    Y_val = to_categorical(Y_val,num_classes=2)

    run_mlp(X,Y,X_val,Y_val,2)

    # h = net.fit(X, Y, epochs=n_epochs, batch_size=best_hyperparameters['batch_size'], validation_data=(X_val, Y_val), callbacks=[lr])