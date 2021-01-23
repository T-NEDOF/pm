# for the convolutional network
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten,LSTM
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
import keras
from keras.layers.core import Activation

# from neural_network_model.config import config
from packages.neural_network_model.neural_network_model.config import config

import keras.backend as K


def r2_keras(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def lstm_clc_model(nb_features=25,nb_out=1,sequence_length=50):

    modelclassification = Sequential()

    modelclassification.add(LSTM(
             input_shape=(sequence_length, nb_features),
             units=100,
             return_sequences=True))
    modelclassification.add(Dropout(0.2))

    modelclassification.add(LSTM(
              units=50,
              return_sequences=False))
    modelclassification.add(Dropout(0.2))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

    modelclassification.add(Dense(units=nb_out, activation='sigmoid'))
    modelclassification.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return modelclassification

def lstm_rgs_model(nb_features=25,nb_out=1,sequence_length=50):
    modelregression = Sequential()
    modelregression.add(LSTM(
            input_shape=(sequence_length, nb_features),
            units=100,
            return_sequences=True))
    modelregression.add(Dropout(0.2))
    modelregression.add(LSTM(
            units=50,
            return_sequences=False))
    modelregression.add(Dropout(0.2))
    modelregression.add(Dense(units=nb_out))
    modelregression.add(Activation("linear"))
    modelregression.compile(loss='mean_squared_error', optimizer='rmsprop',metrics=['mae',r2_keras])
    return modelregression

checkpoint = ModelCheckpoint(config.MODEL_PATH,
                             monitor='acc',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

clc_callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(config.CLASSIFICATION_MODEL,monitor='val_loss', save_best_only=True, mode='min', verbose=0)]


rgs_callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(config.REGRESSION_MODEL,monitor='val_loss', save_best_only=True, mode='min', verbose=0)]

lstm_clc = KerasClassifier(build_fn=lstm_clc_model,
                          epochs=30,
                          batch_size=200,
                          validation_split=0.05,
                          verbose=2,  
                          callbacks=clc_callbacks_list                   
                          )

lstm_rgs = KerasRegressor(build_fn=lstm_rgs_model,
                          epochs=30,
                          batch_size=200,
                          validation_split=0.05,
                          verbose=2,  
                          callbacks=rgs_callbacks_list                   
                          )

if __name__ == '__main__':
    clc_model = lstm_clc_model()
    rgs_model = lstm_rgs_model()
    # model.summary()
