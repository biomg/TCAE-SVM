import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Multiply, Conv1DTranspose
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D,Flatten,add,ReLU
import matplotlib

matplotlib.use('TkAgg')
import warnings

warnings.filterwarnings('ignore')

Train = np.load('Generic_dataset.npy')

def fit_model_seq(model,model_file, x_train, y_train, frac_val, epochs, batch_size, verbose):
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=5)
    mc = ModelCheckpoint(model_file, monitor='val_loss', mode='min', verbose=verbose, save_best_only=True)
    history = model.fit(x_train, y_train, validation_split=frac_val, epochs=epochs, batch_size=batch_size,
                        verbose=verbose, callbacks=[es,mc])
    return history
model_file = 'TCAE.h5'

if __name__ == '__main__':
    inputt = Input(shape=(Train.shape[1], Train.shape[2]))
    conv1 = Conv1D(filters=32, kernel_size=32, activation='relu', padding='same')(inputt)
    pool = MaxPooling1D(77)(conv1)
    pool1 = MaxPooling1D(7)(conv1)
    conv2 = Conv1D(filters=32, kernel_size=32, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling1D(11)(conv2)
    pool3 = add([pool,pool2])
    pool3 = ReLU()(pool3)
    attention_probs = Dense(32, activation='softmax', name='attention_probs')(pool3)
    attention_mul = Multiply(name='attention_mul')([pool3, attention_probs])
    encoded = attention_mul
    Upconv1 = Conv1DTranspose(32, 32, activation='relu', padding='same')(encoded)
    Upool = UpSampling1D(77)(Upconv1)
    Uppool1 = UpSampling1D(11)(Upconv1)
    Upconv2 = Conv1DTranspose(32, 32, activation='relu', padding='same')(Uppool1)
    Uppool2 = UpSampling1D(7)(Upconv2)
    UP = add([Upool,Uppool2])
    UP = ReLU()(UP)
    decoded = Conv1DTranspose(4, 3, activation='relu', padding='same')(UP)
    model = Model(inputs=inputt, outputs=decoded)
    MPRA_data = np.load('MPRA_dataset.npy')

    history = fit_model_seq(model,model_file,Train, Train, frac_val=0.2,epochs=30, batch_size=128,verbose=1)
    encoder = Model(inputs=inputt, outputs=encoded)
    Encoded_MPRA1 = encoder.predict(MPRA_data)
    flatten = Flatten()
    Encoded_MPRA = flatten(Encoded_MPRA1)
    np.save('MPRA_feature',Encoded_MPRA)
