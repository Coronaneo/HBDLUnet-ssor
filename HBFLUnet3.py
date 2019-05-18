'''
HBFLUnet: 
         This net is devised to find a special LU factorization of A to minimize norm(L(U(b))-A*b)/norm(A*b), given a kernel A. 

Copyright (c) 2019 Pang Qiyuan
'''

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Multiply, Input, Flatten, Dense, Lambda, Reshape, Concatenate, Add, Subtract
import keras
import json
import scipy.io as sio
from keras import backend as K
import h5py
from keras.utils import plot_model
import mat4py
import numpy.matlib
from HBFLU_auxiliaryfunc import *

## ========================================================== set parameters ==================================================================== ##
BATCH_SIZE = 200
EPOCHS = 10
lr = 0.01
trainable = 0

## ========================================================== rearrange data ==================================================================== ##

filename1 = 'data.mat'
filename2 = 'LUold.mat'


data = mat4py.loadmat(filename1)
data = data['data']
train_data_real = data['train_data_real']
train_data_imag = data['train_data_imag']
target_data_real = data['target_data_real']
target_data_imag = data['target_data_imag']
LUold = mat4py.loadmat(filename2)
LUold = LUold['LUold']
Url = LUold['Url']
Uim = LUold['Uim']
Lrl = LUold['Lrl']
Lim = LUold['Lim']
Drl = LUold['Drl']
Dim = LUold['Dim']


Lrl = L_rearrange(Lrl)
Lim = L_rearrange(Lim)
Url = U_rearrange(Url)
Uim = U_rearrange(Uim)
Drl = np.diag(np.array(Drl))
Dim = np.diag(np.array(Dim))

M = len(train_data_real)
N = len(train_data_real[0])

InputArray_real = np.array(train_data_real).reshape(M,N)
InputArray_imag = np.array(train_data_imag).reshape(M,N)
OutputArray_real = np.array(target_data_real).reshape(M,N)
OutputArray_imag = np.array(target_data_imag).reshape(M,N)

n_train = int(M * 0.5)
n_test  = M - n_train
X_trainr = InputArray_real[0:n_train, :]
X_traini = InputArray_imag[0:n_train, :]
Y_trainr = OutputArray_real[0:n_train, :]
Y_traini = OutputArray_imag[0:n_train, :]
X_testr  = InputArray_real[(M-n_test):M, :]
X_testi  = InputArray_imag[(M-n_test):M, :]
Y_testr  = OutputArray_real[(M-n_test):M, :]
Y_testi  = OutputArray_imag[(M-n_test):M, :]
'''
Drl = np.array(Drl).reshape((1,N))
Dim = np.array(Dim).reshape((1,N))
Drl = np.tile(Drl,(BATCH_SIZE,1))
Dim = np.tile(Dim,(BATCH_SIZE,1))

## ========================================================= model : HBFLUnet ===================================================================== ##
D_rl = tf.constant(Drl)
D_im = tf.constant(Dim)
'''
K.set_floatx('float64')
inputr = Input(batch_shape = (BATCH_SIZE,N), name = 'inputr',dtype = 'float64')
inputi = Input(batch_shape = (BATCH_SIZE,N), name = 'inputi',dtype = 'float64')

(Linv_rl,Linv_im,xrl,xim) = Linvnet(Lrl,Lim,inputr,inputi,BATCH_SIZE,N)

D_rl = Dense(N,use_bias = False, kernel_initializer = keras.initializers.Constant(Drl), trainable = False)
D_im = Dense(N,use_bias = False, kernel_initializer = keras.initializers.Constant(Dim), trainable = False)
xrl1 = D_rl(xrl)
xrl2 = D_im(xim)
xim1 = D_rl(xim)
xim2 = D_im(xrl)
xrl = Subtract()([xrl1,xrl2])
xim = Add()([xim1,xim2])

'''
xrl1 = Multiply()([D_rl,xrl])
xrl2 = Multiply()([D_im,xim])
xim1 = Multiply()([D_rl,xim])
xim2 = Multiply()([D_im,xrl])
xrl = Subtract()([xrl1,xrl2])
xim = Add()([xim1,xim2])
'''
(Uinv_rl,Uinv_im,xrl,xim) = Uinvnet(Url,Uim,xrl,xim,BATCH_SIZE,N)

outputr = xrl
outputi = xim
HBFLUnet = Model(inputs = [inputr,inputi], outputs = [outputr,outputi])

HBFLUnet.summary()
try:
    plot_model(HBFLUnet, to_file='HBFLUnet.png', show_shapes=True)
except ImportError:
    print("plot_model is not supported on this device")

HBFLUnet.compile(optimizer=keras.optimizers.Adam(lr=lr),loss = relative_err)

HBFLUnet.fit([X_trainr,X_traini], [Y_trainr,Y_traini], batch_size = BATCH_SIZE, epochs = EPOCHS, validation_data = ([X_testr,X_testi],[Y_testr,Y_testi]))

net_para = {'BATCH_SIZE':BATCH_SIZE,'EPOCHS':EPOCHS,'optimizer':'Adam','lr':lr,'loss':'relative_err','n_train':n_train}
sio.savemat('net_para.mat',net_para)

Dr = D_rl.get_weights()
Di = D_im.get_weights()

LU = {'Lrnew':0,'Linew':0,'Urnew':0,'Uinew':0,'Drnew':0,'Dinew':0}

LU['Lrnew'] = extract_L(Linv_rl)
print('Lrnew constructed!')
LU['Linew'] = extract_L(Linv_im)
print('Linew constructed!')
LU['Urnew'] = extract_U(Uinv_rl)
print('Urnew constructed!')
LU['Uinew'] = extract_U(Uinv_im)
print('Uinew constructed!')
LU['Drnew'] = Drl
LU['Dinew'] = Dim
print('Dnew constructed!')

#print(LU['Lrnew']['A21']['U'][0][0])

sio.savemat('LUnew.mat',LU)
print('A new LU has been cinstructed!')


