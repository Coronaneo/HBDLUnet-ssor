import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import Conv1D, Input, Flatten, Dense, Lambda, Reshape, Concatenate, Add, Subtract
import numpy as np

def  slices(x, start, end):
     return x[:,start:end]

def  init_dict(num):
     dict0 = {0:0}
     for i in range(num):
         dict0.update({i:0})
     return dict0

def  init_list(num):
     list0 = [0]
     for i in range(1,num):
         list0.append(i)
     return list0

def relative_err(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred-y_true))/tf.reduce_mean(tf.square(y_true)))


def L_rearrange(L):
    if  L['isLeaf'] == 1:
        Lnew = {'isLeaf':1,'sz':L['sz'],'szsub':L['szsub'],'A11':0,'A12':0,'A21':0,'A22':0}
        Lnew['A11'] = np.array(L['A11']).reshape((L['szsub'],L['szsub']))
        Lnew['A21'] = np.array(L['A21']).reshape((L['szsub'],L['szsub']))
        Lnew['A22'] = np.array(L['A22']).reshape((L['szsub'],L['szsub']))
    else:
        Lnew = {'isLeaf':0,'sz':L['sz'],'szsub':L['szsub'],'A11':0,'A12':0,'A21':0,'A22':0}
        Lnew['A21'] = {'data':L['A21']['data'],'U':0,'V':0,'S':0}
        Lnew['A11'] = L_rearrange(L['A11'])
        Lnew['A22'] = L_rearrange(L['A22'])
        data = L['A21']['data']
        Lnew['A21']['U'] = init_dict(data[1])
        Lnew['A21']['V'] = init_dict(data[1])
        Lnew['A21']['S'] = init_dict(4**data[1])
        if  data[1] > 1:
            for i in range(data[1]):
                Lnew['A21']['U'][i] = init_dict(data[4])
                Lnew['A21']['V'][i] = init_dict(data[4])
                if  i == 0:
                    r = data[2]
                else:
                    r = 2*data[3]
                for j in range(data[4]):
                    Lnew['A21']['V'][i][j] = np.array(L['A21']['V'][i][0][j]).reshape((data[3],r))
                    Lnew['A21']['U'][i][j] = np.array(L['A21']['U'][i][0][j]).reshape((r,data[3]))
        else:
            Lnew['A21']['U'][0] = init_dict(data[4])
            Lnew['A21']['V'][0] = init_dict(data[4])
            for j in range(data[4]):
                Lnew['A21']['V'][0][j] = np.array(L['A21']['V'][j]).reshape((data[3],data[2]))
                Lnew['A21']['U'][0][j] = np.array(L['A21']['U'][j]).reshape((data[2],data[3]))            

        for i in range(4**(data[1])):
            Lnew['A21']['S'][i] = np.array(L['A21']['S'][i]).reshape((data[5],data[5]))
    return Lnew

def U_rearrange(U):
    if  U['isLeaf'] == 1:
        Unew = {'isLeaf':1,'sz':U['sz'],'szsub':U['szsub'],'A11':0,'A12':0,'A21':0,'A22':0}
        Unew['A11'] = np.array(U['A11']).reshape((U['szsub'],U['szsub']))
        Unew['A12'] = np.array(U['A12']).reshape((U['szsub'],U['szsub']))
        Unew['A22'] = np.array(U['A22']).reshape((U['szsub'],U['szsub']))
    else:
        Unew = {'isLeaf':0,'sz':U['sz'],'szsub':U['szsub'],'A11':0,'A12':0,'A21':0,'A22':0}
        Unew['A12'] = {'data':U['A12']['data'],'U':0,'V':0,'S':0}
        Unew['A11'] = U_rearrange(U['A11'])
        Unew['A22'] = U_rearrange(U['A22'])
        data = U['A12']['data']
        Unew['A12']['U'] = init_dict(data[1])
        Unew['A12']['V'] = init_dict(data[1])
        Unew['A12']['S'] = init_dict(4**data[1])
        if  data[1] > 1:
            for i in range(data[1]):
                Unew['A12']['U'][i] = init_dict(data[4])
                Unew['A12']['V'][i] = init_dict(data[4])
                if  i == 0:
                    r = data[2]
                else:
                    r = 2*data[3]
                for j in range(data[4]):
                    Unew['A12']['V'][i][j] = np.array(U['A12']['V'][i][0][j]).reshape((data[3],r))
                    Unew['A12']['U'][i][j] = np.array(U['A12']['U'][i][0][j]).reshape((r,data[3]))
        else:
            Unew['A12']['U'][0] = init_dict(data[4])
            Unew['A12']['V'][0] = init_dict(data[4])
            for j in range(data[4]):
                Unew['A12']['V'][0][j] = np.array(U['A12']['V'][j]).reshape((data[3],data[2]))
                Unew['A12']['U'][0][j] = np.array(U['A12']['U'][j]).reshape((data[2],data[3]))            

        for i in range(4**(data[1])):
            Unew['A12']['S'][i] = np.array(U['A12']['S'][i]).reshape((data[5],data[5]))
    return Unew

def rearrange(start,end):
    length = int(end - start)
    y = np.zeros(length, dtype = int)
    if  np.mod(length, 4) != 0:
        print('ERROR: end - start should be a multiple of 4')
        
    if  length > 4:
        int1 = int(length/4)
        y[0:int1] = rearrange(start,start + int1)
        y[int1:int1*2] = rearrange(start + int1*2, start + 3*int1)
        y[int1*2:3*int1] = rearrange(start + int1, start + int1*2)
        y[3*int1:length] = rearrange(start + 3*int1, end)
    else:
        y[0] = start
        y[1] = start + 2
        y[2] = start + 1
        y[3] = end - 1
    return y

def Linvnet(Lrl,Lim,brl,bim,M,N):
    #Linv_rl = Lrl
    #Linv_im = Lim
    if  Lrl['isLeaf'] == 1:
        Linv_rl = {'isLeaf':1,'sz':Lrl['sz'],'szsub':Lrl['szsub'],'A11':0,'A12':0,'A21':0,'A22':0}
        Linv_im = {'isLeaf':1,'sz':Lim['sz'],'szsub':Lim['szsub'],'A11':0,'A12':0,'A21':0,'A22':0}
        Linv_rl['A11'] = Dense(int(N/2),use_bias = False, kernel_initializer = keras.initializers.Constant(Lrl['A11']), trainable = True)
        #brl1 = K.slice(brl,[0,0],[M,int(N/2)])
        brl1 = Lambda(slices, output_shape = (int(N/2),), arguments = {'start':0, 'end':int(N/2)})(brl)
        xrl11 = Linv_rl['A11'](brl1)
        Linv_im['A11'] = Dense(int(N/2),use_bias = False, kernel_initializer = keras.initializers.Constant(Lim['A11']), trainable = True)
        #bim1 = K.slice(bim,[0,0],[M,int(N/2)])
        bim1 = Lambda(slices, output_shape = (int(N/2),), arguments = {'start':0, 'end':int(N/2)})(bim)
        xrl12 = Linv_im['A11'](bim1)
        xrl1 = Subtract()([xrl11,xrl12])
        xim11 = Linv_rl['A11'](bim1)
        xim12 = Linv_im['A11'](brl1)
        xim1 = Add()([xim11,xim12])

        Linv_rl['A21'] = Dense(int(N/2),use_bias = False, kernel_initializer = keras.initializers.Constant(Lrl['A21']), trainable = True)
        Linv_im['A21'] = Dense(int(N/2),use_bias = False, kernel_initializer = keras.initializers.Constant(Lim['A21']), trainable = True)
        b2rl1 = Linv_rl['A21'](xrl1)
        b2rl2 = Linv_im['A21'](xim1)
        b2rl = Subtract()([b2rl1,b2rl2])
        b2im1 = Linv_rl['A21'](xim1)
        b2im2 = Linv_im['A21'](xrl1)
        b2im = Add()([b2im1,b2im2])
        #brl2 = K.slice(brl,[0,int(N/2)],[M,N])
        #bim2 = K.slice(bim,[0,int(N/2)],[M,N])
        brl2 = Lambda(slices, output_shape = (int(N/2),), arguments = {'start':int(N/2), 'end':N})(brl)
        bim2 = Lambda(slices, output_shape = (int(N/2),), arguments = {'start':int(N/2), 'end':N})(bim)
        brl2 = Subtract()([brl2,b2rl])
        bim2 = Subtract()([bim2,b2im])

        Linv_rl['A22'] = Dense(int(N/2),use_bias = False, kernel_initializer = keras.initializers.Constant(Lrl['A22']), trainable = True)
        xrl21 = Linv_rl['A22'](brl2)
        Linv_im['A22'] = Dense(int(N/2),use_bias = False, kernel_initializer = keras.initializers.Constant(Lim['A22']), trainable = True)
        xrl22 = Linv_im['A22'](bim2)
        xrl2 = Subtract()([xrl21,xrl22])
        xim21 = Linv_rl['A22'](bim2)
        xim22 = Linv_im['A22'](brl2)
        xim2 = Add()([xim21,xim22])

        xrl = Concatenate(axis = 1)([xrl1,xrl2])
        xim = Concatenate(axis = 1)([xim1,xim2])
    else:
        Linv_rl = {'isLeaf':0,'sz':Lrl['sz'],'szsub':Lrl['szsub'],'A11':0,'A12':0,'A21':0,'A22':0}
        Linv_rl['A21'] = {'data':Lrl['A21']['data'],'U':Lrl['A21']['U'],'V':Lrl['A21']['V'],'S':Lrl['A21']['S']}
        Linv_im = {'isLeaf':0,'sz':Lim['sz'],'szsub':Lim['szsub'],'A11':0,'A12':0,'A21':0,'A22':0}
        Linv_im['A21'] = {'data':Lim['A21']['data'],'U':Lim['A21']['U'],'V':Lim['A21']['V'],'S':Lim['A21']['S']}

        #brl1 = K.slice(brl,[0,0],[M,int(N/2)])
        #bim1 = K.slice(bim,[0,0],[M,int(N/2)])
        brl1 = Lambda(slices, output_shape = (int(N/2),), arguments = {'start':0, 'end':int(N/2)})(brl)
        bim1 = Lambda(slices, output_shape = (int(N/2),), arguments = {'start':0, 'end':int(N/2)})(bim)
        (Linv_rl['A11'],Linv_im['A11'],xrl1,xim1) = Linvnet(Lrl['A11'],Lim['A11'],brl1,bim1,M,int(N/2))


        data = Lrl['A21']['data']
        dict0r = init_dict(data[4])
        dict0i = init_dict(data[4])
        for i in range(data[4]):
            #dict0r[i] = K.slice(xrl1,[0,i*data[2]/2],[M,(i+1)*data[2]/2]) 
            #dict0i[i] = K.slice(xim1,[0,i*data[2]/2],[M,(i+1)*data[2]/2]) 
            dict0r[i] = Lambda(slices, output_shape = (int(data[2]/2),), arguments = {'start':int(i*data[2]/2), 'end':int((i+1)*data[2]/2)})(xrl1)
            dict0i[i] = Lambda(slices, output_shape = (int(data[2]/2),), arguments = {'start':int(i*data[2]/2), 'end':int((i+1)*data[2]/2)})(xim1)
        for i in range(data[1]):
            dict1r = init_dict(data[4])
            dict1i = init_dict(data[4])
            for j in range(4**(i)):
                for k in range(2):
                    for n in range(int(data[4]/4**(i)/2)):
                        xr = Concatenate(axis = 1)([dict0r[2*int(j*data[4]/4**(i)/2+n)],dict0r[2*int(j*data[4]/4**(i)/2+n)+1]])
                        xi = Concatenate(axis = 1)([dict0i[2*int(j*data[4]/4**(i)/2+n)],dict0i[2*int(j*data[4]/4**(i)/2+n)+1]])
                        Linv_rl['A21']['V'][i][int((j+k/2)*data[4]/4**(i)+n)] = Dense(data[3],use_bias = False, kernel_initializer = keras.initializers.Constant(Lrl['A21']['V'][i][int((j+k/2)*data[4]/4**(i)+n)]), trainable = True)
                        Linv_im['A21']['V'][i][int((j+k/2)*data[4]/4**(i)+n)] = Dense(data[3],use_bias = False, kernel_initializer = keras.initializers.Constant(Lim['A21']['V'][i][int((j+k/2)*data[4]/4**(i)+n)]), trainable = True)
                        dr1 = Linv_rl['A21']['V'][i][int((j+k/2)*data[4]/4**(i)+n)](xr)
                        dr2 = Linv_im['A21']['V'][i][int((j+k/2)*data[4]/4**(i)+n)](xi)
                        di1 = Linv_rl['A21']['V'][i][int((j+k/2)*data[4]/4**(i)+n)](xi)
                        di2 = Linv_im['A21']['V'][i][int((j+k/2)*data[4]/4**(i)+n)](xr)
                        dict1r[int((j+k/2)*data[4]/4**(i)+n)] = Subtract()([dr1,dr2])
                        dict1i[int((j+k/2)*data[4]/4**(i)+n)] = Add()([di1,di2])
            dict0r = dict1r
            dict0i = dict1i
            del dict1r
            del dict1i
        
        y = rearrange(0, 4**data[1])
        flag = int(data[5]/data[3])
        dict1r = init_dict(4**data[1])
        dict1i = init_dict(4**data[1])
        for i in range(4**data[1]):
            if  flag == 2:
                xr = Concatenate(axis = 1)([dict0r[2*y[i]],dict0r[2*y[i]+1]])
                xi = Concatenate(axis = 1)([dict0i[2*y[i]],dict0i[2*y[i]+1]])
            else:
                xr = dict0r[y[i]]
                xi = dict0i[y[i]]
            Linv_rl['A21']['S'][i] = Dense(data[5],use_bias = False, kernel_initializer = keras.initializers.Constant(Lrl['A21']['S'][i]), trainable = True)
            Linv_im['A21']['S'][i] = Dense(data[5],use_bias = False, kernel_initializer = keras.initializers.Constant(Lim['A21']['S'][i]), trainable = True)
            dr1 = Linv_rl['A21']['S'][i](xr)
            dr2 = Linv_im['A21']['S'][i](xi)
            dict1r[i] = Subtract()([dr1,dr2])
            di1 = Linv_rl['A21']['S'][i](xi)
            di2 = Linv_im['A21']['S'][i](xr)
            dict1i[i] = Add()([di1,di2])
        if  flag == 2:
            dict0r = init_dict(data[4])
            dict0i = init_dict(data[4])
            for i in range(4**data[1]):
                #dict0r[2*i] = K.slice(dict1r[i],[0,0],[M,data[3]])
                #dict0i[2*i] = K.slice(dict1i[i],[0,0],[M,data[3]])
                #dict0r[2*i+1] = K.slice(dict1r[i],[0,data[3]],[M,data[5]])
                #dict0i[2*i+1] = K.slice(dict1i[i],[0,data[3]],[M,data[5]])
                dict0r[2*i] = Lambda(slices, output_shape = (data[3],), arguments = {'start':0, 'end':data[3]})(dict1r[i])
                dict0i[2*i] = Lambda(slices, output_shape = (data[3],), arguments = {'start':0, 'end':data[3]})(dict1i[i])
                dict0r[2*i+1] = Lambda(slices, output_shape = (data[3],), arguments = {'start':data[3], 'end':data[5]})(dict1r[i])
                dict0i[2*i+1] = Lambda(slices, output_shape = (data[3],), arguments = {'start':data[3], 'end':data[5]})(dict1i[i])
        else:
            dict0r = dict1r
            dict0i = dict1i
        del dict1r
        del dict1i

        for i in range(data[1]-1,-1,-1):
            if  i == 0:
                r = data[2]
                dict1r = init_dict(int(data[4]/2))
                dict1i = init_dict(int(data[4]/2))
            else:
                r = 2*data[3]
                dict1r = init_dict(int(data[4]))
                dict1i = init_dict(int(data[4]))
            for j in range(4**(i)):
                for k in range(int(data[4]/4**(i)/2)):
                    Linv_rl['A21']['U'][i][int(j*data[4]/4**(i)+k)] = Dense(r,use_bias = False, kernel_initializer = keras.initializers.Constant(Lrl['A21']['U'][i][int(j*data[4]/4**(i)+k)]), trainable = True)
                    Linv_im['A21']['U'][i][int(j*data[4]/4**(i)+k)] = Dense(r,use_bias = False, kernel_initializer = keras.initializers.Constant(Lim['A21']['U'][i][int(j*data[4]/4**(i)+k)]), trainable = True)
                    dr1 = Linv_rl['A21']['U'][i][int(j*data[4]/4**(i)+k)](dict0r[int(j*data[4]/4**(i)+k)])
                    dr2 = Linv_im['A21']['U'][i][int(j*data[4]/4**(i)+k)](dict0i[int(j*data[4]/4**(i)+k)])
                    dr = Subtract()([dr1,dr2])
                    di1 = Linv_rl['A21']['U'][i][int(j*data[4]/4**(i)+k)](dict0i[int(j*data[4]/4**(i)+k)])
                    di2 = Linv_im['A21']['U'][i][int(j*data[4]/4**(i)+k)](dict0r[int(j*data[4]/4**(i)+k)])
                    di = Add()([di1,di2])
                    Linv_rl['A21']['U'][i][int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)] = Dense(r,use_bias = False, kernel_initializer = keras.initializers.Constant(Lrl['A21']['U'][i][int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)]), trainable = True)
                    Linv_im['A21']['U'][i][int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)] = Dense(r,use_bias = False, kernel_initializer = keras.initializers.Constant(Lim['A21']['U'][i][int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)]), trainable = True)
                    dr1 = Linv_rl['A21']['U'][i][int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)](dict0r[int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)])
                    dr2 = Linv_im['A21']['U'][i][int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)](dict0i[int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)])
                    ddr = Subtract()([dr1,dr2])
                    di1 = Linv_rl['A21']['U'][i][int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)](dict0i[int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)])
                    di2 = Linv_im['A21']['U'][i][int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)](dict0r[int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)])
                    ddi = Add()([di1,di2])

                    #dict1r[int(j*data[4]/4**(i)/2+k)] = Add()([dr,ddr])
                    #dict1i[int(j*data[4]/4**(i)/2+k)] = Add()([di,ddi])
                    
                    if  i > 0:
                        dr = Add()([dr,ddr])
                        di = Add()([di,ddi])
                        dict1r[2*int(j*data[4]/4**(i)/2+k)] = Lambda(slices, output_shape = (data[3],), arguments = {'start':0, 'end':int(r/2)})(dr)
                        dict1r[2*int(j*data[4]/4**(i)/2+k)+1] = Lambda(slices, output_shape = (data[3],), arguments = {'start':int(r/2), 'end':r})(dr)
                        dict1i[2*int(j*data[4]/4**(i)/2+k)] = Lambda(slices, output_shape = (data[3],), arguments = {'start':0, 'end':int(r/2)})(di)
                        dict1i[2*int(j*data[4]/4**(i)/2+k)+1] = Lambda(slices, output_shape = (data[3],), arguments = {'start':int(r/2), 'end':r})(di)
                    else:
                        dict1r[int(j*data[4]/4**(i)/2+k)] = Add()([dr,ddr])
                        dict1i[int(j*data[4]/4**(i)/2+k)] = Add()([di,ddi])


            dict0r = dict1r
            dict0i = dict1i
            del dict1r
            del dict1i

        b2rl = dict0r[0]
        b2im = dict0i[0]
        for i in range(1,int(data[4]/2)):
        	b2rl = Concatenate(axis = 1)([b2rl,dict0r[i]])
        	b2im = Concatenate(axis = 1)([b2im,dict0i[i]])
        
        #brl2 = K.slice(brl,[0,int(N/2)],[M,N])
        #bim2 = K.slice(bim,[0,int(N/2)],[M,N])
        brl2 = Lambda(slices, output_shape = (int(N/2),), arguments = {'start':int(N/2), 'end':N})(brl)
        bim2 = Lambda(slices, output_shape = (int(N/2),), arguments = {'start':int(N/2), 'end':N})(bim)
        
        brl2 = Subtract()([brl2,b2rl])
        bim2 = Subtract()([bim2,b2im])

        (Linv_rl['A22'],Linv_im['A22'],xrl2,xim2) = Linvnet(Lrl['A22'],Lim['A22'],brl2,bim2,M,int(N/2))

        xrl = Concatenate(axis = 1)([xrl1,xrl2])
        xim = Concatenate(axis = 1)([xim1,xim2])
    return Linv_rl, Linv_im, xrl, xim


def Uinvnet(Url,Uim,brl,bim,M,N):
    #Uinv_rl = Url
    #Uinv_im = Uim
    if  Url['isLeaf'] == 1:
        Uinv_rl = {'isLeaf':1,'sz':Url['sz'],'szsub':Url['szsub'],'A11':0,'A12':0,'A21':0,'A22':0}
        Uinv_im = {'isLeaf':1,'sz':Uim['sz'],'szsub':Uim['szsub'],'A11':0,'A12':0,'A21':0,'A22':0}
        Uinv_rl['A22'] = Dense(int(N/2),use_bias = False, kernel_initializer = keras.initializers.Constant(Url['A22']), trainable = True)
        #brl1 = K.slice(brl,[0,0],[M,int(N/2)])
        brl2 = Lambda(slices, output_shape = (int(N/2),), arguments = {'start':int(N/2), 'end':N})(brl)
        xrl21 = Uinv_rl['A22'](brl2)
        Uinv_im['A22'] = Dense(int(N/2),use_bias = False, kernel_initializer = keras.initializers.Constant(Uim['A22']), trainable = True)
        #bim1 = K.slice(bim,[0,0],[M,int(N/2)])
        bim2 = Lambda(slices, output_shape = (int(N/2),), arguments = {'start':int(N/2), 'end':N})(bim)
        xrl22 = Uinv_im['A22'](bim2)
        xrl2 = Subtract()([xrl21,xrl22])
        xim21 = Uinv_rl['A22'](bim2)
        xim22 = Uinv_im['A22'](brl2)
        xim2 = Add()([xim21,xim22])

        Uinv_rl['A12'] = Dense(int(N/2),use_bias = False, kernel_initializer = keras.initializers.Constant(Url['A12']), trainable = True)
        Uinv_im['A12'] = Dense(int(N/2),use_bias = False, kernel_initializer = keras.initializers.Constant(Uim['A12']), trainable = True)
        b1rl1 = Uinv_rl['A12'](xrl2)
        b1rl2 = Uinv_im['A12'](xim2)
        b1rl = Subtract()([b1rl1,b1rl2])
        b1im1 = Uinv_rl['A12'](xim2)
        b1im2 = Uinv_im['A12'](xrl2)
        b1im = Add()([b1im1,b1im2])
        #brl2 = K.slice(brl,[0,int(N/2)],[M,N])
        #bim2 = K.slice(bim,[0,int(N/2)],[M,N])
        brl1 = Lambda(slices, output_shape = (int(N/2),), arguments = {'start':0, 'end':int(N/2)})(brl)
        bim1 = Lambda(slices, output_shape = (int(N/2),), arguments = {'start':0, 'end':int(N/2)})(bim)
        brl1 = Subtract()([brl1,b1rl])
        bim1 = Subtract()([bim1,b1im])

        Uinv_rl['A11'] = Dense(int(N/2),use_bias = False, kernel_initializer = keras.initializers.Constant(Url['A11']), trainable = True)
        xrl11 = Uinv_rl['A11'](brl1)
        Uinv_im['A11'] = Dense(int(N/2),use_bias = False, kernel_initializer = keras.initializers.Constant(Uim['A11']), trainable = True)
        xrl12 = Uinv_im['A11'](bim1)
        xrl1 = Subtract()([xrl11,xrl12])
        xim11 = Uinv_rl['A11'](bim1)
        xim12 = Uinv_im['A11'](brl1)
        xim1 = Add()([xim11,xim12])

        xrl = Concatenate(axis = 1)([xrl1,xrl2])
        xim = Concatenate(axis = 1)([xim1,xim2])
    else:
        Uinv_rl = {'isLeaf':0,'sz':Url['sz'],'szsub':Url['szsub'],'A11':0,'A12':0,'A21':0,'A22':0}
        Uinv_rl['A12'] = {'data':Url['A12']['data'],'U':Url['A12']['U'],'V':Url['A12']['V'],'S':Url['A12']['S']}
        Uinv_im = {'isLeaf':0,'sz':Uim['sz'],'szsub':Uim['szsub'],'A11':0,'A12':0,'A21':0,'A22':0}
        Uinv_im['A12'] = {'data':Uim['A12']['data'],'U':Uim['A12']['U'],'V':Uim['A12']['V'],'S':Uim['A12']['S']}

        #brl1 = K.slice(brl,[0,0],[M,int(N/2)])
        #bim1 = K.slice(bim,[0,0],[M,int(N/2)])
        brl2 = Lambda(slices, output_shape = (int(N/2),), arguments = {'start':int(N/2), 'end':N})(brl)
        bim2 = Lambda(slices, output_shape = (int(N/2),), arguments = {'start':int(N/2), 'end':N})(bim)
        (Uinv_rl['A22'],Uinv_im['A22'],xrl2,xim2) = Uinvnet(Url['A22'],Uim['A22'],brl2,bim2,M,int(N/2))


        data = Url['A12']['data']
        dict0r = init_dict(data[4])
        dict0i = init_dict(data[4])
        for i in range(data[4]):
            #dict0r[i] = K.slice(xrl1,[0,i*data[2]/2],[M,(i+1)*data[2]/2]) 
            #dict0i[i] = K.slice(xim1,[0,i*data[2]/2],[M,(i+1)*data[2]/2]) 
            dict0r[i] = Lambda(slices, output_shape = (int(data[2]/2),), arguments = {'start':int(i*data[2]/2), 'end':int((i+1)*data[2]/2)})(xrl2)
            dict0i[i] = Lambda(slices, output_shape = (int(data[2]/2),), arguments = {'start':int(i*data[2]/2), 'end':int((i+1)*data[2]/2)})(xim2)
        for i in range(data[1]):
            dict1r = init_dict(data[4])
            dict1i = init_dict(data[4])
            for j in range(4**(i)):
                for k in range(2):
                    for n in range(int(data[4]/4**(i)/2)):
                        xr = Concatenate(axis = 1)([dict0r[2*int(j*data[4]/4**(i)/2+n)],dict0r[2*int(j*data[4]/4**(i)/2+n)+1]])
                        xi = Concatenate(axis = 1)([dict0i[2*int(j*data[4]/4**(i)/2+n)],dict0i[2*int(j*data[4]/4**(i)/2+n)+1]])
                        Uinv_rl['A12']['V'][i][int((j+k/2)*data[4]/4**(i)+n)] = Dense(data[3],use_bias = False, kernel_initializer = keras.initializers.Constant(Url['A12']['V'][i][int((j+k/2)*data[4]/4**(i)+n)]), trainable = True)
                        Uinv_im['A12']['V'][i][int((j+k/2)*data[4]/4**(i)+n)] = Dense(data[3],use_bias = False, kernel_initializer = keras.initializers.Constant(Uim['A12']['V'][i][int((j+k/2)*data[4]/4**(i)+n)]), trainable = True)
                        dr1 = Uinv_rl['A12']['V'][i][int((j+k/2)*data[4]/4**(i)+n)](xr)
                        dr2 = Uinv_im['A12']['V'][i][int((j+k/2)*data[4]/4**(i)+n)](xi)
                        di1 = Uinv_rl['A12']['V'][i][int((j+k/2)*data[4]/4**(i)+n)](xi)
                        di2 = Uinv_im['A12']['V'][i][int((j+k/2)*data[4]/4**(i)+n)](xr)
                        dict1r[int((j+k/2)*data[4]/4**(i)+n)] = Subtract()([dr1,dr2])
                        dict1i[int((j+k/2)*data[4]/4**(i)+n)] = Add()([di1,di2])
            dict0r = dict1r
            dict0i = dict1i
            del dict1r
            del dict1i
        
        y = rearrange(0, 4**data[1])
        flag = int(data[5]/data[3])
        dict1r = init_dict(4**data[1])
        dict1i = init_dict(4**data[1])
        for i in range(4**data[1]):
            if  flag == 2:
                xr = Concatenate(axis = 1)([dict0r[2*y[i]],dict0r[2*y[i]+1]])
                xi = Concatenate(axis = 1)([dict0i[2*y[i]],dict0i[2*y[i]+1]])
            else:
                xr = dict0r[y[i]]
                xi = dict0i[y[i]]
            Uinv_rl['A12']['S'][i] = Dense(data[5],use_bias = False, kernel_initializer = keras.initializers.Constant(Url['A12']['S'][i]), trainable = True)
            Uinv_im['A12']['S'][i] = Dense(data[5],use_bias = False, kernel_initializer = keras.initializers.Constant(Uim['A12']['S'][i]), trainable = True)
            dr1 = Uinv_rl['A12']['S'][i](xr)
            dr2 = Uinv_im['A12']['S'][i](xi)
            dict1r[i] = Subtract()([dr1,dr2])
            di1 = Uinv_rl['A12']['S'][i](xi)
            di2 = Uinv_im['A12']['S'][i](xr)
            dict1i[i] = Add()([di1,di2])
        if  flag == 2:
            dict0r = init_dict(data[4])
            dict0i = init_dict(data[4])
            for i in range(4**data[1]):
                #dict0r[2*i] = K.slice(dict1r[i],[0,0],[M,data[3]])
                #dict0i[2*i] = K.slice(dict1i[i],[0,0],[M,data[3]])
                #dict0r[2*i+1] = K.slice(dict1r[i],[0,data[3]],[M,data[5]])
                #dict0i[2*i+1] = K.slice(dict1i[i],[0,data[3]],[M,data[5]])
                dict0r[2*i] = Lambda(slices, output_shape = (data[3],), arguments = {'start':0, 'end':data[3]})(dict1r[i])
                dict0i[2*i] = Lambda(slices, output_shape = (data[3],), arguments = {'start':0, 'end':data[3]})(dict1i[i])
                dict0r[2*i+1] = Lambda(slices, output_shape = (data[3],), arguments = {'start':data[3], 'end':data[5]})(dict1r[i])
                dict0i[2*i+1] = Lambda(slices, output_shape = (data[3],), arguments = {'start':data[3], 'end':data[5]})(dict1i[i])
        else:
            dict0r = dict1r
            dict0i = dict1i
        del dict1r
        del dict1i

        for i in range(data[1]-1,-1,-1):
            if  i == 0:
                r = data[2]
                dict1r = init_dict(int(data[4]/2))
                dict1i = init_dict(int(data[4]/2))
            else:
                r = 2*data[3]
                dict1r = init_dict(int(data[4]))
                dict1i = init_dict(int(data[4]))
            for j in range(4**(i)):
                for k in range(int(data[4]/4**(i)/2)):
                    Uinv_rl['A12']['U'][i][int(j*data[4]/4**(i)+k)] = Dense(r,use_bias = False, kernel_initializer = keras.initializers.Constant(Url['A12']['U'][i][int(j*data[4]/4**(i)+k)]), trainable = True)
                    Uinv_im['A12']['U'][i][int(j*data[4]/4**(i)+k)] = Dense(r,use_bias = False, kernel_initializer = keras.initializers.Constant(Uim['A12']['U'][i][int(j*data[4]/4**(i)+k)]), trainable = True)
                    dr1 = Uinv_rl['A12']['U'][i][int(j*data[4]/4**(i)+k)](dict0r[int(j*data[4]/4**(i)+k)])
                    dr2 = Uinv_im['A12']['U'][i][int(j*data[4]/4**(i)+k)](dict0i[int(j*data[4]/4**(i)+k)])
                    dr = Subtract()([dr1,dr2])
                    di1 = Uinv_rl['A12']['U'][i][int(j*data[4]/4**(i)+k)](dict0i[int(j*data[4]/4**(i)+k)])
                    di2 = Uinv_im['A12']['U'][i][int(j*data[4]/4**(i)+k)](dict0r[int(j*data[4]/4**(i)+k)])
                    di = Add()([di1,di2])
                    Uinv_rl['A12']['U'][i][int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)] = Dense(r,use_bias = False, kernel_initializer = keras.initializers.Constant(Url['A12']['U'][i][int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)]), trainable = True)
                    Uinv_im['A12']['U'][i][int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)] = Dense(r,use_bias = False, kernel_initializer = keras.initializers.Constant(Uim['A12']['U'][i][int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)]), trainable = True)
                    dr1 = Uinv_rl['A12']['U'][i][int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)](dict0r[int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)])
                    dr2 = Uinv_im['A12']['U'][i][int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)](dict0i[int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)])
                    ddr = Subtract()([dr1,dr2])
                    di1 = Uinv_rl['A12']['U'][i][int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)](dict0i[int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)])
                    di2 = Uinv_im['A12']['U'][i][int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)](dict0r[int(j*data[4]/4**(i)+k+data[4]/4**(i)/2)])
                    ddi = Add()([di1,di2])

                    #dict1r[int(j*data[4]/4**(i)/2+k)] = Add()([dr,ddr])
                    #dict1i[int(j*data[4]/4**(i)/2+k)] = Add()([di,ddi])
                    
                    if  i > 0:
                        dr = Add()([dr,ddr])
                        di = Add()([di,ddi])
                        dict1r[2*int(j*data[4]/4**(i)/2+k)] = Lambda(slices, output_shape = (data[3],), arguments = {'start':0, 'end':int(r/2)})(dr)
                        dict1r[2*int(j*data[4]/4**(i)/2+k)+1] = Lambda(slices, output_shape = (data[3],), arguments = {'start':int(r/2), 'end':r})(dr)
                        dict1i[2*int(j*data[4]/4**(i)/2+k)] = Lambda(slices, output_shape = (data[3],), arguments = {'start':0, 'end':int(r/2)})(di)
                        dict1i[2*int(j*data[4]/4**(i)/2+k)+1] = Lambda(slices, output_shape = (data[3],), arguments = {'start':int(r/2), 'end':r})(di)
                    else:
                        dict1r[int(j*data[4]/4**(i)/2+k)] = Add()([dr,ddr])
                        dict1i[int(j*data[4]/4**(i)/2+k)] = Add()([di,ddi])


            dict0r = dict1r
            dict0i = dict1i
            del dict1r
            del dict1i

        b1rl = dict0r[0]
        b1im = dict0i[0]
        for i in range(1,int(data[4]/2)):
            b1rl = Concatenate(axis = 1)([b1rl,dict0r[i]])
            b1im = Concatenate(axis = 1)([b1im,dict0i[i]])
        
        #brl2 = K.slice(brl,[0,int(N/2)],[M,N])
        #bim2 = K.slice(bim,[0,int(N/2)],[M,N])
        brl1 = Lambda(slices, output_shape = (int(N/2),), arguments = {'start':0, 'end':int(N/2)})(brl)
        bim1 = Lambda(slices, output_shape = (int(N/2),), arguments = {'start':0, 'end':int(N/2)})(bim)

        brl1 = Subtract()([brl1,b1rl])
        bim1 = Subtract()([bim1,b1im])

        (Uinv_rl['A11'],Uinv_im['A11'],xrl1,xim1) = Uinvnet(Url['A11'],Uim['A11'],brl1,bim1,M,int(N/2))

        xrl = Concatenate(axis = 1)([xrl1,xrl2])
        xim = Concatenate(axis = 1)([xim1,xim2])
    return Uinv_rl, Uinv_im, xrl, xim

def extract_L(Lnet):
    if  Lnet['isLeaf'] == 1:
        L = {'isLeaf':1,'sz':Lnet['sz'],'szsub':Lnet['szsub'],'A11':0,'A12':[],'A21':0,'A22':0}
        zz = Lnet['A11'].get_weights()
        L['A11'] = zz[0].flatten()
        zz = Lnet['A21'].get_weights()
        L['A21'] = zz[0].flatten()
        zz = Lnet['A22'].get_weights()
        L['A22'] = zz[0].flatten()
    else:
        L = {'isLeaf':0,'sz':Lnet['sz'],'szsub':Lnet['szsub'],'A11':0,'A12':[],'A21':0,'A22':0}
        L['A11'] = extract_L(Lnet['A11'])
        L['A22'] = extract_L(Lnet['A22'])
        L['A21'] = {'data':Lnet['A21']['data'],'U':0,'V':0,'S':0}
        data = L['A21']['data']
        L['A21']['U'] = init_list(data[1])
        L['A21']['V'] = init_list(data[1]) 
        for i in range(data[1]):
            L['A21']['U'][i] = init_list(data[4])
            L['A21']['V'][i] = init_list(data[4])
            for j in range(data[4]):
                zz = Lnet['A21']['V'][i][j].get_weights()
                L['A21']['V'][i][j] = zz[0].flatten()
                zz = Lnet['A21']['U'][i][j].get_weights()
                L['A21']['U'][i][j] = zz[0].flatten()
        L['A21']['S'] = init_list(4**data[1])
        for i in range(4**data[1]):
            zz = Lnet['A21']['S'][i].get_weights()
            L['A21']['S'][i] = zz[0].flatten()
    return L

def extract_U(Unet):
    if  Unet['isLeaf'] == 1:
        U = {'isLeaf':1,'sz':Unet['sz'],'szsub':Unet['szsub'],'A11':0,'A12':[],'A21':0,'A22':0}
        zz = Unet['A11'].get_weights()
        U['A11'] = zz[0].flatten()
        zz = Unet['A12'].get_weights()
        U['A12'] = zz[0].flatten()
        zz = Unet['A22'].get_weights()
        U['A22'] = zz[0].flatten()
    else:
        U = {'isLeaf':0,'sz':Unet['sz'],'szsub':Unet['szsub'],'A11':0,'A12':[],'A21':0,'A22':0}
        U['A11'] = extract_U(Unet['A11'])
        U['A22'] = extract_U(Unet['A22'])
        U['A12'] = {'data':Unet['A12']['data'],'U':0,'V':0,'S':0}
        data = U['A12']['data']
        U['A12']['U'] = init_list(data[1])
        U['A12']['V'] = init_list(data[1]) 
        for i in range(data[1]):
            U['A12']['U'][i] = init_list(data[4])
            U['A12']['V'][i] = init_list(data[4])
            for j in range(data[4]):
                zz = Unet['A12']['V'][i][j].get_weights()
                U['A12']['V'][i][j] = zz[0].flatten()
                zz = Unet['A12']['U'][i][j].get_weights()
                U['A12']['U'][i][j] = zz[0].flatten()
        U['A12']['S'] = init_list(4**data[1])
        for i in range(4**data[1]):
            zz = Unet['A12']['S'][i].get_weights()
            U['A12']['S'][i] = zz[0].flatten()
    return U


            		
