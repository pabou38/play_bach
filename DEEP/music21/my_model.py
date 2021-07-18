#!/usr/bin/env  python 

# 26 Jan

import tensorflow as tf
import numpy as np
import config_bach

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '..') # this is in above working dir
import pabou

"""
def my_loss(y_true,y_pred):
    # param are tensorflow tensors
    x = K.square(y_true - y_pred)
    x= K.sum(x,axis = -1)
    loss= K.sqrt(x) 
    return(loss)
"""

rnn_1 = 300
# number of hiden unit
# each cells step, i gets two inputs: xi and hidden unit step i-1
# hidden unit vector of size rnn
# last hiden unit is the 'summary' of the input sequence goes thru dense+softmax  
# in stacked RNN, 2nd level gets own hidden unit and input = hiden unit of previous layer 
# no attention.  data at the begining of the sequence may get lost
rnn_2 = 300
rnn_3 = 300

# 300, test acc 0.7 30mb h5 file
# 400 same acc but 48mb h5 file
# 2 layers, 300 acc 0,68 21 mb h5 file

##################################################################################################################  
# CREATE MODEL 1 . one head. functional
# BAD  batch norm, activation, dropout
# reinitialized_model = keras.models.model_from_json(json_config) 
# hot size is size of output layer, ie softmax
# input: seqlen of one hot 
# ################################################################################################################
def create_model_1(hot_size, plot_file):

    print('\n\nmy model: create FUNCTIONAL model 1 simple LSTM, hot (aka softmax) size %d\n\n' %(hot_size))

    #reinitialized_model = keras.models.model_from_json(json_config)
    #number of param same order of magnitude as data set size. then increase dropout

    # default dtype is float32
    input_tensor = tf.keras.Input(shape=(config_bach.seqlen, hot_size), name='pitch')

    x = tf.keras.layers.LSTM(rnn_1, return_sequences=True, name='LSTM1') (input_tensor)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.LeakyReLU() (x)
    x = tf.keras.layers.Dropout(0.4) (x)

    x = tf.keras.layers.LSTM(rnn_2, return_sequences=False,  name = 'LSTM2') (x)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.LeakyReLU() (x)
    x = tf.keras.layers.Dropout(0.4) (x)

    """
    x = tf.keras.layers.LSTM(rnn_3, return_sequences=False, name = 'LSTM3') (x)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.LeakyReLU() (x)
    x = tf.keras.layers.Dropout(0.4) (x)
    """      
    # sigmoid and binary_crossentropy as multi label;  Softmax and categorical for multi class
    # input = vector len = number of hidden unit, output = hotsize
    output_tensor = tf.keras.layers.Dense(hot_size, activation = 'softmax' , name ='softmax') (x)

    model = tf.keras.Model(input_tensor, output_tensor, name='model1')

    my_loss = 'categorical_crossentropy'
    my_optimizer = 'adam'
    #metrics = [tf.keras.metrics.Accuracy()]
    my_metrics = ['accuracy', 'CategoricalAccuracy'] # will be displayed in training logs. validation version if validation data is provided in fit
    
    print('compile model with metrics %s ' %(my_metrics))
    model.compile(optimizer=my_optimizer, loss = my_loss, metrics=my_metrics) 

    nb_params = pabou.see_model_info(model, plot_file) # info and plot model
    print ('my model: ======== >>>>>> we have to train %d params' %(nb_params))

    print('\nmy model: FUNCTIONAL MODEL CREATED , %d params' %nb_params)

    return(model, nb_params)


##################################################################################################################  
# CREATE MODEL 3 . one head. functional
# input seqlen of float32 or int32
# embedding layer on top
# BAD  batch norm, activation, dropout
# hot size is size of output layer, ie softmax
# vocab size, number of unique pitches
# boolean micro to simplify model for micro controler or edge
# ################################################################################################################
def create_model_3(hot_size, plot_file, micro=False):

    # vocab size is actually hot_size
    # embedding , integers into embedding vectors
    embed_size = 50 # size of embedding vector

    print('\n\nmy model: create FUNCTIONAL model 3. LSTM with embedding, softmax size %d, embedding dim %d\n\n' %(hot_size, embed_size))
    #reinitialized_model = keras.models.model_from_json(json_config)
    #number of param same order of magnitude as data set size. then increase dropout
    # no space in name

    # if (None, seqlen)  created dim mismatch
    # signature is 3 dims comming from lstm colab example  batch, seq, hot. so with this definition, will have 
    # x_test[0] array of int32,  if dtype not set, default to float32. error in gen data set
    #dtype	The data type expected by the input, as a string (float32, float64, int32...)

    ##############################################
    # WTF. when setting dtype to int16 or int32 (which is the default when vectorizing a list)
    # conversion to TPU dies silently
    # so let default float32, but then need to cast int32 to float 32 in the generator
    # this is only for conversion. the TPU model will requires uint8 input
    # BUTTTTTTTTTTT CAST op problem for tflite micro
    ##############################################
    
    input_tensor = tf.keras.Input(shape=(config_bach.seqlen), name='pitch', dtype='int32')

    #input_tensor = tf.keras.Input(shape=(config_bach.seqlen,), name='pitch') # default float32
    # <KerasTensor: shape=(None, 40) dtype=float32 (created by layer 'pitch')>

    # if shape = (seqlen,1) creates dim problem None,40,1,50  vs None,40,50


    # embedding layer
    # Turns positive integers (indexes) into dense vectors of fixed size.
    # input_dim: Integer. Size of the vocabulary, i.e. maximum integer index + 1.
    # output_dim: Integer. Dimension of the dense embedding.
    # input_length: Length of input sequences, when it is constant. This argument is required if you are going to connect Flatten then Dense layers upstream (without it, the shape of the dense outputs cannot be computed).

    #2D tensor with shape: (batch_size, input_length).
    #3D tensor with shape: (batch_size, input_length, output_dim).

    if micro == False:
        print('my model: model 3 ')
        x = tf.keras.layers.Embedding(input_dim=hot_size, output_dim=embed_size, name = 'embed') (input_tensor)

        x = tf.keras.layers.LSTM(rnn_1, return_sequences=True, name='LSTM1') (x)
        x = tf.keras.layers.BatchNormalization() (x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.Dropout(0.4) (x)

        x = tf.keras.layers.LSTM(rnn_3, return_sequences=False, name = 'LSTM2') (x)
        x = tf.keras.layers.BatchNormalization() (x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.Dropout(0.4) (x)

    else: # if single LSTM and return = True, cause error 

        print('my model: model 3 for micro controller')

        ########################################################
        # simpler model for microcontroler
        # only one LSTM, embedding ?
        ########################################################
        #x = tf.keras.layers.Embedding(input_dim=hot_size, output_dim=embed_size, name = 'embed') (input_tensor) # <KerasTensor: shape=(None, 40, 50) dtype=float32 (created by layer 'embed')>
        x = input_tensor

        x = tf.keras.layers.LSTM(rnn_1, return_sequences=False, name='LSTM1') (x) # <KerasTensor: shape=(None, 300) dtype=float32 (created by layer 'LSTM1')>
        x = tf.keras.layers.BatchNormalization() (x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.Dropout(0.2) (x)

    # sigmoid and binary_crossentropy as multi label;  Softmax and categorical for multi class
    output_tensor = tf.keras.layers.Dense(hot_size, activation = 'softmax' , name ='softmax') (x)

    model = tf.keras.Model(input_tensor, output_tensor, name='model3')

    my_loss = 'categorical_crossentropy'
    my_optimizer = 'adam'
    #metrics = [tf.keras.metrics.Accuracy()]
    my_metrics = ['accuracy', 'CategoricalAccuracy']
    
    print('my model: compile model with metrics %s ' %(my_metrics))
    model.compile(optimizer=my_optimizer, loss = my_loss, metrics=my_metrics) 

    nb_params = pabou.see_model_info(model, plot_file) # info and plot model

    print ('my model: input : %s' %(model.input)) # keras tensor

    # model.input <KerasTensor: shape=(None, 40) dtype=int32 (created by layer 'pitch')>
    # model.input.dtype tf.int32
    # model.input[0].dtype tf.int32
    # if only one input can use either input or inputs

    print ('my model: inputs: %s' %(model.inputs)) # inputS is a list
    #KerasTensor(type_spec=TensorSpec(shape=(None, 40), dtype=tf.int32, name='pitch'), name='pitch', description="created  by layer 'pitch'")
    print('\nmy model: FUNCTIONAL MODEL CREATED, %d params' %nb_params)

    #model outputs  [<KerasTensor: shape=(None, 95) dtype=float32 (created by layer 'softmax')>]

    return(model, nb_params)



##################################################################################################################  
# CREATE MODEL 4 . one head. functional
# treat a seqlen of n*n as a nxn matrix
# conv2D need 4 dim inputs
#model inputs  [<KerasTensor: shape=(None, 10, 10, 1) dtype=float32 (created by layer 'pitch')>]
#model outputs  [<KerasTensor: shape=(None, 95) dtype=float32 (created by layer 'softmax')>]
# ################################################################################################################
def create_model_4(hot_size, plot_file):
    
    print('my model: create FUNCTIONAL model 4. softmax size %d' %(hot_size))
    
    #input_tensor = tf.keras.Input(shape=(config_bach.size, config_bach.size, 1), name='pitch', dtype=tf.int32)
    #TypeError: Input 'filter' of 'Conv2D' Op has type float32 that does not match type int32 of argument 'input'.

    # conv2D requires float32 input
    
    input_tensor = tf.keras.Input(shape=(config_bach.size, config_bach.size, 1), name='pitch', dtype = tf.float32) # default is float32
    # uint8 TypeError: Value passed to parameter 'input' has DataType uint8 not in list of allowed values: float16, bfloat16, float32, float64, int32

    x = input_tensor

    x = tf.keras.layers.Conv2D(filters=64+16, kernel_size=(4,2), strides = (1,1) , padding='same', activation='relu') (x)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Dropout(0.2) (x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides = (1,1) , padding='same', activation='relu') (x)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Dropout(0.2) (x)

    #x = tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides = (1,1) , padding='same', activation='relu') (x)
    #x = tf.keras.layers.BatchNormalization() (x)
    #x = tf.keras.layers.Dropout(0.2) (x)

    x = tf.keras.layers.Flatten() (x)
    
    """
    NONE OF THIS WORKED ON MICRO
    # MICRO op code gathering missing if using embedding and flatten
    #x = tf.keras.layers.Embedding(input_dim=hot_size, output_dim=embed_size, name = 'embed') (x) #<KerasTensor: shape=(None, 40, 50) dtype=float32 (created by layer 'embed')>
    x = tf.keras.layers.Dense(256, activation='relu', name = 'dense') (x)
    x = tf.keras.layers.Dense(256, activation='relu', name = 'dense1') (x)
    #x = tf.keras.layers.Flatten() (x) # if embedding, if not, output is None,40,95  vs expected None,95

    # conv1D required a 3D tensor , ie input shape (seqlen, 1)
    # MICRO Didn't find op for builtin opcode 'EXPAND_DIMS' version '1'. An older version of this builtin might be supported. Are you using an old TFLite binary with a newer model?
    x = tf.keras.layers.Conv1D(filters = 32, kernel_size =10,  name='CONV1D1', activation='relu') (x) #<KerasTensor: shape=(None, 40, 32) dtype=float32 (created by layer 'CONV1D1')>
    x = tf.keras.layers.MaxPool1D(pool_size=2, strides=1, padding="valid") (x) #<KerasTensor: shape=(None, 39, 32) dtype=float32 (created by layer 'max_pooling1d')>
    x = tf.keras.layers.Conv1D(filters = 32, kernel_size =10,  name='CONV1D2', activation='relu') (x) # <KerasTensor: shape=(None, 39, 32) dtype=float32 (created by layer 'CONV1D2')>
    x = tf.keras.layers.GlobalMaxPool1D() (x) # remove step dimension

    #only one subgraph error in create interpreter
    #x=tf.keras.layers.GRU(32, dropout=0.2, recurrent_dropout=0.4, return_sequences= True, name='GRU1') (x)
    #x=tf.keras.layers.GRU(8, dropout=0.2, recurrent_dropout=0.4, return_sequences= False, name='GRU2' , activation='relu') (x)
    """

    output_tensor = tf.keras.layers.Dense(hot_size, activation = 'softmax' , name ='softmax') (x)

    model = tf.keras.Model(input_tensor, output_tensor, name='model4')

    my_loss = 'categorical_crossentropy'
    my_optimizer = 'RMSprop'
    #metrics = [tf.keras.metrics.Accuracy()]
    my_metrics = ['accuracy', 'CategoricalAccuracy']
    
    print('my model: compile model with metrics %s ' %(my_metrics))
    model.compile(optimizer=my_optimizer, loss = my_loss, metrics=my_metrics) 

    nb_params = pabou.see_model_info(model, plot_file) # info and plot model

    print ('my model: input : %s' %(model.input)) # keras tensor
    # model.input <KerasTensor: shape=(None, 40) dtype=int32 (created by layer 'pitch')>
    # model.input.dtype tf.int32
    # model.input[0].dtype tf.int32
    # if only one input can use either input or inputs


    # use this to make sure the input is later well formated
    print ('my model: inputs: %s' %(model.inputs)) # inputS is a list
    #KerasTensor(type_spec=TensorSpec(shape=(None, 40), dtype=tf.int32, name='pitch'), name='pitch', description="created  by layer 'pitch'")
    print ('my model: outputs: %s' %(model.outputs)) 
    
    print('\nmy model: FUNCTIONAL MODEL CREATED, %d params' %nb_params)

    return(model, nb_params)


############################
# same as model 1, but Bidirectional LSTM
# not used
#############################
def create_model_1_B(hot_size, plot_file):

    # hot size used to create last dense layer

    print('\n\nmy model: create FUNCTIONAL model 2, bidirectional LSTM, hot (aka softmax) size %d\n' %(hot_size))

    #reinitialized_model = keras.models.model_from_json(json_config)
    
    #number of param same order of magnitude as data set size. then increase dropout
    # input to Bidirectional is a layer, not a tensor.  x is a tensor
    #
    input_tensor = tf.keras.Input((config_bach.seqlen, hot_size), name='pitch')
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_1, return_sequences=True, name='LSTM1_bi')) (input_tensor)
    
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.LeakyReLU() (x)
    x = tf.keras.layers.Dropout(0.4) (x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_2, return_sequences=False, name = 'LSTM2_bi')) (x)
    
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.LeakyReLU() (x)
    x = tf.keras.layers.Dropout(0.4) (x)

        
    # sigmoid and binary_crossentropy as multi label;  Softmax and categorical for multi class
    output_tensor = tf.keras.layers.Dense(hot_size, activation = 'softmax' , name ='softmax') (x)

    model = tf.keras.Model(input_tensor, output_tensor)

    my_loss = 'categorical_crossentropy'
    my_optimizer = 'adam'
    #metrics = [tf.keras.metrics.Accuracy()]
    my_metrics = ['accuracy']

    print('compile model with metrics %s ' %(my_metrics))
    model.compile(optimizer=my_optimizer, loss = my_loss, metrics=my_metrics) 

    nb_params = pabou.see_model_info(model, plot_file) # info and plot model
    print ('my model: ======== >>>>>> we have to train %d params' %(nb_params))

    print('\nmy model: FUNCTIONAL MODEL CREATED , %d params' %nb_params)

    return(model, nb_params)


###############################################
# model 2
# RNN with attention, multi labels, embedding
# no velocity
# 2 softmax size as params
###############################################

#https://keras.io/guides/training_with_built_in_methods/

def create_model_2_nv(n_pitches, n_duration, plot_file):

    print('\n\nmy model: create FUNCTIONAL model2 without velocity. n_pitches %d, n_duration %d\n' %(n_pitches,n_duration))

    # 2 inputs
    pitch_input_tensor = tf.keras.Input(shape=(None,))
    duration_input_tensor = tf.keras.Input(shape=(None,))

    # embedding , integers into embedding vectors
    embed_size = 100
    
    # Turns positive integers (indexes) into dense vectors of fixed size.
    # input_dim: Integer. Size of the vocabulary, i.e. maximum integer index + 1.
    # output_dim: Integer. Dimension of the dense embedding.
    # input_length: Length of input sequences, when it is constant. This argument is required if you are going to connect Flatten then Dense layers upstream (without it, the shape of the dense outputs cannot be computed).

    #2D tensor with shape: (batch_size, input_length).
    #3D tensor with shape: (batch_size, input_length, output_dim).

    # no space in name
    x1 = tf.keras.layers.Embedding(input_dim=n_pitches, output_dim=embed_size, name = 'embed_pitch') (pitch_input_tensor)
    x2 = tf.keras.layers.Embedding(n_duration, embed_size, name = 'embed_duration') (duration_input_tensor)

    # one long vector input to RNN vs one hot in model 1
    x = tf.keras.layers.Concatenate()([x1,x2])

    # stacked LSTM. pass all hiden states , vs only last one
    x = tf.keras.layers.LSTM(rnn_1, return_sequences=True, name='LSTM1') (x)

    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.LeakyReLU() (x)
    x = tf.keras.layers.Dropout(0.4) (x)

    x = tf.keras.layers.LSTM(rnn_2, return_sequences=False, name='LSTM2') (x) 
    # if return is true, seems get get seqlen of softmax instead of one softmax. in that case take [0]?
    # TensorShape([None, None, 300])
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.LeakyReLU() (x)
    x = tf.keras.layers.Dropout(0.4) (x)

    # each hiden state (vector of len rnn_units) is passed thru an alignment function to generate a scalar
    e = tf.keras.layers.Dense(1, activation = 'tanh') (x) # alignment
    #TensorShape([None, None, 1])

    # Layer that reshapes inputs into the given shape.
    e = tf.keras.layers.Reshape([-1]) (e) # TensorShape([None, None])
    # vector or e1, e2  alignment value

    # vector of weight computed using softmax
    alpha = tf.keras.activations.softmax (e)   # TensorShape([None, None])

    # each hidden state vector multiplied by respective weight. the sum is a context vector
    # context vector , len rnn_units, is passed thru dense (n_notes) , with softmax

    c = tf.keras.layers.RepeatVector(rnn_1)(alpha) # TensorShape([None, 300, None]) (num_samples, features) to (num_samples, n, features)
    c = tf.keras.layers.Permute([2,1]) (c) # TensorShape([None, None, 300]) For instance, (2, 1) permutes the first and second dimensions of the input. input_shape=(10, 64) 64, 10)
    c = tf.keras.layers.Multiply() ([x,c]) # TensorShape([None, None, 300]) x TensorShape([None, None, 300])

    c = tf.keras.layers.Lambda( lambda xin: tf.keras.backend.sum(xin, axis=1), output_shape=(rnn_2,) ) (c)

    pitch_output_tensor = tf.keras.layers.Dense(n_pitches, activation = 'softmax' , name = 'pitch_output') (c)
    duration_output_tensor = tf.keras.layers.Dense(n_duration, activation = 'softmax' , name = 'duration_output') (c)

    # model has two head input and two head output
    model = tf.keras.Model([pitch_input_tensor, duration_input_tensor], [pitch_output_tensor,duration_output_tensor], name = 'model2_NV')

    # different losses for each heads
    my_loss = ['categorical_crossentropy', 'categorical_crossentropy']

    # name with dictionary
    #my_loss = {'pitch_output':'categorical_crossentropy', 'duration_output':'categorical_crossentropy'}

    my_optimizer = tf.keras.optimizers.RMSprop(lr= 0.001)

    my_metrics = ['accuracy']
    # one metric by head
    #my_metrics = ['accuracy' ,tf.keras.metrics.CategoricalAccuracy() ]

    print('compile model with metrics %s ' %(my_metrics))
    model.compile(optimizer = my_optimizer, loss = my_loss, metrics=my_metrics, loss_weights={"pitch_output": 2.0, "duration_output": 1.0}) 


    nb_params = pabou.see_model_info(model, plot_file) # info and plot model
    print ('my model: ======== >>>>>> we have to train %d params' %(nb_params))

    print('\nmy model: FUNCTIONAL MODEL CREATED , %d params' %nb_params)

    """
    Passing data to a multi-input or multi-output model in fit works in a similar way as specifying a loss function in compile: you can pass lists of NumPy arrays (with 1:1 mapping to the outputs that received a loss function) or dicts mapping output names to NumPy arrays.
    """

    return(model, nb_params)


#########################################################
# model 2 with velocity
# add velocity. last is return = ?
# 3 softmax size as params
##########################################################
def create_model_2(n_pitches, n_duration, n_velocity, plot_file):
    # param are size of embedding

    print('\n\nmy model: create FUNCTIONAL model 2 with velocity. n_pitches %d, n_duration %d n_velocity %d\n\n' %(n_pitches,n_duration,n_velocity))

    # 3 inputs
    #pitch_input_tensor = tf.keras.Input(shape=(None,), name="pitch") # should appear in TFlite input details
    # shape (steps, dim)
    pitch_input_tensor = tf.keras.Input(shape=(config_bach.seqlen,), name="pitch") # should appear in TFlite input details
    duration_input_tensor = tf.keras.Input(shape=(config_bach.seqlen,), name="duration")
    velocity_input_tensor = tf.keras.Input(shape=(config_bach.seqlen,), name = "velocity")

    # embedding , integers into embedding vectors
    embed_size = 200
    rnn_unit = 300

    # Turns positive integers (indexes) into dense vectors of fixed size.
    # input_dim: Integer. Size of the vocabulary, i.e. maximum integer index + 1.
    # output_dim: Integer. Dimension of the dense embedding.
    # input_length: Length of input sequences, when it is constant. This argument is required if you are going to connect Flatten then Dense layers upstream (without it, the shape of the dense outputs cannot be computed).

    #2D tensor with shape: (batch_size, input_length).
    #3D tensor with shape: (batch_size, input_length, output_dim).

    # NO SPACE IN LAYER MODEL NAME !!!!!!!!!!!!!!!!!!!!
    x1 = tf.keras.layers.Embedding(input_dim=n_pitches, output_dim=embed_size, name = 'embed_pitch') (pitch_input_tensor)
    x2 = tf.keras.layers.Embedding(n_duration, embed_size, name = 'embed_duration') (duration_input_tensor)
    x3 = tf.keras.layers.Embedding(n_velocity, embed_size, name = 'embed_velocity') (velocity_input_tensor)

    # one long vector input to RNN vs one hot in model 1
    x = tf.keras.layers.Concatenate()([x1,x2,x3])

    # stacked LSTM. pass all hiden states , vs only last one
    x = tf.keras.layers.LSTM(rnn_1, return_sequences=True, name='LSTM1') (x)



    x = tf.keras.layers.LSTM(rnn_2, return_sequences=False, name='LSTM2') (x) 
    
    # if return is true, seems get get seqlen of softmax instead of one softmax. in that case take [0]?
    # TensorShape([None, None, 300])

    # each hiden state (vector of len rnn_units) is passed thru an alignment function to generate a scalar
    e = tf.keras.layers.Dense(1, activation = 'tanh') (x) # alignment
    #TensorShape([None, None, 1])

    # Layer that reshapes inputs into the given shape.
    e = tf.keras.layers.Reshape([-1]) (e) # TensorShape([None, None])
    # vector or e1, e2  alignment value

    # vector of weight computed using softmax
    alpha = tf.keras.activations.softmax (e)   # TensorShape([None, None])

    # each hidden state vector multiplied by respective weight. the sum is a context vector
    # context vector , len rnn_units, is passed thru dense (n_notes) , with softmax

    c = tf.keras.layers.RepeatVector(rnn_1)(alpha) # TensorShape([None, 300, None]) (num_samples, features) to (num_samples, n, features)
    c = tf.keras.layers.Permute([2,1]) (c) # TensorShape([None, None, 300]) For instance, (2, 1) permutes the first and second dimensions of the input. input_shape=(10, 64) 64, 10)
    c = tf.keras.layers.Multiply() ([x,c]) # TensorShape([None, None, 300]) x TensorShape([None, None, 300])

    c = tf.keras.layers.Lambda( lambda xin: tf.keras.backend.sum(xin, axis=1), output_shape=(rnn_unit,) ) (c)

    # NO SPACE IN LAYER MODEL NAME !!!!!!!!!!!!!!!!!!!!
    pitch_output_tensor = tf.keras.layers.Dense(n_pitches, activation = 'softmax' , name = 'pitch_output') (c)
    duration_output_tensor = tf.keras.layers.Dense(n_duration, activation = 'softmax' , name = 'duration_output') (c)
    velocity_output_tensor = tf.keras.layers.Dense(n_velocity, activation = 'softmax' , name = 'velocity_output') (c)

    #  name of layer used in metrics


    """
    loss and val loss
    name_loss  name_accuracy
    val_name_loss  val_name_accuracy

    Epoch 46/60
    355/355 [==============================] - 11s 30ms/step - loss: 0.4680 - pitch_output_loss: 0.1904 - 
    duration_output_loss: 0.0689 - velocity_output_loss: 0.0182 - pitch_output_accuracy: 0.9496 - 
    duration_output_accuracy: 0.9790 - velocity_output_accuracy: 0.9946 - val_loss: 8.5875 - 
    val_pitch_output_loss: 3.7237 - val_duration_output_loss: 0.8487 - val_velocity_output_loss: 0.2915 - 
    val_pitch_output_accuracy: 0.4738 - val_duration_output_accuracy: 0.8501 - val_velocity_output_accuracy: 0.9451
    """
   
    # NO SPACE IN LAYER MODEL NAME !!!!!!!!!!!!!!!!!!!!
    model = tf.keras.Model([pitch_input_tensor, duration_input_tensor, velocity_input_tensor], [pitch_output_tensor,duration_output_tensor,velocity_output_tensor], name = 'model2_V')

    # different losses for each heads
    my_loss = ['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy' ]

    # name with dictionary
    #my_loss = {'pitch_output':'categorical_crossentropy', 'duration_output':'categorical_crossentropy'}

    my_optimizer = tf.keras.optimizers.RMSprop(lr= 0.001)

    my_metrics = ['accuracy']
    # one metric by head
    #my_metrics = ['accuracy' ,tf.keras.metrics.CategoricalAccuracy() ]

    print('my model: compile model with metrics %s ' %(my_metrics))
    model.compile(optimizer = my_optimizer, loss = my_loss, metrics=my_metrics, loss_weights={"pitch_output": 2.0, "duration_output": 1.0}) 

    nb_params = pabou.see_model_info(model, plot_file) # info and plot model
    print ('my model: ======== >>>>>> we have to train %d params' %(nb_params))

    print('\nmy model: FUNCTIONAL MODEL CREATED , %d params' %nb_params)

    """
    Passing data to a multi-input or multi-output model in fit works in a similar way as specifying a loss function in compile: you can pass lists of NumPy arrays (with 1:1 mapping to the outputs that received a loss function) or dicts mapping output names to NumPy arrays.
    """

    return(model, nb_params)


"""
##################################################################################################################  
# CREATE MODEL 1 . one head. sequential
# BAD  batch norm, activation, dropout

#reinitialized_model = keras.models.model_from_json(json_config)
# ################################################################################################################
def create_model_1_s(hot_size, plot_file): 

    print('\n\nmy model: create SEQUENTIAL model, hot size %d' %(hot_size))
    
    #number of param same order of magnitude as data set size. then increase dropout
    model = tf.keras.models.Sequential()
   
    model.add(tf.keras.layers.LSTM(rnn_1, input_shape=(config_bach.seqlen, hot_size),return_sequences=True))
   
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3)) # before next layer ?
    
    model.add(tf.keras.layers.LSTM(rnn_2))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
        
    #loss = 'kullback_leibler_divergence'
    #loss = my_loss
    
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    optimizer = 'adam'
    #metrics = [tf.keras.metrics.Accuracy()]
    metrics = ['accuracy']
        
    # sigmoid and binary_crossentropy as multi label;  Softmax and categorical for multi class
    model.add(tf.keras.layers.Dense(hot_size,activation= activation))
    
    print('compile model with metrics %s ' %(metrics))
    model.compile(optimizer=optimizer, loss = loss, metrics=metrics) 
    #model.compile(loss=tf.keras.losses.MSE , optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), metrics=[tf.keras.metrics.categorical_accuracy], sample_weight_mode='temporal')

    nb_params = pabou.see_model_info(model, plot_file) # info and plot model
    print ('my model: ======== >>>>>> we have to train %d params' %(nb_params))
    
    print('\nmy model: SEQUENTIAL MODEL CREATED')

    return(model, nb_params)
"""

"""
#https://ai.stackexchange.com/questions/3156/how-to-select-number-of-hidden-layers-and-number-of-memory-cells-in-lstm
# upper bound on number of hidden neuron = number of sample in training / (number of input neuron + number of output neuron) * 2 to 10
#440000 input 100, output 300    440000  / 400 * 2   500 to 200

256 input neurons 300 output neurons    600   600x2  to 600x10   1200 to 6000
number of samples in training set 340k
340k/1200   340k/6000   280 to 56  upper bound to number of hidden neuron that won't result in over-fitting
256 LSTM (dense is not hidden)
"""

"""
    use mean, std from training, not from this sample
    
    print('load training mean and std from json %s' %(config_bach.mean_std_json_file))   
    with open(config_bach.mean_std_json_file) as json_file:
        data = json.load(json_file)
    
    mean = data['mean']
    std = data['std']
    print('from training:', data , mean, std)
    X = X - mean
    # check mean X is now zero
    a= np.atleast_1d(np.mean(X))
    b= np.atleast_1d(0)
    np.testing.assert_allclose(a,b,atol=1e-5) # mean X should now be zero absolute difference
    
    X = X / std # shape (50,)   ndim 1 dtype float64
    
    # check std X is now 1
    a= np.atleast_1d(np.std(X))
    b= np.atleast_1d(1)
    np.testing.assert_allclose(a,b,atol=1e-5) # std X should now be 1 absolute difference
"""


"""
if model_type == 3: # Y is pi and duration
        print('type 2. multi hot')
        
        print ('create multilabel binarizer, fit on entiere corpus os (str,str), and convert output_list to 2 hot')      
        # multilabel binarizer 
        #https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/
        # labels is list of (str,str) . create two hots labels = [ "blue", "jeans"), ("blue", "dress"),...
        mlb = MultiLabelBinarizer() 
        mlb.fit(corpus_list) # list of [str,str]
        # mlb.classes_ is an array with all individuals pi and du, in same array
        print ('all labels classes, ie pitches and duration, len %d' %(len(mlb.classes_)))
        #print('all pu and du labels ' , mlb.classes_)
        # mlb.transform([(p,d)]) convert to 2 hots

        # convert tuple of 2 strings into one two hot
        Y=mlb.transform(network_output_list)
        
        X=np.reshape(network_input_list, (len(network_input_list) , config_bach.seqlen, 2) ) # shape (xxx,50,2)
"""