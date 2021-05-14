#!/usr/bin/env  python 

# 26 Jan

import tensorflow as tf
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
rnn_2 = 300
rnn_3 = 300

# 300, test acc 0.7 30mb h5 file
# 400 same acc but 48mb h5 file
# 2 layers, 300 acc 0,68 21 mb h5 file

##################################################################################################################  
# CREATE MODEL 1 . one head. functional
# BAD  batch norm, activation, dropout
# reinitialized_model = keras.models.model_from_json(json_config)
# ################################################################################################################
def create_model_1(hot_size, plot_file):

    print('\n\nmy model: create FUNCTIONAL model 1 simple LSTM, hot (aka softmax) size %d\n\n' %(hot_size))

    #reinitialized_model = keras.models.model_from_json(json_config)
    
    #number of param same order of magnitude as data set size. then increase dropout

    input_tensor = tf.keras.Input(shape=(config_bach.seqlen, hot_size), name='pitch')

    x = tf.keras.layers.LSTM(rnn_1, return_sequences=True, name='LSTM1') (input_tensor)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.LeakyReLU() (x)
    x = tf.keras.layers.Dropout(0.4) (x)

    x = tf.keras.layers.LSTM(rnn_2, return_sequences=False, name = 'LSTM2') (x)
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
    output_tensor = tf.keras.layers.Dense(hot_size, activation = 'softmax' , name ='softmax') (x)

    model = tf.keras.Model(input_tensor, output_tensor, name='model1')

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
    input_tensor = tf.keras.Input((config_bach.seqlen, hot_size))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_1, return_sequences=True, name='LSTM1')) (input_tensor)
    
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.LeakyReLU() (x)
    x = tf.keras.layers.Dropout(0.4) (x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_2, return_sequences=False, name = 'model1_bi')) (x)
    
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
###############################################

#https://keras.io/guides/training_with_built_in_methods/


# no velocity
# 2 softmax size as params
def create_model_2_nv(n_pitches, n_duration, plot_file):

    print('\n\nmy model: create FUNCTIONAL model2 without velocity. n_pitches %d, n_duration %d\n' %(n_pitches,n_duration))

    # 2 inputs
    pitch_input_tensor = tf.keras.Input(shape=(None,))
    duration_input_tensor = tf.keras.Input(shape=(None,))

    # embedding , integers into embedding vectors
    embed_size = 200
    

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
    x = tf.keras.layers.LSTM(rnn_2, return_sequences=True, name='LSTM2') (x) 
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
# add velocity. last is return = ?
# 3 softmax size as params

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
    x = tf.keras.layers.LSTM(rnn_2, return_sequences=True, name='LSTM2') (x) 
    
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