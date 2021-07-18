#!/usr/bin/python3
# -*- coding: utf-8 -*-

# 30 jan 2021

#################################
# nvidia-smi: driver, CUDA release
# nvcc --version: CUDA release eg 11.2
# in (tf24) conda list cudnn
#################################

##############################################
# generic functions, to be used across applications
# one level up to the main applications sources

# files names definition
# parse arguments
# see training results
# see model infos, tensorflow infos, data inspect 
# model accuracy
# save, load full model
# get h5 size
# add tflite meta data
# see tflite tensors
# representative dataset generator
# save all tflite models
# tflite single inference
# benchmark full model
# benchmarl tflite one model
# evaluate model
# create pretty table
##############################################

"""
#!/usr/bin/env python
"""

"""
/usr/bin/python is absolute path
env look at env variable. how to set PYTHON env variable?
.bashrc alias python="/usr/bin/python3.6"
"""
# pip install prettytable

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

import numpy as np
import math

import matplotlib.pyplot as plt
#%matplotlib inline  # interactive in jupyter and colab
import tensorflow as tf
import numpy as np
import time
import platform
import argparse
import sklearn
import scipy
from prettytable import PrettyTable
# from (tf2) pip3 install PTable. conda install PTable does not work
# http://zetcode.com/python/prettytable/
from datetime import date

from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops.gen_array_ops import list_diff_eager_fallback

today = date.today()
title = today.strftime("%d/%m/%Y")

#################################################
# file names definition
#################################################

# suffix file names.
# file names are models/app1_c_o + <suffix> 
full_model_SavedModel_dir = "_full_model"
full_model_h5_file = '_full_model.h5' # will be saved as .h5py
model_json_file = '_full_model.json'
lite_model_file = '_lite.tflite'

#will be prefixed by models/app
label_txt = '_labels.txt' # dictionnary of labels, to be included in metadata
micro_model_file = '_micro.cpp' # for microcontroler
corpus_cc = 'corpus.h'
dict_cc = 'dictionary.h'



# GUI corpus built by removing  _full_model.h5  and .tflite
# need to find a way to distinguish lite files in GUI, where the .tflite extension is lost
# if a corpus ends with '_lite' it is a TFlite. test in main 

# TFlite suffix file name. will be prefixed by models/app
# app is id1_c_o and is the same for full and lite
# append _fp32_lite.tflite to app
tflite_fp32_file = '_fp32' + lite_model_file # no quant fp32 only
tflite_default_file = '_default'  + lite_model_file # default quant variable param still float
tflite_default_representative_file = '_adapt' + lite_model_file # variable param quantized
tflite_int_fp_fallback_file = '_quant-fp' + lite_model_file
tflite_int_only_file = '_TPU' + lite_model_file # TPU, micro
tflite_FP16_file = '_GPU' + lite_model_file # GPU delegate
tflite_16by8_file = '_16by8' + lite_model_file # similar to int only, 

# all tflite files
all_tflite_file = [tflite_int_only_file, tflite_fp32_file,
tflite_default_file, tflite_default_representative_file, 
tflite_int_fp_fallback_file, tflite_FP16_file, tflite_16by8_file]

# file system name is app + tflite_...._file, ie cello1_nc_mo _GPU _lite.tflite

# edgeTPU file (output from compiler)     bla.tflite => bla_edgeptu.tflite
#_TPU_lite.tflite  => _TPU_lite_edgetpu.tflite

tflite_edge_file = tflite_int_only_file.replace('lite.tflite','lite_edgetpu.tflite')


#######################################################
# parse CLI arguments 
# set CLI in launch.json for vscode
# set CLI with -xxx for colab
######################################################

def parse_arg(model_type): # parameter not used yet
    parser = argparse.ArgumentParser()

    # assumes we can pass CLI arguments in colab
    # default are for run time, ie not for training. load, no fit, no predict, no save TFlite, no benchmark
    
    parser.add_argument("-l", "--load_model", type=int, choices=[0, 1], default=1,  help="optional. default 1.  use -l=1 to load an already trained SavedModel or h5. use -l=0 to create empty model")
    parser.add_argument("-f", "--fit", help="optional. -f to fit model. default FALSE, ie do not fit or refit", action="store_true", default =0)
    parser.add_argument("-pm", "--predict_midi", help="optional. -pm to predict with FULL model, benchmark and create midi file. default FALSE", action="store_true", default =0)
    parser.add_argument("-bf", "--benchmark_full", help="optional. -b to benchmark full model. default FALSE", action="store_true", default =0)
    parser.add_argument("-bl", "--benchmark_lite", help="optional. -b to benchmark all lite model. default FALSE", action="store_true", default =0)
    parser.add_argument("-sl", "--savelite", action="store_true", default =0,   help="optional. default FALSE. use -sl to convert SaveModel to TFlite models")
    parser.add_argument("-st", "--stream", action="store_true", default =0,   help="optional. default FALSE. use -st to initiate real time stream")
    # use of TFlite for prediction is thru GUI

    parser.add_argument("-e", "--epochs", type=int, help="optional. use -e=5 to overwrite number of epochs in config file. DEFAULT=10. using debug mode will set to small number anyway")
    
    # return from parsing is NOT a dict, a namespace
    parsed_args=parser.parse_args() #Namespace(load=0, new=True, predict=0, retrain=0) NOT A DICT
    # can access as parsed_args.new if stay as namespace
    
    parsed_args = vars(parsed_args) # convert object to dict. get __dict__ attribute
    print('parsed argument as dictionary: ', parsed_args)
    
    print('keys:' , end = ' ')
    for i in parsed_args.keys():
        print (i , end = ' ')
    print('\n')
    
    return(parsed_args) # dict



####################################
# see training result. ie plot
####################################

# history give accuracy history. test is from evaluate (on test data)
# acc , val_acc are list. test_acc scalar

def see_training_result(history, test_acc , png_file, model_type):
    # test_acc is list of 3 floats . scalar 
    # for model 1, 2nd and 3rd are 0

    # train, val is from history . list

    # model 1: test_acc[0] is pitch, test_acc[1] is 0 , test_acc[2] is 0
    # model 2: test_acc[0] is pitch, test_acc[1] is duration . pass velocity as test_acc[2] or 0 if not used

    # history is a keras object
    # acc , val_acc are float list. test_acc scalar

    ########################################
    # WARNING:  this module is not yet generic, as dictionary for model 2 is specific to bach
    # if sofmax layer name is bla we get bla_accuracy and val_bla_acuracy
    #########################################

    """
    history_dict
    {'duration_output_accuracy': [0.9240724], 'duration_output_cate...l_accuracy': [0.9240724], 
    'duration_output_loss': [0.2427445], 'loss': [1.5433628956109549], 'lr': [0.001], 'pitch_output_accuracy': [0.80364454], 
    'pitch_output_categor...l_accuracy': [0.80364454], 'pitch_output_loss': [0.65030944], 
    'val_duration_output_accuracy': [0.83225507], 'val_duration_output_...l_accuracy': [0.83225507], 
    'val_duration_output_loss': [0.63128304], 'val_loss': [5.073272387774286], 'val_pitch_output_accuracy': [0.478435], 
    'val_pitch_output_cat...l_accuracy': [0.478435], ...}

    accuracy and categorical accuracy are the same
    """

    #plt.ion() # non blocking
    history_dict=history.history
    print ("PABOU: history dictionary keys after fit: %s: " %(history_dict.keys()))
    #print ("history dictionary value after fit: %s: " %(history_dict.values()))
    #print ("history dictionary after fit: %s: " %(history_dict))
    

    # !!!!!   acc on colab  accuracy on windows

    loss = history_dict['loss'] # always there, but no global accuracy ?
    val_loss = history_dict['val_loss']
     
    if model_type in[1,3,4]:
        """
        history dictionary keys: dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy', 'lr']) :
        """
        validation_acc=history_dict['val_accuracy'] # is a list, one entry per epoch
        training_acc=history_dict['accuracy']
        e = len(training_acc)
        
        print ("PABOU: ACCURACY:  training (last) %0.2f, validation (last) %0.2f, test %0.2f " %(training_acc[-1], validation_acc[-1], test_acc[0]))
        e = len(training_acc)
        print('PABOU: actuals EPOCHS: ' , e) # in case of early exit, epoch not 60

    
    if model_type == 2: 

        # get list of float from history dictionary
        # names are from model definition. find a way to query ?

        # pitch_output is the name of softmax layer in model definition
        # pitch_output_accuracy, val_pitch_output_accuracy
        # loss , val_loss 

        training_pitch_accuracy=history_dict['pitch_output_accuracy']
        training_duration_accuracy=history_dict['duration_output_accuracy']

        validation_pitch_accuracy=history_dict['val_pitch_output_accuracy']
        validation_duration_accuracy=history_dict['val_duration_output_accuracy']

        e = len(training_pitch_accuracy) # number of epochs

        try:
            training_velocity_accuracy=history_dict['velocity_output_accuracy'] # if there, use it
            validation_velocity_accuracy=history_dict['val_velocity_output_accuracy']
            print('found history for velocity')
        except:
            print('no history for velocity training') # otherwize dummy list
            training_velocity_accuracy = [0] * e # create a dummy list with dummy value
            validation_velocity_accuracy = [0] * e # create a dummy list with dummy value
            
        
        e = len(training_pitch_accuracy)
        # provision for 3 heads
        
        
        print ("PABOU: ACCURACY (last epoch): pitch,duration, velocity.  training: %0.2f %0.2f %0.2f, validation: %0.2f %0.2f %0.2f, test: %0.2f %0.2f %0.2f" \
            %(training_pitch_accuracy[-1], training_duration_accuracy[-1] , training_velocity_accuracy[-1] , \
             validation_pitch_accuracy[-1], validation_duration_accuracy[-1] , validation_velocity_accuracy[-1] , \
            test_acc[0] , test_acc[1], test_acc[2]))

        print('PABOU: actuals EPOCHS: ' , e) # in case of early exit, epoch not 60

    epochs = range (1, e + 1)

    # new window
    # size in inch, create figure 1
    figure= plt.figure(1,figsize=(10,10), facecolor="blue", edgecolor = "red" ) # figsize = size of box
    plt.tight_layout()
    plt.style.use("ggplot")
    plt.grid(True)
    plt.suptitle(title)
        
    """
    simple
    plt.plot(x_values, y_values, 'b.')
    plt.plot(x_test, y_test, 'r.', label="Test")   y.  g.  bo ro bx gx
    # plt.plot, .scatter , .bar
    """

    if model_type in [1,3,4]:   

        # loss and single accuracy

        plt.subplot(2,1,1) # 2 subplot 1,1, 1,2 horizontal stack.  1,1 this is the top one
         # Three integers (nrows, ncols, index). The subplot will take the index position on a grid with nrows rows and ncols columns. index starts at 1 
        
        plt.plot(epochs, validation_acc, color="blue", linewidth=2 , linestyle="-", marker = 'o', markersize=8,  alpha = 0.5, label = 'validation acc')
        plt.plot(epochs, training_acc, color="red", linewidth=2 , linestyle="--", marker = 'x', markersize=8,  alpha = 0.5, label = 'training acc')
        plt.legend(loc = "lower left", fontsize = 10)
        plt.ylabel('accuracy')
        plt.xlabel("epoch")
        #plt.draw()    # to continue computation.  show() will block

        plt.subplot(2,1,2) # 1,1 was top   1,2 is bottom
        plt.plot(epochs, val_loss, color="blue", linewidth=2 , linestyle="-", marker = 'o', markersize=8,  alpha = 0.5, label = 'validation loss')
        plt.plot(epochs, loss, color="red", linewidth=2 , linestyle="--", marker = 'x', markersize=8,  alpha = 0.5, label = 'training loss')
        plt.legend(loc = "lower left", fontsize = 10)
        plt.ylabel('loss')
        plt.xlabel("epoch")
        #plt.draw()    # to continue computation.  show() will block

        print('PABOU: save training to file ', png_file)
        figure.savefig(png_file)

    """
    plt.show() will display the current figure that you are working on.
    plt.draw() will re-draw the figure. This allows you to work in interactive mode and, should you have changed your data or formatting, allow the graph itself to change.
    """


    if model_type == 2:   # provision for 4 subplots 

        # 3 accuracy and loss

        # Three integers (nrows, ncols, index). The subplot will take the index position on a grid with nrows rows and ncols columns. index starts at 1 
        plt.subplot(2,2,1) #  this is the first one
        plt.plot(epochs, validation_pitch_accuracy, color="blue", linewidth=2 , linestyle="-", marker = 'o', markersize=8,  alpha = 0.5, label = 'pitch validation acc')
        plt.plot(epochs, training_pitch_accuracy, color="red", linewidth=2 , linestyle="--", marker = 'x', markersize=8,  alpha = 0.5, label = 'pitch training acc')
        
        plt.legend(loc = "lower left", fontsize = 10)
        plt.ylabel('accuracy')
        plt.xlabel("epoch")
        plt.draw()    # to continue computation.  show() will block
    
        plt.subplot(2,2,2)
        plt.plot(epochs, validation_duration_accuracy, marker = 'o', color="blue",  label = "duration validation acc")
        plt.plot(epochs, training_duration_accuracy, marker = 'x', color="red",  label = "duration training acc")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.draw()

        plt.subplot(2,2,3)
        plt.plot(epochs, validation_velocity_accuracy, marker = 'o', color="blue",  label = "velocity validation acc")
        plt.plot(epochs, training_velocity_accuracy, marker = 'x', color="red",  label = "velocity training acc")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.draw()


        plt.subplot(2,2,4)
        plt.plot(epochs, loss, marker = 'o', color="blue",  label = "loss")
        plt.plot(epochs, val_loss, marker = 'x', color="red",  label = "val_loss")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.draw()

        print('PABOU: save training histoty plot ', png_file)
        figure.savefig(png_file)
            
    #################################
    print('PABOU: plot training history')
    # comment to not block on display
    #plt.show(block=False)
    #plt.show()
    #################################

######################################################
# print various info. return os and platform
######################################################
def print_tf_info():
    print ("\nPABOU: ========= > tensorflow version: < ============ ",tf.__version__)
    #https://blog.tensorflow.org/2020/12/whats-new-in-tensorflow-24.html
    # TensorFlow 2.4 supports CUDA 11 and cuDNN 8,

    if float(tf.__version__[:3]) <= 2.3:
        print('warning: running below 2.3')

    """
    print('\nget cuda version')
    #bug in 2.3. ok in 2.1
    #https://github.com/tensorflow/tensorflow/issues/26395
    try: 
        from tensorflow.python.platform import build_info as tf_build_info
        print('cuda ', tf_build_info.cuda_version_number)
        print('cudnn ', tf_build_info.cudnn_version_number)
    except Exception as e:
        print('PABOU Exception cuda cudnn version not available ' , str(e))
    """

    import tensorflow.python.platform.build_info as build
    print('PABOU: CUDA ', build.build_info['cuda_version'])
    print('PABOU: cudnn ',build.build_info['cudnn_version'])

    print('\nPABOU: list devices')
    from tensorflow.python.client import device_lib
    print ("\nPABOU: devices available:\n ", str(device_lib.list_local_devices()))

    print('\nPABOU: test is gpu is available')
    try:
        device_name = tf.test.gpu_device_name()
        print('PABOU: list all gpu: ', tf.config.list_physical_devices('GPU'))
        print('PABOU: gpu device name: %s' %(device_name))
        print('PABOU: build with gpu support ? ',  tf.test.is_built_with_gpu_support()) # built with GPU (i.e. CUDA or ROCm) support
        # do  not use, will be deprecated
        #print("is gpu available ? %s\n\n" % tf.test.is_gpu_available()) # deprecated
    except:
        print("PABOU: error gpu")

    if device_name != '/device:GPU:0':
        print ('GPU device not found')
    print('======= > Found GPU at: {}'.format(device_name))

    print ("PABOU: numpy ", np.__version__)
    print ("PABOU: sklearn ", sklearn.__version__) 
    print ("PABOU: scipy", scipy.__version__)  
    
    print('PABOU: path for module to be imported: %s\n' %(sys.path))
    print('PABOU: path for executable: %s\n' % os.environ['PATH'])
    
    print('PABOU: python executable: %s' %( sys.executable))
    print('PABOU: python version: %s' %(sys.version))
    
    print('PABOU: os: %s, sys.platform: %s, platform.machine: %s' % (os.name, sys.platform, platform.machine()))
    print('PABOU: system name: %s' %( platform.node()))
    print(' ')
    # COLAB os = posix sys.platform = linux for
    # w10 os = nt, sys.platform = win32, platform.machine = AMD64

    return(os.name, sys.platform, platform.machine)



######################################################
# look at struncture of object
######################################################
def inspect(s,a):
    if isinstance(a,np.ndarray):
        try:
            print ("%s is NUMPY. len: %s, shape: %s, dim: %s, dtype: %s, min: %0.3f, max: %0.3f mean: %0.2f, std: %0.2f" %(s, len(a), a.shape, a.ndim, a.dtype, np.min(a), np.max(a), np.mean(a), np.std(a)))
        except:
            print ("%s is NUMPY. len %s, shape %s, dim %s, dtype %s" %(s, len(a), a.shape, a.ndim, a.dtype ))
            
    elif isinstance(a,list):
        print ("%s is a LIST. len %d, type %s" % (s, len(a), type(a[0]) ) )
        
    elif isinstance(a,dict):
        print ("%s is DICT. len %d" %(s, len(a)) )
        
    else: # int has no len
        print ("%s:  type %s" %(s,type(a)))
        

######################################################
# print various info on model just compliled, also plot model
######################################################

def see_model_info(model, plot_file):
    # add /home/pabou/anaconda3/envs/gpu/lib/graphviz manually in PATH in .bashrc
    print('PABOU: plot model')
    tf.keras.utils.plot_model(model,to_file=plot_file,show_shapes=True)

    model.summary()

    print ("PABOU: number of parameters: ", model.count_params())
    print("PABOU: number of trainable weights " , len(model.trainable_weights))

    print ("PABOU: model metrics: ", model.metrics_names)
    for layer in model.layers:
        print("\tPABOU:layer output shape: ",layer.output_shape)

    print ("PABOU: model inputs ", model.inputs)
    print ("PABOU: model outputs ", model.outputs)

    print("PABOU: number of layers " , len(model.layers))
    print("PABOU: layers list ", model.layers)

    return(model.count_params())
    

##################################################
# model accuracy from any dataset
# uses logit and keras accuracy
# model(x) returns logits = softmax for our model
##################################################
def see_model_accuracy(model, x_test, y_test):
    try:
        logits = model(x_test) #  array of prediction, ie softmax for MNIST: TensorShape([48000, 10]) 
        # get prediction label from softmax
        prediction = np.argmax(logits, axis=1) # array([3, 6, 6, ..., 0, 4, 9], dtype=int64) (48000,)

        # MNIST y_test array([3, 6, 6, ..., 0, 4, 9], dtype=uint8)

        # test if if label is one hot (not for MNIST)
        try:
            truth = np.argmax(y_test, axis=1) # assumes Y is one hot
        except:
            truth = y_test
        keras_accuracy = tf.keras.metrics.Accuracy()
        keras_accuracy(prediction, truth)

        # result() is tensor. convert to float

        print("PABOU: Raw model accuracy: %0.2f" %(keras_accuracy.result()))
    except Exception as e:
        print('PABOU: Exception see model accuracy ' , str(e))

    
######################################################
# save FULL model. SavedModel, h5, weigth from checkpoints (999), json
# tflite model are saved in save_all_tflite_models, after conversion from SavedModel
# WARNING: SavedModel have a signature for output, add a dimension as per TFlite [1,40,129]
# not added for h5 model, which are used as Full model
######################################################
# generic save FULL model in various forms. save in Saved Model (dir), hf5(file) and json
# app is a string, identifies the app, make file names unique, even if not in same dir
# checkpoint path is handled in main,path are /ddd or \nnnn 
def save_full_model(app, model, checkpoint_path): # app is a string

    # model 1 input is TensorShape([None, 40, 95])
    # model 3 TensorShape([None, 40])

    ##############################################
    # check if generic enough
    ##############################################
    print ("model inputs ", model.inputs) #  [0] is <KerasTensor: shape=(None, 40, 483) dtype=float32 (created by layer 'pitch')>]
    print ("model outputs ", model.outputs) #  [<KerasTensor: shape=(None, 483) dtype=float32 (created by layer 'softmax')>]
    seqlen = model.inputs[0].shape[1] # do not import config_bach to stay generic
    
    if len(model.inputs[0].shape) > 2:
        hotsize = model.inputs[0].shape[2]
    else: # model 3
        hotsize = 1 # is input size = 1 (ie seqlen of int the same of dim = 2 ?)
   
    print('PABOU: saving full model for: %s, seqlen %d, hotsize %d' %(app ,seqlen, hotsize))
    # return size of h5 file
    # suffix with _ done here
    # app is set to cello 1|2 nc nv mo|so as defined  config_bach.py
    # seqlen and hotsize are needed for signature
    
    tf_dir = app +  full_model_SavedModel_dir
    h5_file = app +  full_model_h5_file # will be saved as .h5py
    json_file = app  + model_json_file

    tf_dir = os.path.join('models' , tf_dir)
    h5_file = os.path.join('models' , h5_file)
    json_file = os.path.join('models' , json_file)

    # MODEL FULL save. ONLY AT THE END of FIT
    # export a whole model to the TensorFlow SavedModel format. SavedModel is a standalone serialization format for TensorFlow objects, supported by TensorFlow serving as well as TensorFlow implementations other than Python.
    # Note that the optimizer state is preserved as well: you can resume training where you left off.
    #The SavedModel files that were created contain: A TensorFlow checkpoint containing the model weights. A SavedModel proto containing the underlying TensorFlow graph.
    
    print('\nPABOU: save full model as SavedModel. directory is: ' , tf_dir)
    
    #################################################################
    # Keras LSTM fusion Codelab.ipynb
    #https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/experimental_new_converter/Keras_LSTM_fusion_Codelab.ipynb#scrollTo=tB1NZBUHDogR
    run_model = tf.function(lambda x: model(x))
    # This is important, let's fix the input size.
    BATCH_SIZE = 1
    STEPS = seqlen
    INPUT_SIZE = hotsize # is input size = 1 (ie seqlen of int the same of dim = 2 ?)
    signature = run_model.get_concrete_function(
     tf.TensorSpec(shape = [BATCH_SIZE, STEPS, INPUT_SIZE], dtype = model.inputs[0].dtype))
    # NOTE: use inputs[0]

    # so inferences input need to have 3 dim.  
    ###################################################################
    
    model.save(tf_dir, save_format="tf", signatures=signature) # save_format is tf by default. pass a dir
    #model.save(tf_dir, save_format="tf")

    # Recreate the exact same model
    #new_model = tf.keras.models.load_model('path_to_saved_model')
    
    # WARNING. to restart training, use save_model (h5 or tf) , not save json + weigths
    # load json and weights . However, unlike model.save(), this will not include the training config and the optimizer. You would have to call compile() again before using the model for training.

    # h5 format. single file
    # The model's architecture, The model's weight values (which were learned during training) The model's training config (what you passed to compile), if any The optimizer and its state, if any (this enables you to restart training where you left off)
    # Save the entire model to a HDF5 file. The '.h5' extension indicates that the model shuold be saved to HDF5.
    print('\nPABOU: save full model in h5 format. file is : ', h5_file)
    model.save(h5_file, save_format="h5") 
    
    # Recreate the exact same model purely from the file
    # new_model = tf.keras.models.load_model('path_to_my_model.h5')

    # get file size
    h5_size = os.path.getsize(h5_file)
    print('PABOU: h5 file size Mo: ', round(float(h5_size/(1024*1024)),2))
    
    print('PABOU: save weights manualy after fit. use epoch = 999. callback also save checkpoints')
    # if extension is .h5 or .keras, save in Keras HDF5 format. else TF checkpoint format
    # or use, save_format = 'h5' or 'tf'
    model.save_weights(checkpoint_path.format(epoch=999)) # argument is a file path
    #load_weights(fpath)
    
    print('PABOU: save model architecture as json: ' , json_file)
    json_config = model.to_json()
    with open(json_file, 'w') as json_file:
        json_file.write(json_config)
    #reinitialized_model = keras.models.model_from_json(json_config)

    return(tf_dir, h5_file, h5_size)

#####################################
# h5 size in Meg
#####################################

def get_h5_size(app):
     h5_file = app +  full_model_h5_file # will be saved as .h5py
     h5_file = os.path.join('models' , h5_file)
     h5_size = round(float(os.path.getsize(h5_file) / (1024*1024)),1)
     return(h5_size)

    
######################################################
# load model Full ou Lite
# can load from h5, SavedModel or from empty model and checkpoint. can also load TFlite model
# return Keras model or TFlite interpreter 
######################################################
# if load from checkpoint, expect empty to be a empty model. otherwize not used
# app is the string which identify apps, ALREADY has cell1_c_o
# type is 'h5' or 'tf' (SavedModel) or 'cp', or 'li'. 

# empty and checkpoint parameters only used when loading from checkpoint
# cp load latest chckp in empty model which passed as argument

# full model are loaded from h5 file
# tflite models are loaded from SavedModel (see signatures)

def load_model(app, type, empty, checkpoint_path):

    # add suffix ie cello1_c_mo  cello1_c_mo _xxxxx
    tf_dir = app +  full_model_SavedModel_dir
    h5_file = app  +  full_model_h5_file 
    json_file = app  + model_json_file
    lite_file = app + lite_model_file

    # put under models
    tf_dir = os.path.join('models' , tf_dir)
    h5_file = os.path.join('models' , h5_file)
    json_file = os.path.join('models' , json_file)
    lite_file = os.path.join('models' , lite_file)

    if type == 'h5':
        try: 
            model = tf.keras.models.load_model(h5_file) 
            print ("PABOU loaded FULL h5 model from %s" %(h5_file))
            return(model)
        except Exception as e:
            print('PABOU: Exception %s when loading h5 model from %s.' %(str(e),h5_file)) 
            return(None)
            
    if type == 'tf':
        try:
            
            model = tf.keras.models.load_model(tf_dir) # dir
            print ("PABOU: loaded full tf model from %s" %(tf_dir))
            return(model)
        except Exception as e:
            print('PABOU: Exception %s when loading tf model from %s' %(str(e),tf_dir))
            return(None)
            
    if type == 'cp':    
        try:
                print('PABOU: got an empty model. load latest checkpoint')
                checkpoint_dir = os.path.dirname(checkpoint_path)
                latest = tf.train.latest_checkpoint(checkpoint_dir)
                print ('PABOU: load latest checkpoint. weights only:', latest)
                empty.load_weights(latest)
                return(empty)
        except Exception as e:
                print('!!!! cannot load checkpoint %s' %(str(e)))
                return(None)

    if type == 'li':
        try:
                print('PABOU: load TFlite model from %s: ' %(lite_file))
                # creater interpreter once for all. create from TFlite file
                interpreter = tf.lite.Interpreter(lite_file) 
                return(interpreter)       
        except Exception as e:
                print('!!!! cannot load TFlite file %s' %(str(e)))
                return(None)
                
    print('PABOU: !!!!! ERROR load model. unknown type')
    return(None)


############################################################
# add meta data to tflite model
# model 1 only for now
# ZIP file unzip mobilenet_v1_0.75_160_quantized_1_metadata_1.tflite
############################################################

#https://stackoverflow.com/questions/64097085/issue-in-creating-tflite-model-populated-with-metadata-for-object-detection
#https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/metadata/metadata_writer_for_image_classifier.py


def add_tflite_meta_data(app, tflite_file_name, model_type, hot_size_pi, description):

    # hot size is needed for metadata output range
    # app is needed to include label.txt
    # app_labels.txt is created in models directory. list of dictionaries. just referenced in metadata
    # tflite file name is the name of the output. same as input
    # description is custo string, will go in meta.description

    # insert label, but also additional files not required, requirements.txt

    if model_type == 2:
        print('not implemented for model %d' %model_type)
        return(None)


    # label contains 3 dictionary with unique pi, du, ve
    x = app + label_txt
    label_file_name = os.path.join('models', x) # label_file is an object
    print('PABOU: metadata (label dictionaries) file name: ', label_file_name)
    

    from tflite_support import flatbuffers
    from tflite_support import metadata as _metadata
    from tflite_support import metadata_schema_py_generated as _metadata_fb

    # Creates model info.
    #Description générale du modèle ainsi que des éléments tels que les termes de la licence
    
    model_meta = _metadata_fb.ModelMetadataT() # object
    model_meta.name = "Bach generator"
    model_meta.description = ("Real time streaming to any browser. LSTM with attention. Multiple corpus. GUI")
    model_meta.version = "v2"
    model_meta.author = "pabou. pboudalier@gmail.com. Meaudre Robotics"
    model_meta.license = ("GNU GPL 3")

    # Creates input info.
    #Description des entrées et du prétraitement requis, comme la normalisation
    # as many metadata as entries

    input_pitch_meta = _metadata_fb.TensorMetadataT()
    #input_pitch_meta.description = "sequence of one hot encoded representing pitches index"
    input_pitch_meta.description = description
    input_pitch_meta.name = "input tensor: pitch as one hot or interger"
    

    # how to specifify LSTM seqlen etc ..
    input_pitch_meta.content = _metadata_fb.ContentT()
    input_pitch_meta.content.contentPropertiesType = _metadata_fb.ContentProperties.FeatureProperties
    input_pitch_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()
    #input_pitch_meta.content.contentProperties = _metadata_fb. ????
    
    """
    image : input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()  
    # what about need to specific seq of one hot
    
    """
    
    input_stats = _metadata_fb.StatsT()
    input_stats.max = [1]
    input_stats.min = [0]
    input_pitch_meta.stats = input_stats

    # Creates output info.
    # as many as output heads
    #Description de la sortie et du post-traitement requis, comme le mappage aux étiquettes.

    # output generic
    output_pitch_meta = _metadata_fb.TensorMetadataT()

    output_pitch_meta.name = "softmax probability. model 1 or 3"
    output_pitch_meta.description = "Probabilities of next note, duration, velocity"

    # output content
    output_pitch_meta.content = _metadata_fb.ContentT()
    output_pitch_meta.content.contentPropertiesType = (_metadata_fb.ContentProperties.FeatureProperties)
    output_pitch_meta.content.contentProperties = (_metadata_fb.FeaturePropertiesT())

    # output stats
    output_stats = _metadata_fb.StatsT()
    output_stats.max = [1.0]
    output_stats.min = [0.0]
    output_pitch_meta.stats = output_stats

    # output labels
    label_file = _metadata_fb.AssociatedFileT()
    #expected str, bytes or os.PathLike object, not AssociatedFileT
    label_file.name = os.path.basename(label_file_name) 
    label_file.description = "list of dictionaries for pitch, duration and velocity."
    label_file.type = _metadata_fb.AssociatedFileType.TENSOR_VALUE_LABELS
    output_pitch_meta.associatedFiles = [label_file]

    # output range 
    output_pitch_meta.content.range = _metadata_fb.ValueRangeT()
    output_pitch_meta.content.range.min = hot_size_pi
    output_pitch_meta.content.range.max = hot_size_pi

    
    # Creates subgraph info.
    # combine les informations du modèle avec les informations d'entrée et de sortie:
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_pitch_meta]
    subgraph.outputTensorMetadata = [output_pitch_meta]
    model_meta.subgraphMetadata = [subgraph]

    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_buf = b.Output()

    # populate

    # Une fois les Flatbuffers de métadonnées créées, les métadonnées et le fichier d'étiquette 
    # sont écrits dans le fichier TFLite via la méthode populate
    """Populates metadata and label file to the model file."""
    populator = _metadata.MetadataPopulator.with_model_file(tflite_file_name)
    
    #The number of input tensors (3) should match the number of input tensor metadata (1). issue with model 2
    populator.load_metadata_buffer(metadata_buf)

    # warning. those file are not in metadata, but can still be inserted
    f1 = os.path.join(os.getcwd(), 'requirements','my_requirement_pip.txt')
    f2 = os.path.join(os.getcwd(), 'requirements','my_requirement_conda.txt')
    
    #print('PABOU: add file %s %s %s in metadata. labels and requirements' %(f1,f2,label_file_name) )

    #populator.load_associated_files([f1,f2, label_file_name])
    populator.load_associated_files([label_file_name])
    populator.populate()

    # create json file with metadata
    # normaly metadata are zipped in TFlite file
    displayer = _metadata.MetadataDisplayer.with_model_file(tflite_file_name) # can use a new file name ?
    json_content = displayer.get_metadata_json()

    # save metadata as json file in models directory
    metadata_json_file = os.path.join(os.getcwd(), 'models', app + '_metadata.json')
    with open(metadata_json_file, "w") as f:
        f.write(json_content)
    #print('PABOU: save metadata to json file %s' % metadata_json_file)
    #print('PABOU: metadata json content: ', json_content)

    print("PABOU: associated files in metadata: ", displayer.get_packed_associated_file_list())

    #expected str, bytes or os.PathLike object, not AssociatedFileT



def see_tflite_tensors(tflite_model):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    input_type = interpreter.get_input_details()[0]['dtype']
    print('TFlite input tensor: ', input_type)
    output_type = interpreter.get_output_details()[0]['dtype']
    print('TFlite output tensor: ', output_type)

#Netron is a viewer for neural network, deep learning and machine learning models.

#################################################################
# A generator that provides a representative dataset
# Unlike constant tensors such as weights and biases, variable tensors such as model input, activations (outputs of intermediate layers) and model output cannot be calibrated unless we run a few inference cycles
# To support multiple inputs, each representative data point is a list and elements in the list are fed to the model according to their indices.
#################################################################
gen = [] # need to be visible by generator in this module
def representative_dataset_gen():
    global gen # test set set in save_all_lite gen (x_test) shape: (3239, 100), dtype: int32. single sample (100,) int32
    global model_ttype # set in save_all_lite
    global input_dtype #  set in save_all_lite. np type
    
    print('PABOU: calling representative dataset generator: gen (x_test) shape: %s, dtype: %s. single sample %s %s' %(gen.shape, gen.dtype, gen[0].shape, gen[0].dtype))
    print('PABOU: input dtype ' , input_dtype) # model input. float32 for CONV2D
    for input_value in gen[:1000]: #(40,)
        
        # model 1 is one hot of float32

        if model_ttype == 1:
            input_value = np.expand_dims(input_value,axis=0) # add batch dimension ie  (1,....)
            input_value = input_value.astype(input_dtype) 

        if model_ttype in [2,3]:
            # input layer was left as default in model3 definition, ie float 32. 
            # because setting it to int32 cause TPU TFlite conversion to dies mysteriously and silently
            # x_test is from vectorization and default to int32

            input_value = np.expand_dims(input_value,axis=0) # add batch dimension ie  (1,....)
            input_value = np.expand_dims(input_value,axis=-1)

            # cast to input type as defined in model
            input_value = input_value.astype(input_dtype) 

        if model_ttype == 4:
            d = input_value.shape[0] # array of int32 (100,)
            d = int(math.sqrt(d))
            input_value = np.reshape(input_value, (d, d)) # (100,) to (10, 10) , still int32

            # at this point (6,6) conv2D expect 4 dims
            input_value = np.expand_dims(input_value,axis=0) # add batch
            input_value = np.expand_dims(input_value,axis=-1) # (1,10,10,1), still int32

            input_value = input_value.astype(input_dtype)  # float32 for CONV2D

        yield [input_value]

    # 

####################################################
# TF lite save ALL models 
# all quantization
# also creates model.cpp file for micro controler and edgetpu model, both from INT only quantization
# alignas(8) const unsigned char model_tflite[] = {
# note see also CLI tflite_convert --output_file --saved_model_dir --keras_model_file
# only on linux. on windows , use WSL to create CC model
####################################################
def save_all_tflite_models(app, x_gen, model_type, hot_size_pi, model):

    # hot size is for metadata
    global gen # to be visible in generator
    global model_ttype # to be visible in generator
    global input_dtype # to be visible in generator
    gen = x_gen # x_gen is test set
    model_ttype = model_type # from param

    # expected input type, as Numpy  type
    # needed by generator.  generator will cast to this 
    input_dtype = model.inputs[0].dtype.as_numpy_dtype # float32 for CONV2D

    #x_gen is x_train, or [x_train, x1_train] or [x_train, x1_train, x2_train]. use for representative data set
    # representative data set for quantization need to least 100
    meg = 1024*1024
    # hot size is for metadata

    print('PABOU: will use representative data set: ', x_gen.shape)
    # always use converter from save model (vs from keras instanciated model)

    # path to full model dir, to convert from to TFlite
    tf_dir = app +  full_model_SavedModel_dir # from save model
    tf_dir = os.path.join('models' , tf_dir)

    # path to h5 file , to convert from to TFlite
    h5_file = app +  full_model_h5_file # from h5 file
    h5_file = os.path.join('models' , h5_file)


    # file name for TFlite models.  all file are .tflite
    # tflite_* are full name in file system. just appended some prefix
    # tflite_*_file are just file name which do not exists in file system

    x = app +  tflite_fp32_file
    tflite_fp32 = os.path.join('models', x)

    x = app +  tflite_default_file
    tflite_default = os.path.join('models', x)

    x = app +  tflite_default_representative_file
    tflite_default_representative = os.path.join('models', x)

    x = app +  tflite_int_fp_fallback_file
    tflite_int_fp_fallback = os.path.join('models', x)

    x = app +  tflite_int_only_file
    tflite_int_only = os.path.join('models', x)

    x = app +  tflite_FP16_file
    tflite_fp16 = os.path.join('models', x)

    x = app +  tflite_16by8_file
    tflite_16by8 = os.path.join('models', x)
    
    #model = tf.keras.models.load_model(tf_dir) # dir
    #model = tf.keras.models.load_model(h5_file) # file
    #converter = tf.lite.TFLiteConverter.from_keras_model(model)


    # create converter
    # recommended . create converter from disk based savedModel
    # always use converter from save model (vs from keras instanciated model)
    print ('PABOU: convert to TFlite flat buffer from SavedModel: ' , tf_dir)

    # USES representative data set:  TPU, default variable, int fp fallback , experimental
    # do NOT use: fp32 (no quantization), default, fp16, 


    def tpu(model_type, hot_size_pi, description):
        ############################################################################ 
        # case 2 TPU: enforce INT int only and generate error for ops that cannot be quantized
        # 4x smaller, 3x speeded
        # CPU, TPU, micro
        ############################################################################ 

        """
        Additionally, to ensure compatibility with integer only devices (such as 8-bit microcontrollers) and accelerators 
        (such as the Coral Edge TPU), you can enforce full integer quantization for all ops including the input and output, 
        by using the following steps:
        To quantize the input and output tensors, and make the converter throw an error if it encounters an operation it cannot quantize, 
        convert the model again with some additional parameters:
        """
        print('PABOU: INT8 only for TPU. creates file: %s\n' % tflite_int_only)

        converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir) # creates new converter each time ?

        # This enables quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # This sets the representative dataset for quantization
        converter.representative_dataset = representative_dataset_gen

        # This ensures that if any ops can't be quantized, the converter throws an error
        
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
        converter.target_spec.supported_types = [tf.int8]
        # These set the input and output tensors to uint8 (added in r2.3)
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        #micro wants int8, not uint8

        try:
            tflite_model = converter.convert()
            open(tflite_int_only, "wb").write(tflite_model)

            print('\nPABOU:============================= OK INT8 only. CPU, TPU, MICRO. %s\n' % tflite_int_only)
            print('full model %0.1f, lite mode %0.1f' %( os.path.getsize(h5_file)/meg, os.path.getsize(tflite_int_only)/meg )) 
            see_tflite_tensors(tflite_model)

            # models for micro controler and edge TPU created after all tflite models
            # edgetpu compiler only run on linux 

            try:
                add_tflite_meta_data(app, tflite_int_only, model_type, hot_size_pi, description) # pass file name which exist in file system, not binary model
                # model 1 only for now
            except Exception as e:
                print('PABOU: cannot add metadata to %s. %s. maybe model 2' %(tflite_fp32, str(e)))

        except Exception as e:
            print('PABOU:===== Exception converting to tflite INT8 only ', str(e))
            #module 'tensorflow.lite.python.schema_py_generated' has no attribute 'Model'
            # Failed to parse the model: pybind11::init(): factory function returned nullptr.


    def fp32(model_type, hot_size_pi, description):
        ##############################################################
        # case 1: no quantization. fp32 value for all 
        ##############################################################
        # hot needed for metadata

        print('PABOU: TFlite convert: fp32 without quantization. creates file: %s\n' % tflite_fp32)

        converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)

        converter.inference_input_type = tf.float32 # is set to int8 in tpu
        converter.inference_output_type = tf.float32
        
        try:
            tflite_model = converter.convert() 
            # tflite model is a bytes b' \x00\x00\x00TFL3\x00\x00\x00\  <class 'bytes'>
            open(tflite_fp32, "wb").write(tflite_model) # complete name in file system with prefix
            print('\nPABOU:========================== OK: created %s. using 32-bit float values for all parameter data\n' %tflite_fp32)
            print('PABOU: full model %0.1f Meg, lite model %0.1f Meg ' %( os.path.getsize(h5_file)/meg, os.path.getsize(tflite_fp32)/meg )) 
            see_tflite_tensors(tflite_model)

            ##############################################
            # metadata. only implemented for model 1
            ##############################################
            print('PABOU:add metadata to TFlite file model 1')
            try:
                add_tflite_meta_data(app, tflite_fp32, model_type, hot_size_pi, description) # pass file name which exist in file system, not binary model
                # model 1 only for now
            except Exception as e:
                print('PABOU: cannot add metadata to %s. %s. maybe model 2' %(tflite_fp32, str(e)))

        except Exception as e:
            print('cannot convert fp32')
    
        #Vous pouvez utiliser Netron pour visualiser vos métadonnées,

    def default(model_type, hot_size_pi, description):
        #############################################################
        # case 3: Weights statically converted from fp to int8
        # inpout, output still fp 
        # 4x smaler, 2,3 x speed
        # for CPU
        # no need for representative data set
        #############################################################

        """
        The simplest form of post-training quantization statically quantizes only the weights from 
        floating point to integer, which has 8-bits of precision:

        At inference, weights are converted from 8-bits of precision to floating point and computed using floating-point kernels. 
        This conversion is done once and cached to reduce latency.

        To further improve latency, "dynamic-range" operators dynamically quantize activations based on their range to 8-bits 
        and perform computations with 8-bit weights and activations. This optimization provides latencies close to fully 
        fixed-point inference. However, the outputs are still stored using floating point so that the speedup
        with dynamic-range ops is less than a full fixed-point computation.
        """

        print('PABOU: Default, Weigths converted to int8. creates file: %s \n' % tflite_default)

        converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.inference_input_type = tf.float32 # was set to int8 in tpu
        converter.inference_output_type = tf.float32

        try:
            tflite_model = converter.convert()
            open(tflite_default, "wb").write(tflite_model) # tflite_model_size is a file name
            print('\nPABOU:============================ OK %s. quantized weights, but other variable data is still in float format.\n' %tflite_default)
            print('full model %0.1f, lite mode %0.1f ' %( os.path.getsize(h5_file)/meg, os.path.getsize(tflite_default)/meg )) 
            see_tflite_tensors(tflite_model)

            try:
                add_tflite_meta_data(app, tflite_default, model_type, hot_size_pi, description) # pass file name which exist in file system, not binary model
                # model 1 only for now
            except Exception as e:
                print('PABOU: cannot add metadata to %s. %s. maybe model 2' %(tflite_fp32, str(e)))

        except Exception as e:
            print('PABOU:==== Exception converting to tflite weigth int', str(e))
    

    def default_variable(model_type, hot_size_pi, description):
        #############################################################
        # case 3.1: dynamic range. fixed and variable params
        # 4x smaler, 2,3 x speed
        # for CPU 
        # USES representative data set
        #############################################################

        print('PABOU: fixed and variable converted to int8. creates file: %s \n' % tflite_default_representative)

        converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen

        try:
            tflite_model = converter.convert()
            open(tflite_default_representative, "wb").write(tflite_model) # tflite_model_size is a file name
            print('\nPABOU:=================== OK %s. fixed and variable quantized.\n' %tflite_default)
            print('full model %0.1f, lite mode %0.1f ' %( os.path.getsize(h5_file)/meg, os.path.getsize(tflite_default_representative)/meg )) 
            see_tflite_tensors(tflite_model)

            try:
                add_tflite_meta_data(app, tflite_default_representative, model_type, hot_size_pi, description) # pass file name which exist in file system, not binary model
                # model 1 only for now
            except Exception as e:
                print('PABOU: cannot add metadata to %s. %s. maybe model 2' %(tflite_fp32, str(e)))

        except Exception as e:
            print('PABOU:==== Exception converting to tflite weigth int', str(e))

            
    def int_fp_fallback(model_type, hot_size_pi, description):
        ##############################################################
        # case 4: full integer quantization . all math integer
        # measure dynamic range thru sample data
        # 4x smaller, 3x speeded
        # CPU,  not tpu TPU, micro as input/output still  
        # USES representative data set
        ##############################################################


        """
        Note: This tflite_quant_model won't be compatible with integer only devices (such as 8-bit microcontrollers) 
        and accelerators (such as the Coral Edge TPU) because the input and output still remain float in order to have the same interface as the original float only model.
        
        You can get further latency improvements, reductions in peak memory usage, and compatibility with integer 
        only hardware devices or accelerators by making sure all model math is integer quantized.

        For full integer quantization, you need to measure the dynamic range of activations and inputs by supplying 
        sample input data to the converter. Refer to the representative_dataset_gen() function used in the following code.
        
        Integer with float fallback (using default float input/output)
        In order to fully integer quantize a model, but use float operators when they don't have an integer implementation (to ensure conversion occurs smoothly),
        use the following steps:

        That's usually good for compatibility, but it won't be compatible with devices that perform only integer-based operations, 
        such as the Edge TPU.

        Additionally, the above process may leave an operation in float format if TensorFlow Lite doesn't include a 
        quantized implementation for that operation. This strategy allows conversion to complete so you have a smaller and more 
        efficient model, but again, it won't be compatible with integer-only hardware. 
        (All ops in this MNIST model have a quantized implementation.)

        Now all weights and variable data are quantized, and the model is significantly smaller compared to the original TensorFlow Lite model.
        However, to maintain compatibility with applications that traditionally use float model input and output tensors, the TensorFlow Lite Converter leaves the model input and output tensors in float:
        """

        """
        to rescale at inference time see post_training_integer_quant colab
        # Check if the input type is quantized, then rescale input data to uint8
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image / input_scale + input_zero_point

            test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
            interpreter.set_tensor(input_details["index"], test_image)

        """

        print('PABOU: full integer quantization, with fall back to fp. need representative data set. creates: %s \n' %tflite_int_fp_fallback)
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen

        #Cannot set tensor: Got value of type NOTYPE but expected type FLOAT32 for input 0, name: flatten_input 
        try:
            tflite_model = converter.convert()
            open(tflite_int_fp_fallback, "wb").write(tflite_model)
            print('\nPABOU:====================== OK %s. However, to maintain compatibility with applications that traditionally use float model input and output tensors, the TensorFlow Lite Converter leaves the model input and output tensors in float.\n' %tflite_int_fp_fallback)
            print('full model %0.1f, lite mode %0.1f ' %( os.path.getsize(h5_file)/meg, os.path.getsize(tflite_int_fp_fallback)/meg ))
            see_tflite_tensors(tflite_model)

            try:
                add_tflite_meta_data(app, tflite_int_fp_fallback, model_type, hot_size_pi, description) # pass file name which exist in file system, not binary model
                # model 1 only for now
            except Exception as e:
                print('PABOU: cannot add metadata to %s. %s. maybe model 2' %(tflite_fp32, str(e)))

        except Exception as e:
            print('PABOU:===== Exception converting to tflite int fall back fp', str(e))
        

    def fp16(model_type, hot_size_pi, description):

        #######################################################################
        # case 5: FP16 only
        # W to fp16 , vs fp32
        # CPU, GPU delegate
        # 2x smaller, GPU acceleration
        # GPU will perform on fp16, but CPU will dequantize to fp32
        #######################################################################


        """
        You can reduce the size of a floating point model by quantizing the weights to float16, the IEEE standard for 16-bit floating point numbers. To enable float16 quantization of weights, 
        use the following steps:
        """

        print('PABOU: quantization FP16. GPU acceleration delegate.  creates file: %s\n' % tflite_fp16)
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        try:
            

            tflite_model = converter.convert()
            open(tflite_fp16, "wb").write(tflite_model)
            print('\nPABOU:================================= OK FP16, GPU delegate %s.\n' %tflite_fp16)
            print('full model %0.1f, lite mode %0.1f ' %( os.path.getsize(h5_file)/meg, os.path.getsize(tflite_fp16)/meg )) 
            see_tflite_tensors(tflite_model) # still fp32

            try:
                add_tflite_meta_data(app, tflite_fp16, model_type, hot_size_pi, description) # pass file name which exist in file system, not binary model
                # model 1 only for now
            except Exception as e:
                print('PABOU: cannot add metadata to %s. %s. maybe model 2' %(tflite_fp32, str(e)))

        except Exception as e:
            print('PABOU:===== Exception converting to tflite FP16. ', str(e))
        

    def experimental(model_type, hot_size_pi, description):

        ####################################################################
        # case 6: 16 bit activation, 8 bit weight
        # experimental
        # improve accuracy
        # small size reduction
        ####################################################################

        """
        This is an experimental quantization scheme. It is similar to the "integer only" scheme, but activations are 
        quantized based on their range to 16-bits, weights are quantized in 8-bit integer and bias is quantized into 64-bit integer. 
        This is referred to as 16x8 quantization further.

        The main advantage of this quantization is that it can improve accuracy significantly, but only slightly increase model size
        Currently it is incompatible with the existing hardware accelerated TFLite delegates.
        """

        print('PABOU: quantization 16x8.  creates file: %s\n ' % tflite_16by8)
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)

        converter.representative_dataset = representative_dataset_gen
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]

        """
        If 16x8 quantization is not supported for some operators in the model, then the model still can be quantized, 
        but unsupported operators kept in float. The following option should be added to the target_spec to allow this.
        """
        #converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
        
        #add tf.lite.Opset.TFLITE_BUILTINS to keep unsupported ops in float

        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        #Exception converting to tflite 16x8 The inference_input_type and inference_output_type must be tf.float32.

        try:
            # The inference_input_type and inference_output_type must be in ['tf.float32', 'tf.int16'].
            
            tflite_model = converter.convert()
            open(tflite_16by8, "wb").write(tflite_model)
            print('\nPABOU:===== OK int only with activation 16 bits %s \n' %tflite_16by8)
            print('full model %0.1f, lite mode %0.1f ' %( os.path.getsize(h5_file)/meg, os.path.getsize(tflite_16by8)/meg )) 
            see_tflite_tensors(tflite_model)

            try:
                add_tflite_meta_data(app, tflite_16by8, model_type, hot_size_pi, description) # pass file name which exist in file system, not binary model
                # model 1 only for now
            except Exception as e:
                print('PABOU: cannot add metadata to %s. %s. maybe model 2' %(tflite_fp32, str(e)))

        except Exception as e:
            print('PABOU:===== Exception converting to tflite 16x8', str(e))


    ######################################
    # run all TFlite conversions
    # metadata is only for model 1
    ##################################### 
    
    # All except fp32 and GPU are small models  

    # below uses representative data generator
    tpu(model_type,hot_size_pi, 'INT8 only. For TPU, CPU, Micro. input/output are int8.') 
    default_variable(model_type,hot_size_pi, 'default with representative data set. Fixed and variable are int8. input/output are uint8.')
    int_fp_fallback(model_type,hot_size_pi, 'Integer with floating point fall back. input/output are FLOAT.') # However, to maintain compatibility with applications that traditionally use float model input and output tensors, the TensorFlow Lite Converter leaves the model input and output tensors in float.
    experimental(model_type,hot_size_pi, 'experimental TFlite model. input/output are fp32.') #Quantization to 16x8-bit not yet supported for op: 'UNIDIRECTIONAL_SEQUENCE_LSTM'.
    
    # do not use the generator
    fp32(model_type,hot_size_pi, 'fp32 TFlite model. input/output are fp32.')
    default(model_type,hot_size_pi, 'default TFlite model. Weigths encoded to int8.')
    fp16(model_type,hot_size_pi, 'fp16 TFlite model. inpout/output are fp32.')

    print('\nPABOU: All tfLite models created as .tflite files')

    ######################################
    # creates model.cpp file for micro
    """
    Arduino expects the following:
    #include "model.h"
    alignas(8) const unsigned char model_tflite[] = {
    const unsigned int model_tflite_len = 92548;
    """

    # select tflite model quantization to convert to C 
    # fp32 and GPU are big. other are smaller
    # in WSL models  xxd -i ***.tflite > model.cpp
    # edit model.cpp to include above, both array definition and len
    # copy resulting model.cpp into arduino folder, as model.cpp (so that it get compiled and linked)
    # no need to touch model.h in arduino folder
    ######################################

    out = os.path.join('models', app + micro_model_file)
    in_ = os.path.join('models', app + tflite_int_only_file)
    print ('PABOU: convert INT8 only tflite file to micro-controler. C file: %s, from %s' %(out, in_))
    try:
        s = 'xxd -i ' + in_  + ' > ' + out
        os.system(s) # exception stays in shell
    except:
        print('PABOU: cannot create model.cpp. xxd not found. please run on linux or WSL')

    ######################################
    # creates EDGE TPU model
    # compiler only run on linux
    # https://colab.research.google.com/github/google-coral/tutorials/blob/master/compile_for_edgetpu.ipynb
    # The filename for each compiled model is input_filename_edgetpu.tflite
    ######################################

    in_ = os.path.join('models', app + tflite_int_only_file)
    try:
        s = 'edgetpu_compiler -s -o ' + os.path.join('models' , 'edge')  +  ' ' + in_
        print('PABOU: create edgetpu model ', s)
        os.system(s) # if not found, will not raise exception here. 
    except:
        print('PABOU: cannot create edgetpu model. please run on Linux or WSL')


#####################################################
# benchmark for full (non tflite) model
# for slice and iteration
# display with pretty table
# use test set as input
# look at ELAPSE and ACCURACY (vs simple inference)
######################################################
# param: string, model object, test set x and y, number of iteration, h5 file size

def bench_full(st, model, x_test, y_test, x1_test, y1_test, x2_test, y2_test, nb, h5_size, model_type):
    
    # use test set; x_test.shape (3245, 40, 483), y_test.shape (3245, 483)
    #1 and 2 are None for model 1
    #2 is None for model 2 NV

    # x_test[0] is seqlen array of int (embedding)
    # pass model type and use x2_test = [] to avoid importing config_bach. application agnostic
    
    error_pi=0
    
    print("\nPABOU: running FULL benchmark for: %s. iterative" %(st))

    # create table
    pt = PrettyTable()
    pt.field_names = ["model", "iters", "infer (ms)",  "acc pi", "acc du", "acc ve",  "h5 size", 'TFlite size']
    
    ##############################################
    # full model, iterate on single sample, ie one at a time like TFlite invoque
    # iterate, return one softmax
    ##############################################

    start_time = time.time() 
    for i in range(nb):  # use for loop as for lite (one inference at a time)

        if model_type == 2:
            if x2_test != []:  # velocity
            # test on [] vs None
                input = [x_test[i], x1_test[i], x2_test[i]] # x list of 3 list of seqlen int
                y = [y_test[i], y1_test[i], y2_test[i]]
            else:
                input = [x_test[i], x1_test[i]]
                y = [y_test[i], y1_test[i]]

        if model_type in [1,3]:  
            y =y_test[i]  # one hot, len 483 (483,)
            input = x_test[i]
            input = np.expand_dims(input,axis=0)

        if model_type == 4:
            # need to reshape size*size sequence into size x size matrix. assumes squared matrix 
            input = x_test[i]
            d = input.shape[0]
            d = int(math.sqrt(d))
            input = np.reshape(input, (d, d))
            input = np.expand_dims(input,axis=0)
            input = np.expand_dims(input,axis=-1) # (1,6,6,1) 
        

        result = model.predict(input) # Keras call. 

        # for model 2, result[[]]  is a list of multiple softmax.  result[0] [0] (pitches), result[1] [0] , result[2] [0]
        # nothing identify which param the softmax refers to, assume same order from model definition
        # np.sum(result[0] [0]) = 1
        #results = np.squeeze(output_data) is the same as [0]
        # model 1 result is [[]] result[0] is a simple softmax for pitches.  result[1] does not exist 
    
        elapse = (time.time() - start_time)*1000.0 # time.time() nb of sec since 1070. *1000 to get ms

        if model_type in [1,3,4]:
            softmax = result[0] # squeeze

        if model_type == 2:
            softmax = result[0] [0] # 1st one. and squeeze. assume pitches. # len 129, pitches

            # test acuracy; only look at error for pitches, as we get result[0]
            
        # top_k = results.argsort()[-5:][::-1]

        
        # for MNIST y  y_test[i] are integers, not softmax
        # for bach, softmax  483 fp32 type ndarray , NOT LIST

        #if type(y_test[i]) is list:  # test if Y is softmax or scalar uint8 for MNIST

        if isinstance(y_test[i],np.ndarray):
            if (np.argmax(softmax) != np.argmax(y_test[i])) : # test againts y_test which is ground thruth for pitches
                error_pi = error_pi + 1
        else:
            if np.argmax(softmax) != y_test[i] : # test againts y_test which is ground thruth for pitches
                #ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                error_pi = error_pi + 1

        # for MNIST y are integers 

    # for inferences

    a= float(nb-error_pi) / float(nb)
    elapse_per_iter = elapse/float(nb)
        
    print("PABOU: TF full model with loop: elapse %0.2f ms, %d samples. %d pi error, %0.2f pi accuracy" % (elapse_per_iter, nb, error_pi, a))
    pt.add_row(["full model", nb, round(elapse_per_iter,2), round(a,2), None, None, h5_size, None ]) 
    

    ##############################################
    # full model slice
    # batch dimension added automatically 
    ##############################################

    print("\nPABOU: running FULL benchmark for: %s. slice" %(st))
    error_pi = 0
    start_time = time.time()

    if model_type in [1,3]:
        result=model.predict(x_test[:nb]) # list of softmax
        # result.shape (100, 483)   list of 100 softmax .  for MNIST (100,10)

    if model_type in [4]:
        slice = x_test[:nb]
        d = x_test[0].shape[0]
        d = int(math.sqrt(d))
        slice = np.reshape(slice, (slice.shape[0],d, d))
        input = np.expand_dims(input,axis=-1) # (100,6,6,1) 
        result=model.predict(slice) # list of softmax


    if model_type == 2 and x2_test != []:
        result=model.predict([x_test[:nb], x1_test[:nb], x2_test[:nb]])
        # len result 3   
        # 3 elements. each 100 (nb of prediction) , each softmax of len corresponding to the output
        # result [0] is for pitches . len result[0] 100 softmax.  len result[0] [0] 129
        # len result [2] [2] 5


    if model_type == 2 and x2_test == []:
        result=model.predict([x_test[:nb], x1_test[:nb]])

    elapse = (time.time() - start_time)*1000.0

    # average time
    elapse_per_iter = elapse/float(nb)


    # average accuracy for pitches
    
    for i in range(nb):

        if isinstance(y_test[i],np.ndarray):

            if model_type == 2:
                softmax = result[0] [i]
            if model_type in [1,3,4]: 
                softmax = result[i]

            if np.argmax(softmax) != np.argmax(y_test[i]): # y[i] is a one hot same as we start x_test[:nb]
                error_pi = error_pi + 1

        else:
            softmax = result[i]
            if np.argmax(softmax) != y_test[i]: # MNIST case  y_test is not a softmax but un int
                error_pi = error_pi + 1


    a= float(nb - error_pi) / float(nb)
    
    print("PABOU: TF full model with slice: average %0.2f ms, %d samples. %d pi error, %0.2f pi accuracy" % (elapse_per_iter, nb, error_pi, a))
  
    # add to pretty table
    pt.add_row(["full model: slice", nb, round(elapse_per_iter,2), round(a,2), None, None, h5_size, None ]) 
    
    # print pretty table
    print('\n',pt,'\n')
    

# cannot import at the same time tensorflow and coral run time generic
# generic module which can work with tf or coral
# someone must import either tf or coral runtime before
# contain single inference and benchmark
# benchmark creates interpreter
# when imported from pabou, cannot be used for coral benchmark
# to use for coral benchmark, call from main before importing tensorflow. need X and Y
print('PABOU: import lite_inference')
import lite_inference
# contains inference and bench lite one model


###################################################
# MODEL evaluate 
# print pretty table
# to be used for model just fitted , or model loaded 
####################################################
def model_evaluate(model, model_type, x , y):
    print("\n\nPABOU: evaluate model on test set") 
    # x is x_test (3239,100)

    if model_type in [1,3]:
        # x single vs array 
        score = model.evaluate(x, y, verbose = 0 )  # evaluate verbose  1 a lot of =
        # do not use dict, so get a list

        print('PABOU: evaluate score list: ',score) # [4.263920307159424, 0.11587057262659073]
        print('PABOU: metrics: ', model.metrics_names , type(model.metrics_names))

        test_pitch_accuracy = round(score [1],2)
        print('PABOU: test pitch accuracy %0.2f' %(test_pitch_accuracy))

        test_duration_accuracy = 0.0 # procedure returns then 3 metrics even in model 1
        test_velocity_accuracy = 0.0

        """
        # accuracy from any dataset
        print('model accuracy from any dataset')
        pabou.see_model_accuracy(model, x_train, y_train)
        """

    if model_type == 4: # (3239,100)

        # already reshaped
        #d = x[0].shape[0]
        #d = int(math.sqrt(d))
        #x = np.reshape(x, (x.shape[0], d, d))

        score = model.evaluate(x, y, verbose = 0 )  # evaluate verbose  1 a lot of =

        print('PABOU: evaluate score list: ',score) # [4.263920307159424, 0.11587057262659073]
        print('PABOU: metrics: ', model.metrics_names , type(model.metrics_names))

        test_pitch_accuracy = round(score [1],2)
        print('PABOU: test pitch accuracy %0.2f' %(test_pitch_accuracy))

        test_duration_accuracy = 0.0 # procedure returns then 3 metrics even in model 1
        test_velocity_accuracy = 0.0


    if model_type == 2:
        # use dict
        score = model.evaluate(x, y, verbose = 0, batch_size=128, return_dict=True )

        #If dict True, loss and metric results are returned as a dict, with each key being the name of the metric. If False, they are returned as a list.
        #score : {'duration_output_accuracy': 0.7097072601318359, 'duration_output_loss': 0.9398044943809509, 'loss': 7.060447692871094, 'pitch_output_accuracy': 0.13097073137760162, 'pitch_output_loss': 2.934774398803711, 'velocity_output_accuracy': 0.9143297672271729, 'velocity_output_loss': 0.25109609961509705}
        
        # if false
        #score:  [10.401823997497559, 4.265839099884033, 1.4490309953689575, 0.421114444732666, 0.1109057292342186, 0.5055452585220337, 0.8545902371406555]
        
        #model metrics:  ['loss', 'pitch_output_loss', 'duration_output_loss', 'velocity_output_loss', 'pitch_output_accuracy', 'duration_output_accuracy', 'velocity_output_accuracy']
        
        print('PABOU: model metrics: ', model.metrics_names) # list

        test_duration_accuracy = round(score['duration_output_accuracy'],2)
        test_pitch_accuracy = round(score['pitch_output_accuracy'],2)

        if 'velocity_output_accuracy' in score:
            test_velocity_accuracy = round(score['velocity_output_accuracy'],2)
        else:
            test_velocity_accuracy = 0.0
            
        
        # test are scalar (float) . history contains lists, one entry per epoch
        
        print('PABOU: test set accuracy: pitch %0.2f, duration %0.2f velocity %0.2f' %(test_pitch_accuracy, test_duration_accuracy, test_velocity_accuracy))


    # create table
    pt = PrettyTable()
    pt.field_names = ["accuracy pitch", "accuracy duration", "accuracy velocity"]
    pt.add_row([test_pitch_accuracy, test_duration_accuracy, test_velocity_accuracy])
    print(pt) 
    return (test_pitch_accuracy, test_duration_accuracy, test_velocity_accuracy)

#########################################
# temperature
# ALTERNATIVE method to selecting argmax from model output
# reshape proba, and sample at random from new proba 

# alter prediction softmax, return new softmax = sampling, ie 1 with other zero, from which we get argmax.
# temp = 1 new softmax = input softmax and sampling is based on probability output by the model
# temp = 0.2 even more probability to sample original argmax
# temp = 2 increase probability of selecting other than argmax
#########################################
def get_temperature_pred(input_softmax, temp = 1.0):
    
    input_softmax=np.asarray(input_softmax).astype('float64')
    # temperature magic
    # log will create array of negative numbers as all element of softmax are less than 1
    preds=np.log(input_softmax) / float(temp) # np.log base e  np.log10 base 10 np.log2 base 2

    # make it a softmax again
    e_preds=np.exp(preds)
    new_softmax= e_preds / np.sum(e_preds) 

    proba = np.random.multinomial(1,new_softmax,1)
    """
    draw at random based on new_softmax probalility
    so do not take straigth the argmax 

    1 number of experiment, ie one run
    probability of the p different outcome
    number of runs. if =1 return is [[x,x,x,x]] , else [[], [] ]

    return drawn sample


    Throw a dice 20 times:
    np.random.multinomial(20, [1/6.]*6, size=1)
    array([[4, 1, 7, 5, 2, 1]]) # random
    It landed 4 times on 1, once on 2, etc.

    Now, throw the dice 20 times, and 20 times again:
    np.random.multinomial(20, [1/6.]*6, size=2)
    array([[3, 4, 3, 3, 4, 3], # random
       [2, 4, 3, 4, 0, 7]])

    """
    return(proba)


##############################################
# create corpus and dictionaries as .cpp file for microcontroler
##############################################
def create_corpus_cc(app, a):
    # input is list of int 
    # creates a C array declaration file, which can be included in a Cpp environment

    in_ = os.path.join('models', corpus_cc)
    print ('PABOU: create corpus.cc: %s. len %d' %(in_, len(a)))

    with open(in_, 'wt') as fp: # open as text (default)
        #fp.write('#include "corpus.h"\n')
        fp.write ('const int corpus[] = {\n ')

        i = a[0]
        s = '0x%x' %i
        fp.write(s)

        for i in a[1:]:
            s = ', 0x%x' %i
            fp.write(s)
            #b = (i).to_bytes(nbytes,byteorder='big')
            #fp.write(b) 
            # a bytes-like object is required, not 'int'
            # bytes(i) return i NULL bytes

        fp.write('\n};\n')

        #unsigned int model_tflite_len = 92548;
        fp.write ('const unsigned int corpus_len = ' + str(len(a)) + ';\n')



def create_dict_cc(app, a):
    # input is list of strings 

    in_ = os.path.join('models', dict_cc)
    print ('PABOU: create dict.cc: %s. len %d' %(in_, len(a)))

    with open(in_, 'wt') as fp: # open as text (default)
        #fp.write('#include "dictionary.h"\n')
        fp.write ('char *dictionary[] = {\n ') # const char* generate arduino compilation error

        i = a[0] # a is list of strings
        s = '"%s"' %i
        fp.write(s)

        for i in a[1:]:
            s = ', "%s"' %i
            fp.write(s)
        fp.write('\n};\n')
        fp.write ('const unsigned int dictionary_len = ' + str(len(a)) + ';\n')



"""
    # positional (mandatory) 
    parser.add_argument("test1", help="mandatory test1")
    parser.add_argument("test2", help="mandatory test2", type=int) # default string
    
    # optionel
    parser.add_argument("-v1", "--verb1", help="optional. need to execute with -v <value>")
    
    parser.add_argument("-v2", "--verb2", help="optional. can just use -v", action="store_true", default = 0)
    cannot use -l=1 
      
    parser.add_argument("-v3", "--verb3", help="optional", type=int, choices=[0, 1, 2], default=0)
    
    args=parser.parse_args() 
    args=vars(parser.parse_args()) # vars return dict attributes
    
    #usage: lstm.py [-h] [-v1 VERB1] [-v2] [-v3 {0,1,2}] test1 test2
    
    print (args.test1, type(args.test1))
    print (args.test1, type(args.test1))
    print (args.verb1, type(args.verb1))
    print (args.verb2, type(args.verb2))
    print (args.verb3, type(args.verb3))    
"""

"""
# if we want to get the top n in softmax
# argsort returns indices from smallest
    
ipdb> np.argsort(results)
array([6, 4, 1, 0, 8, 5, 9, 2, 3, 7])
ipdb> np.argsort(results) [-2]
3

ipdb> np.argsort(results) [:-2]
array([6, 4, 1, 0, 8, 5, 9, 2])

ipdb> np.argsort(results) [-2:]
array([3, 7])

# magic to do reverse sorting.  sort does not allow this
ipdb> np.argsort(results) [-2:] [::-1]
array([7, 3])


  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(args.label_file)
  for i in top_k:
    if floating_model:
      print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
    else:
      print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
"""

