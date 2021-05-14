#!/usr/bin/python3
# -*- coding: utf-8 -*-

# 30 jan 2021

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

import tensorflow as tf
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
#%matplotlib inline  # interactive in jupyter and colab
import os,sys
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

today = date.today()
title = today.strftime("%d/%m/%Y")

#################################################
# file names definition
#################################################

# base file names. will be prefixed by app so no ./   ie bach_./ will not work 
# save, load models use models/app_<file name>

full_model_SavedModel_dir = "full_model_SavedModel_dir"
full_model_h5_file = 'full_model_h5.h5' # will be saved as .h5py
model_json_file = 'full_model_json.json'
# png_file(training result) is passed as argument from main

# TFlite base file name, ie before prefix. will be prefixed by models/app_
# tflite_*_file are alias to real file names

tflite_fp32_file = 'fp32-only.tflite' # no quant
tflite_default_file = 'weigths-quantized.tflite' # default quant variable param still float
tflite_default_representative_file = 'weigths-and-layers-quantized.tflite' # variable param quantized
tflite_int_fp_fallback_file = 'quantized-with-floating-point-fallback.tflite'
tflite_int_only_file = 'quantized-for-TPU.tflite' # TPU, micro
tflite_FP16_file = 'fp16-for-GPU.tflite' # GPU delegate
tflite_16by8_file = '16-activation-8-w.tflite' # similar to int only, 


######################################################
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
    parser.add_argument("-p", "--predict", help="optional. -p to predict with FULL model, benchmark and create midi file. default FALSE", action="store_true", default =0)
    parser.add_argument("-bf", "--benchmark_full", help="optional. -b to benchmark full model. default FALSE", action="store_true", default =0)
    parser.add_argument("-bl", "--benchmark_lite", help="optional. -b to benchmark all lite model. default FALSE", action="store_true", default =0)
    parser.add_argument("-sl", "--savelite", action="store_true", default =0,   help="optional. default FALSE. use -sl to convert SaveModel to TFlite models")
    parser.add_argument("-st", "--stream", action="store_true", default =0,   help="optional. default FALSE. use -st to initiate real time stream")
    # use of TFlite for prediction is thru GUI

    # not really used
    parser.add_argument("-e", "--epochs", type=int, help="optional. use -e=5 to overwrite number of epochs in config file. DEFAULT=10. using debug mode will set to small number anyway")
    parser.add_argument("-d", "--debug", action="store_true", default =0,   help="optional. default FALSE. use -d to run in debug mode, ie small corpus (delete any existing pick) and hardcoded few epochs. debug is an OR of this argument and config file")
    parser.add_argument("-i", "--inference", type=int , default=30,  help="optional. number of inferences. typically for benchmarks")
    parser.add_argument("-s", "--ssh", action="store_true", default =0,   help="optional. default FALSE. use -s to disable plotting , when in ssh connection")

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
     
    if model_type == 1:
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

    if model_type == 1:   

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
        
    print('PABOU: plot training history')
    #plt.show(block=False)
    plt.show()

######################################################
# print various info. return os and platform
######################################################
def print_tf_info():

    print('\n')
    print ("PABOU: ========= > tensorflow version: < ============ ",tf.__version__)
    

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
######################################################
# generic save FULL model in various forms. save in Saved Model (dir), hf5(file) and json
# app is a string, identifies the app, make file names unique, even if not in same dir
# checkpoint path is handled in main,path are /ddd or \nnnn 
def save_full_model(app, model, checkpoint_path): # app is a string
    # return size of h5 file

    # suffix with _ done here
    # app is set to cello 1|2 nc nv mo|so in config_bach.py
    
    tf_dir = app + '_' +  full_model_SavedModel_dir
    h5_file = app + '_' +  full_model_h5_file # will be saved as .h5py
    json_file = app + '_' + model_json_file

    tf_dir = os.path.join('models' , tf_dir)
    h5_file = os.path.join('models' , h5_file)
    json_file = os.path.join('models' , json_file)

    # MODEL FULL save. ONLY AT THE END of FIT
    # export a whole model to the TensorFlow SavedModel format. SavedModel is a standalone serialization format for TensorFlow objects, supported by TensorFlow serving as well as TensorFlow implementations other than Python.
    # Note that the optimizer state is preserved as well: you can resume training where you left off.
    #The SavedModel files that were created contain: A TensorFlow checkpoint containing the model weights. A SavedModel proto containing the underlying TensorFlow graph.
    
    print('\nPABOU: save full model as SavedModel. directory is: ' , tf_dir)
    model.save(tf_dir, save_format="tf")

    """
    #################################################################
    # Keras LSTM fusion Codelab.ipynb
    #https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/experimental_new_converter/Keras_LSTM_fusion_Codelab.ipynb#scrollTo=tB1NZBUHDogR
    run_model = tf.function(lambda x: model(x))
    # This is important, let's fix the input size.
    BATCH_SIZE = None
    STEPS = 40
    INPUT_SIZE = 483
    signature = run_model.get_concrete_function(
     tf.TensorSpec(shape = [BATCH_SIZE, STEPS, INPUT_SIZE], dtype = model.inputs[0].dtype))
    model.save(tf_dir, save_format="tf", signatures=signature) # save_format is tf by default. pass a dir
    ###################################################################
    """

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
     h5_file = app + '_' +  full_model_h5_file # will be saved as .h5py
     h5_file = os.path.join('models' , h5_file)
     h5_size = round(float(os.path.getsize(h5_file) / (1024*1024)),1)
     return(h5_size)
    
######################################################
# load FULL model
# from type, can load from h5, SavedModel or from empty model and checkpoint 
######################################################
# if load from checkpoint, expect empty to be a empty model. otherwize not used
# app is the string which identify apps, type is 'h5' or 'tf' (SavedModel) or 'cp'. cp load latest chckp in empty model which is created
def load_model(app, type, empty, checkpoint_path):

    # prefix with cello_
    tf_dir = app + '_' +  full_model_SavedModel_dir
    h5_file = app + '_' +  full_model_h5_file # will be saved as .h5py
    json_file = app + '_' + model_json_file

    # put under models
    tf_dir = os.path.join('models' , tf_dir)
    h5_file = os.path.join('models' , h5_file)
    json_file = os.path.join('models' , json_file)

    if type == 'h5':
        try: 
            print('load FULL model as h5 from: %s' %(h5_file))
            model = tf.keras.models.load_model(h5_file) 
            print ("loaded FULL h5 model from %s" %(h5_file))
            return(model)
        except Exception as e:
            print('Exception %s when loading h5 model from %s.' %(str(e),h5_file)) 
            return(None)
            
    if type == 'tf':
        try:
            print('load FULL model as tf %s' %(tf_dir))
            model = tf.keras.models.load_model(tf_dir) # dir
            print ("loaded full tf model from %s" %(tf_dir))
            return(model)
        except Exception as e:
            print('Exception %s when loading tf model from %s' %(str(e),tf_dir))
            return(None)
            
    if type == 'cp':    
        try:
                print('got an empty model. load latest checkpoint')
                checkpoint_dir = os.path.dirname(checkpoint_path)
                latest = tf.train.latest_checkpoint(checkpoint_dir)
                print ('load latest checkpoint. weights only:', latest)
                empty.load_weights(latest)
                return(empty)
        except Exception as e:
                print('!!!! cannot load checkpoint %s' %(str(e)))
                return(None)
                
    print('unknown type')


############################################################
# add meta data to tflite model
# model 1 only for now
# ZIP file unzip mobilenet_v1_0.75_160_quantized_1_metadata_1.tflite
#https://stackoverflow.com/questions/64097085/issue-in-creating-tflite-model-populated-with-metadata-for-object-detection
############################################################

#https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/metadata/metadata_writer_for_image_classifier.py


def add_tflite_meta_data(app, tflite_file_name, model_type, hot_size_pi, description):

    # hot size is needed for metadata output range
    # app is needed to include label.txt
    # app_labels.txt is created in models directory
    # tflite file name is the name of the output. same as input

    # description is custo string, will go in meta.description

    if model_type != 1:
        print('not implemented for model %d' %model_type)
        return(None)

    x = app + '_' +  'labels.txt'
    label_file_name = os.path.join('models', x) # label_file is an object
    print('PABOU: metadata label file name: ', label_file_name)
    

    from tflite_support import flatbuffers
    from tflite_support import metadata as _metadata
    from tflite_support import metadata_schema_py_generated as _metadata_fb

    # Creates model info.
    #Description générale du modèle ainsi que des éléments tels que les termes de la licence
    
    model_meta = _metadata_fb.ModelMetadataT() # object
    model_meta.name = "Bach generator"
    model_meta.description = ("Real time streaming to any browser. LSTM with attention. Multiple corpus. GUI")
    model_meta.version = "v2"
    model_meta.author = "pabou. Meaudre Robotics"
    model_meta.license = ("GNU GPL 3")

    # Creates input info.
    #Description des entrées et du prétraitement requis, comme la normalisation
    # as many metadata as entries

    input_pitch_meta = _metadata_fb.TensorMetadataT()
    #input_pitch_meta.description = "sequence of one hot encoded representing pitches index"
    input_pitch_meta.description = description
    input_pitch_meta.name = "pitch input for Bach LSTM"
    

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

    output_pitch_meta.name = "softmax probability"
    output_pitch_meta.description = "Probabilities of next note,duration"

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
    
    print('PABOU: add file %s %s %s in metadata. labels and requirements' %(f1,f2,label_file_name) )

    populator.load_associated_files([f1,f2, label_file_name])
    populator.populate()

    # create json file with metadata
    # normaly metadata are zipped in TFlite file
    displayer = _metadata.MetadataDisplayer.with_model_file(tflite_file_name) # can use a new file name ?
    json_content = displayer.get_metadata_json()

    # save metadata as json file in models directory
    metadata_json_file = os.path.join(os.getcwd(), 'models', app + '_metadata.json')
    with open(metadata_json_file, "w") as f:
        f.write(json_content)
    print('PABOU: save metadata to json file %s' % metadata_json_file)
    print('PABOU: metadata json content: ', json_content)

    print("PABOU: associated files in metadata: ", displayer.get_packed_associated_file_list())

    #expected str, bytes or os.PathLike object, not AssociatedFileT



# display type of TFlite input and output tensors
def see_tflite_tensors(tflite_model):

    print('PABOU: show TFlite tensors type')

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    input_type = interpreter.get_input_details()[0]['dtype']
    print('input tensor: ', input_type)
    output_type = interpreter.get_output_details()[0]['dtype']
    print('output tensor: ', output_type)


#Netron is a viewer for neural network, deep learning and machine learning models.

####################################################
# TF lite save ALL model
# note see also CLI tflite_convert --output_file --saved_model_dir --keras_model_file

# tflite_*_file are alias to real file names
# tflite_fp32_file = 'fp32_only.tflite' # no quant

#x = app + '_' +  tflite_fp32_file
#tflite_fp32 = os.path.join('models', x)
#tflite_* are real file name in file system with path to models
####################################################



#################################################################
# A generator that provides a representative dataset
# To quantize the variable data (such as model input/output and intermediates between layers), 
# you need to provide a RepresentativeDataset.
# To support multiple inputs, each representative data point is a list and elements in the 
# list are fed to the model according to their indices.
#################################################################

gen = [] # need to be visible by generator in this module

def representative_dataset_gen():
    global gen
    
    print('PABOU: calling representative dataset generator ' , gen.shape)
    for input_value in tf.data.Dataset.from_tensor_slices(gen).batch(1).take(100):
        #print(input_value) 
        # Model has only one input so each data point has one element.
        # print(input_value.shape, type(input_value))
        #yield [input_value.astype(tf.float32)] tensorflow.python.framework.ops.EagerTensor' object has no attribute 'astype'
        yield [input_value]


def save_all_tflite_models(app, x_gen, model_type, hot_size_pi):
    global gen # to be visible in generator
    gen = x_gen # x_gen is training set

    #x_gen is x_train, or [x_train, x1_train] or [x_train, x1_train, x2_train]. use for representative data set
    # representative data set for quantization need to least 100

    meg = 1024*1024

    # hot size is for metadata

    print('PABOU: representative data set shape', x_gen.shape)
    
    # always use converter from save model (vs from keras instanciated model)

    # path to full model dir, to convert from to TFlite
    tf_dir = app + '_' +  full_model_SavedModel_dir # from save model
    tf_dir = os.path.join('models' , tf_dir)

    # path to h5 file , to convert from to TFlite
    h5_file = app + '_' +  full_model_h5_file # from h5 file
    h5_file = os.path.join('models' , h5_file)


    # file name for TFlite quantization 
    # tflite_* are full name in file system. just appended some prefix
    # tflite_*_file are just file name which do not exists in file system

    x = app + '_' +  tflite_fp32_file
    tflite_fp32 = os.path.join('models', x)

    x = app + '_' +  tflite_default_file
    tflite_default = os.path.join('models', x)

    x = app + '_' +  tflite_default_representative_file
    tflite_default_representative = os.path.join('models', x)

    x = app + '_' +  tflite_int_fp_fallback_file
    tflite_int_fp_fallback = os.path.join('models', x)

    x = app + '_' +  tflite_int_only_file
    tflite_int_only = os.path.join('models', x)

    x = app + '_' +  tflite_FP16_file
    tflite_fp16 = os.path.join('models', x)

    x = app + '_' +  tflite_16by8_file
    tflite_16by8 = os.path.join('models', x)


    # create converter

    # always use converter from save model (vs from keras instanciated model)
    
    """
    print('PABOU: load keras model from SavedModel %s' %(tf_dir))
    model = tf.keras.models.load_model(tf_dir) # dir
    """

    # recommended . create converter from disk based savedModel
    print ('PABOU: convert to TFlite flat buffer from SavedModel: ' , tf_dir)
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
    
    
    """
    # use h5

    print('PABOU: load keras model from h5 file %s, to convert to TFlite' %(h5_file))
    model = tf.keras.models.load_model(h5_file) # file

    print('PABOU: convert to TFlite flat buffer from Keras model')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    """

    """
    prior 2.3 ?
    print('PABOU: use TF ops as well as TFlite')
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    """




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

        print('\nPABOU: INT8 only for TPU. creates file: %s\n' % tflite_int_only)

        
        # This enables quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # This sets the representative dataset for quantization
        converter.representative_dataset = representative_dataset_gen

        # This ensures that if any ops can't be quantized, the converter throws an error
        #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] 
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
        # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.

        converter.target_spec.supported_types = [tf.int8]
        # These set the input and output tensors to uint8 (added in r2.3)
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        try:
            tflite_model = converter.convert()
            open(tflite_int_only, "wb").write(tflite_model)

            print('\nPABOU:===== OK INT8 only. CPU, TPU, MICRO. %s ' % tflite_int_only)
            print('full model %0.1f, lite mode %0.1f ' %( os.path.getsize(h5_file)/meg, os.path.getsize(tflite_int_only)/meg )) 

            see_tflite_tensors(tflite_model)

            try:
                add_tflite_meta_data(app, tflite_int_only, model_type, hot_size_pi, description) # pass file name which exist in file system, not binary model
                # model 1 only for now
            except Exception as e:
                print('PABOU: cannot add metadata to %s. %s. maybe model 2' %(tflite_fp32, str(e)))

        except Exception as e:
            print('\nPABOU:===== Exception converting to tflite INT8 only ', str(e))
            #module 'tensorflow.lite.python.schema_py_generated' has no attribute 'Model'
            # Failed to parse the model: pybind11::init(): factory function returned nullptr.


    def fp32(model_type, hot_size_pi, description):
        ##############################################################
        # case 1: no quantization. fp32 value for all 
        ##############################################################
        # hot needed for metadata

        print('\nPABOU: default fp32 without quantization. creates file: %s\n' % tflite_fp32)

        converter.inference_input_type = tf.float32 # was set to int8 in tpu
        converter.inference_output_type = tf.float32
        
        tflite_model = converter.convert() 
        # tflite model is a bytes b' \x00\x00\x00TFL3\x00\x00\x00\  <class 'bytes'>
        

        open(tflite_fp32, "wb").write(tflite_model) # complete name in file system with prefix
        print('\nPABOU:==== OK: created %s. using 32-bit float values for all parameter data\n\n' %tflite_fp32)

        see_tflite_tensors(tflite_model)

        print('PABOU: full model %0.1f Meg, lite mode %0.1f Meg ' %( os.path.getsize(h5_file)/meg, os.path.getsize(tflite_fp32)/meg )) 

        ##############################################
        # metadata. only implemented for model 1
        ##############################################
        print('PABOU:add metadata to TFlite file model 1')
        try:
            add_tflite_meta_data(app, tflite_fp32, model_type, hot_size_pi, description) # pass file name which exist in file system, not binary model
            # model 1 only for now
        except Exception as e:
            print('PABOU: cannot add metadata to %s. %s. maybe model 2' %(tflite_fp32, str(e)))
        

        #Vous pouvez utiliser Netron pour visualiser vos métadonnées,

    def default(model_type, hot_size_pi, description):

        #############################################################
        # case 3: dynamic range. fixed params, ie Weights converted from fp to int8
        # variable data still fp , no representative data set
        # 4x smaler, 2,3 x speed
        # for CPU
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

        print('\nPABOU: Default, Weigths converted to int8. creates file: %s \n' % tflite_default)

        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.inference_input_type = tf.float32 # was set to int8 in tpu
        converter.inference_output_type = tf.float32

        try:
            tflite_model = converter.convert()
            open(tflite_default, "wb").write(tflite_model) # tflite_model_size is a file name
            print('\nPABOU:===== OK %s. quantized weights, but other variable data is still in float format.\n\n' %tflite_default)
            
            see_tflite_tensors(tflite_model)

            print('full model %0.1f, lite mode %0.1f ' %( os.path.getsize(h5_file)/meg, os.path.getsize(tflite_default)/meg )) 

            try:
                add_tflite_meta_data(app, tflite_default, model_type, hot_size_pi, description) # pass file name which exist in file system, not binary model
                # model 1 only for now
            except Exception as e:
                print('PABOU: cannot add metadata to %s. %s. maybe model 2' %(tflite_fp32, str(e)))

        except Exception as e:
            print('\nPABOU:==== Exception converting to tflite weigth int', str(e))
    

   

    def default_variable(model_type, hot_size_pi, description):

        #############################################################
        # case 3.1: dynamic range. fixed and variable params
        # use representative data set
        # 4x smaler, 2,3 x speed
        # for CPU
        #############################################################

        print('\nPABOU: fixed and variable converted to int8. creates file: %s \n' % tflite_default_representative)

        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen

        try:
            tflite_model = converter.convert()
            open(tflite_default_representative, "wb").write(tflite_model) # tflite_model_size is a file name
            print('\nPABOU:===== OK %s. fiwed and variable quantized .\n\n' %tflite_default)
            
            see_tflite_tensors(tflite_model)

            print('full model %0.1f, lite mode %0.1f ' %( os.path.getsize(h5_file)/meg, os.path.getsize(tflite_default_representative)/meg )) 

            try:
                add_tflite_meta_data(app, tflite_default_representative, model_type, hot_size_pi, description) # pass file name which exist in file system, not binary model
                # model 1 only for now
            except Exception as e:
                print('PABOU: cannot add metadata to %s. %s. maybe model 2' %(tflite_fp32, str(e)))

        except Exception as e:
            print('\nPABOU:==== Exception converting to tflite weigth int', str(e))

            
    def int_fp_fallback(model_type, hot_size_pi, description):
        ##############################################################
        # case 4: full integer quantization . all math integer
        # measure dynamic range thru sample data
        # 4x smaller, 3x speeded
        # CPU,  not tpu TPU, micro as input/output still float
        ##############################################################


        """
        You can get further latency improvements, reductions in peak memory usage, and compatibility with integer 
        only hardware devices or accelerators by making sure all model math is integer quantized.

        For full integer quantization, you need to measure the dynamic range of activations and inputs by supplying 
        sample input data to the converter. Refer to the representative_dataset_gen() function used in the following code.
        
        Integer with float fallback (using default float input/output)
        In order to fully integer quantize a model, but use float operators when they don't have an integer implementation (to ensure conversion occurs smoothly),
        use the following steps:

        This tflite_quant_model won't be compatible with integer only devices (such as 8-bit microcontrollers) and 
        accelerators (such as the Coral Edge TPU) because the input and output still remain float in order to have the 
        same interface as the original float only model.

        That's usually good for compatibility, but it won't be compatible with devices that perform only integer-based operations, 
        such as the Edge TPU.

        Additionally, the above process may leave an operation in float format if TensorFlow Lite doesn't include a 
        quantized implementation for that operation. This strategy allows conversion to complete so you have a smaller and more 
        efficient model, but again, it won't be compatible with integer-only hardware. 
        (All ops in this MNIST model have a quantized implementation.)

        Now all weights and variable data are quantized, and the model is significantly smaller compared to the original TensorFlow Lite model.
        However, to maintain compatibility with applications that traditionally use float model input and output tensors, the TensorFlow Lite Converter leaves the model input and output tensors in float:
        """

        print('\nPABOU: full integer quantization, with fall back to fp. need representative data set. creates: %s \n' %tflite_int_fp_fallback)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen

        #Cannot set tensor: Got value of type NOTYPE but expected type FLOAT32 for input 0, name: flatten_input 
        try:
            tflite_model = converter.convert()
            open(tflite_int_fp_fallback, "wb").write(tflite_model)
            print('\nPABOU:===== OK %s. However, to maintain compatibility with applications that traditionally use float model input and output tensors, the TensorFlow Lite Converter leaves the model input and output tensors in float.\n\n' %tflite_int_fp_fallback)
            
            see_tflite_tensors(tflite_model)

            print('full model %0.1f, lite mode %0.1f ' %( os.path.getsize(h5_file)/meg, os.path.getsize(tflite_int_fp_fallback)/meg ))
            
            try:
                add_tflite_meta_data(app, tflite_int_fp_fallback, model_type, hot_size_pi, description) # pass file name which exist in file system, not binary model
                # model 1 only for now
            except Exception as e:
                print('PABOU: cannot add metadata to %s. %s. maybe model 2' %(tflite_fp32, str(e)))

        except Exception as e:
            print('\nPABOU:===== Exception converting to tflite int fall back fp', str(e))
        

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

        print('\nPABOU: quantization FP16. GPU acceleration delegate.  creates file: %s\n' % tflite_fp16)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        try:
            # COLAB 2.4 TFLITE_BUILTINS_INT8 requires smallest supported type to be INT8.

            tflite_model = converter.convert()
            open(tflite_fp16, "wb").write(tflite_model)
            print('\nPABOU:==== OK FP16, GPU delegate. ', tflite_fp16)

            see_tflite_tensors(tflite_model)

            print('full model %0.1f, lite mode %0.1f ' %( os.path.getsize(h5_file)/meg, os.path.getsize(tflite_fp16)/meg )) 

            try:
                add_tflite_meta_data(app, tflite_fp16, model_type, hot_size_pi, description) # pass file name which exist in file system, not binary model
                # model 1 only for now
            except Exception as e:
                print('PABOU: cannot add metadata to %s. %s. maybe model 2' %(tflite_fp32, str(e)))

        except Exception as e:
            print('\nPABOU:===== Exception converting to tflite FP16. ', str(e))
        

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

        print('\nPABOU: quantization 16x8.  creates file: %s\n ' % tflite_16by8)

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
            print('\nPABOU:===== OK int only with activation 16 bits\n\n', tflite_16by8)

            see_tflite_tensors(tflite_model)

            print('full model %0.1f, lite mode %0.1f ' %( os.path.getsize(h5_file)/meg, os.path.getsize(tflite_16by8)/meg )) 

            try:
                add_tflite_meta_data(app, tflite_16by8, model_type, hot_size_pi, description) # pass file name which exist in file system, not binary model
                # model 1 only for now
            except Exception as e:
                print('PABOU: cannot add metadata to %s. %s. maybe model 2' %(tflite_fp32, str(e)))

        except Exception as e:
            print('\nPABOU:===== Exception converting to tflite 16x8', str(e))


    ######################################
    # run all TFlite conversions
    #####################################   

    # order of conversion is define here

    # works OK on windows tf2.4.1 and on colab
    # input output are fp32
    # metadata is only for model 1

    fp32(model_type,hot_size_pi, 'fp32 TFlite model. input/output are fp32 .input are one hot encoded pitch index') # metadata only for type 1
    default(model_type,hot_size_pi, 'default TFlite model. Weigths encoded to int8. input/output are fp32 input are one hot encoded pitch index') # Weigth to int8
    fp16(model_type,hot_size_pi, 'fp16 TFlite model. inpout/output are fp32. input are one hot encoded pitch index')
    experimental(model_type,hot_size_pi, 'experimental TFlite model. input/output are fp32. input are one hot encoded pitch index')
    
    # below uses representative data generator

    # BACH
    # does not work on Windows for bach. 

    # does not work on colab for bach
    # Exception converting to tflite INT8 only  module 'tensorflow.lite.python.schema_py_generated' has no attribute 'Model'
    # Exception converting to tflite weigth int module 'tensorflow.lite.python.schema_py_generated' has no attribute 'Model'
    # Exception converting to tflite int fall back fp module 'tensorflow.lite.python.schema_py_generated' has no attribute 'Model'
    # Exception converting to tflite INT8 only  Failed to parse the model: pybind11::init(): factory function returned nullptr.
 
    # seems bug with RNN and INT

    # MNIST
    # work on colab for MNIST
    # does not work on Windows module 'tensorflow.lite.python.schema_py_generated' has no attribute 'Model

   
    # order seems to matter, otherwize fp32 will get int8 input.
    #PABOU:===== Exception converting to tflite FP16.  TFLITE_BUILTINS_INT8 requires smallest supported type to be INT8.

    tpu(model_type,hot_size_pi, 'INT8 only . For TPU, CPU, Micro. input/output are uint8. input are one hot encoded pitch index') # work on colab tf 2.4.1. does not seem to work on conda with tf 2.4.1
    default_variable(model_type,hot_size_pi, 'default with representative data set. Fixed and variable are int8. input/output are uint8. input are one hot encoded pitch index')
    int_fp_fallback(model_type,hot_size_pi, 'Integer with floating point fall back. input/output are uint8. input are one hot encoded pitch index')

    print('\n\nPABOU: All tfLite models created as .tflite files')




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
    # full model, iterate , ie one at a time like TFlite invoque
    # iterate, return one softmax
    ##############################################

    start_time = time.time() 
    
    for i in range(nb):  # use for loop as for lite (one inference at a time)

        if model_type == 2:
            if x2_test != []:  # velocity
            # test on [] vs None
                x = [x_test[i], x1_test[i], x2_test[i]] # x list of 3 list of seqlen int
                y = [y_test[i], y1_test[i], y2_test[i]]
            else:
                x = [x_test[i], x1_test[i]]
                y = [y_test[i], y1_test[i]]

        if model_type == 1:  
                x =x_test[i]  #  one hot len 40  (40, 483)
                y =y_test[i]  # one hot, len 483 (483,)
        
        # keras need a batch dimension # got from (40,483) to (1,40,483). 40 is seqlen. 483 is one hot size with concatenate
        if model_type == 1:
            input = np.expand_dims(x,axis=0)    #Error when checking input: expected flatten_2_input to have 3 dimensions, but got array with shape (28, 28)
            
        else:
            input = x # not sure need to add batch dimension for multihead

        
        result = model.predict(input) # Keras call. 

        # for model 2, result[[]]  is a list of multiple softmax.  result[0] [0] (pitches), result[1] [0] , result[2] [0]
        # nothing identify which param the softmax refers to, assume same order from model definition
        # np.sum(result[0] [0]) = 1
        #results = np.squeeze(output_data) is the same as [0]

        # model 1 result is [[]] result[0] is a simple softmax for pitches.  result[1] does not exist 
        
        
        elapse = (time.time() - start_time)*1000.0 # time.time() nb of sec since 1070. *1000 to get ms

        if model_type == 1:
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

    if model_type == 1:
        result=model.predict(x_test[:nb]) # list of softmax
        # result.shape (100, 483)   list of 100 softmax .  for MNIST (100,10)

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
            if model_type == 1: 
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
    

##########################################################
# TF lite inference
# x single sample
# x1 and x2 could be  []
##########################################################

def TFlite_single_inference(x_test, x1_test, x2_test , TFlite_file, model_type):
    # x_test.shape  (40, 483)
    inter = tf.lite.Interpreter(TFlite_file) # could be done outside only once

    inter.allocate_tensors()

    input_details= inter.get_input_details()
    #[{'dtype': <class 'numpy.float32'>, 'index': 0, 'name': 'serving_default_input_1:0', 'quantization': (...), 'quantization_parameters': {...}, 'shape': array([  1,  40, 483]), 'shape_signature': array([ -1,  40, 483]), 'sparsity_parameters': {}}]
    # for model 2 list of 3 dict  input_details [0] , input_details [1], ...
    # for model 1 ist with one dict.  
    # keys: dtype, shape , name , quantization ..

    output_details= inter.get_output_details() 
    #[{'dtype': <class 'numpy.float32'>, 'index': 62, 'name': 'StatefulPartitionedCall:0', 'quantization': (...), 'quantization_parameters': {...}, 'shape': array([  1, 483]), 'shape_signature': array([ -1, 483]), 'sparsity_parameters': {}}]
    
    ##########################################################
    # look at input output details
    # build dict output head name to index
    ##########################################################
    """
    0  velocity , output shape [1 5]
    1  pitch , output shape [  1 129]   array([  1,  40, 483] input for model 1
    2 duration , output shape [ 1 41]
    order in list of dict not then same as model definition 
    """
    d={}    # mapping layer name to index used in input, output details
    # d['picth] is index    input_details[index] ['index', 'shape' , etc ..]

    if model_type == 2:
        try:
            for i in [0,1,2]: # WARNING. 2 only if velocity 
                #print('PABOU: TF lite input type %s, output type %s'%(input_details[i] ['dtype'],output_details[i]['dtype']))
                #print('PABOU: TF lite input shape %s, output shape %s'%(input_details[i]['shape'],output_details[i]['shape']))
                #print('PABOU: TF lite input name %s, output name %s'%(input_details[i]['name'],output_details[i]['name']))
                
                name = input_details [i] ['name']
                name = name.split('_') [-1]
                name = name.split(':')[0]
                d[name] = i
                #print (d) # map name of output head to index used in input_details[index]

        except Exception as e:
            print('PABOU: look at input output details ',str(e))
            

        try:
            for i in [0,1,2]: # number or output heads
                assert input_details[i] ['dtype'] in [np.float32 ,  np.int8, np.uint8]
        except Exception as e:
            print('PABOU: input details unexpected', str(e))

    
    if model_type == 1: # pitch index 0 as the only one; if multiple head, this is not gatenteed
        try:
            for i in [0]: # could avoid  input_details[i] is just input_details
                    #print('PABOU: TF lite input type %s, output type %s'%(input_details[i] ['dtype'],output_details[i]['dtype']))
                    #print('PABOU: TF lite input shape %s, output shape %s'%(input_details[i]['shape'],output_details[i]['shape']))
                    #print('PABOU: TF lite input name %s, output name %s'%(input_details[i]['name'],output_details[i]['name']))

                    # build dict to map names to softmax

                    name = input_details [i] ['name']   # TF lite input name serving_default_flatten_input:0
                    name = name.split('_') [-1]
                    name = name.split(':')[0]
                    
                    #print(name)
                    d[name] = i  # {'pitch' : 0} for model 1
                    #print (d)
        except Exception as e:
            print('PABOU: look at input output details ',str(e))

    # Check if the input type is quantized, then rescale input data to uint8

    if model_type == 1:
        if input_details[0]['dtype'] == np.uint8:
            #print('PABOU: input are uint8')
            input_scale, input_zero_point = input_details[0]["quantization"]
            x_test = x_test / input_scale + input_zero_point

    if model_type == 2:
        index = d['pitch'] 
        if input_details[index]['dtype'] == np.uint8:
            #print('PABOU: pitch input are uint8')
            input_scale, input_zero_point = input_details[index]["quantization"]
            x_test = x_test / input_scale + input_zero_point

        index = d['duration']
        if input_details[index]['dtype'] == np.uint8:
            #print('PABOU: duration input are uint8')
            input_scale, input_zero_point = input_details[index]["quantization"]
            x1_test = x1_test / input_scale + input_zero_point

        if x2_test != []:  # velocity
                index = d['velocity']
                if input_details[index]['dtype'] == np.uint8:
                    #print('PABOU: velocity input are uint8')
                    input_scale, input_zero_point = input_details[index]["quantization"]
                    x2_test = x2_test / input_scale + input_zero_point

    # single inference

    #################################
    # model 1
    #################################

    if model_type == 1:

        input_x = np.expand_dims(x_test, axis=0).astype(input_details[0]["dtype"])
        # 'cast' x_test to expected input float or uint or else , for MNIST (1,28,28) dtype uint8

        #test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])

        #input_x = np.float32(input_x) # NO. could be int8

        try:
            index = d['pitch'] # should be 0 for model type 1
        except:
            index = d['input'] # MNIST output name 
        inter.set_tensor(input_details[index]['index'], input_x) 


    #################################
    # model 2
    #################################

    if model_type == 2:

        ############################
        # for all output head
        # get index
        # add batch dimension and cast to rigth type
        ############################

        index = d['pitch'] 
        
        input_x = x_test
        input_x = np.expand_dims(input_x, axis=0).astype(input_details[index]["dtype"])
        inter.set_tensor(input_details[index]['index'], input_x)   

        # error there. TFlite expect shape [1,1] and input_x is [1,40 ] ie seqlen
        # FINALLY set input shape in model definition (40,)  was(None,) 

        #fhttps://www.tensorflow.org/lite/guide/faq
    

        index = d['duration']
        input_x = x1_test
        input_x = np.expand_dims(input_x, axis=0).astype(input_details[index]["dtype"])         
        inter.set_tensor(input_details[index]['index'], input_x)

        if x2_test != []:  # velocity
            index = d['velocity']
            input_x = x2_test
            input_x = np.expand_dims(input_x, axis=0).astype(input_details[index]["dtype"])
            inter.set_tensor(input_details[index]['index'], input_x)
        # velocity
    
    
    ###########################################
    inter.invoke()
    ###########################################

    # output_details list of dict. same order as input


    if model_type == 1:
        softmax_pi = inter.get_tensor(output_details[0]['index']) [0]
        # array([  0,   0,   0,   1,   0,   7,   0,   0, 247,   0], dtype=uint8) for TPU MNI[ST
        softmax_du = []
        softmax_ve = []
    
    if model_type == 2:
        softmax_pi = inter.get_tensor(output_details[d['pitch']]['index']) [0]
        softmax_du = inter.get_tensor(output_details[d['duration']]['index']) [0]
        if x2_test != []:
            softmax_ve = inter.get_tensor(output_details[d['velocity']]['index']) [0]
        
    return(softmax_pi, softmax_du, softmax_ve)


#####################################################
# benchmark TFlite for ONE quantized model, passed as parameter as file name
# only iteration (vs slide)
# look at ELAPSE and ACCURACY (vs simple inference). use test set as input
# display with pretty table
# params: any string, file name of TFlite model, x test, y test, number of inferences, TFlite file size
#####################################################
def bench_lite_one_model(st,TFlite_file,x_test,y_test,x1_test, y1_test,x2_test, y2_test, nb, model_type, app):

    print("\nPABOU: running LITE benchmark %s for : %s, model type %d, app %s: " %(st,TFlite_file, model_type, app))

    TFlite_file_size = round(float(os.path.getsize(TFlite_file)/(1024*1024)),1) # to print in table

    h5_size = get_h5_size(app)

    # create table
    pt = PrettyTable()
    pt.field_names = ["model", "iters", "infer (ms)",  "acc pi", "acc du", "acc ve",  "h5 size", 'TFlite size']

    # 1 and 2 are [] for model 1
    # 2 is [] for model 2 and no velocity

    # x_test[0] array of int for model 2 (embedding), len = seqlen   # x_test[i] (40,0)
    # x_test[i] one hot for model 1

    error_pi = 0
    error_du = 0
    error_ve = 0

    start_time = time.time()

    for i in range(nb):  
        ############################
        # inference
        ############################


        if x1_test == []:  # avoid out of range
            x1= []
        else:
            x1 = x1_test[i]

        if x2_test== []:
            x2= []
        else:
            x2 = x2_test[i]

        (softmax_pi, softmax_du, softmax_ve) = \
        TFlite_single_inference(x_test[i], x1, x2, TFlite_file, model_type)
        
        # test if Y is a softmax (bach) or an int (MNITS)
        # softmax is not a LIST but an ndarray
        
        # accuracy for one inference
        if isinstance(y_test[i],np.ndarray):
            if np.argmax(softmax_pi) != np.argmax(y_test[i]): # use squeeze to get from [[ ]] to []
                error_pi = error_pi + 1
        else:
            if np.argmax(softmax_pi) != y_test[i]: 
                error_pi = error_pi + 1


        if model_type == 2:
            
            if isinstance(y_test[i],np.ndarray):
                if np.argmax(softmax_du) != np.argmax(y1_test[i]): 
                    error_du = error_du + 1
            else:
                if np.argmax(softmax_du) != y1_test[i]: # use squeeze to get from [[ ]] to []
                    error_du = error_du + 1


            if x2_test != []: # means velocity
                if isinstance(y_test[i],np.ndarray):
                    if np.argmax(softmax_ve) != np.argmax(y2_test[i]): # use squeeze to get from [[ ]] to []
                        error_ve = error_ve + 1
                else:
                    if np.argmax(softmax_ve) != y2_test[i]: # use squeeze to get from [[ ]] to []
                        error_ve = error_ve + 1

        # error are initialized at 0. so if not use stays zero
    # for range
    
    elapse = (time.time() - start_time)*1000.0

    # average elapse per iteration
    elapse_per_iter = elapse / float(nb)

    # accuracies
    e1= float(nb - error_pi) / float(nb)
    e2= float(nb - error_du) / float(nb)
    e3= float(nb - error_ve) / float(nb)
    
    
    print("\nPABOU: TF lite model: %s. elapse %0.5f ms, %d samples." % (TFlite_file, round(elapse_per_iter,2), nb))
    print("\nPABOU: acc pi: %0.2f, acc du: %0.2f, acc ve: %0.2f. " % (e1,e2,e3))

    # models\mnist_quantized-with-floating-point-fallback.tflite
    # TFfile name can be long. screw pretty table

    s= TFlite_file.replace('models', '')
    s = s.replace('.tflite', '')
    
    pt.add_row([s, nb, round(elapse_per_iter,2), e1, e2, e3, h5_size, TFlite_file_size ])
    print('\n\n', pt)
    
    # return to print in main
    return([s, nb, round(elapse_per_iter,2), e1, e2, e3, h5_size,  TFlite_file_size ])
   

###################################################
# MODEL evaluate 
# print pretty table
# to be used for model just fitted , or model loaded 
####################################################
def model_evaluate(model, model_type, x , y):
    print("\n\nPABOU: evaluate model on test set") 
    # x is a list of X for multiple input

    if model_type == 1:
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
# alter prediction softmax, return new.
#########################################
def get_temperature_pred(pred, temp):
    
    p=np.asarray(pred).astype('float64')
    # temperature magic
    p=np.log(p) / float(temp)
    p=np.exp(p)
    # make it a proba
    p= p / np.sum(p) 
    p = np.random.multinomial(1,p,1)
    return(p)
















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

