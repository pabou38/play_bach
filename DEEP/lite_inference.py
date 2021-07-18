
import numpy as np
import os,sys,time
import math
from prettytable import PrettyTable

#########################################
# duplicate/extract from pabou.py
# cannot import both tensorflow and tflite_runtime.interpreter
# this module does NOT import tensorflow
# dedicated to benchmark and inference for Coral
# do NOT import tensorflow prior importing this module to use coral
#########################################

##########################################################
# single TF lite inference
# called by benchmark and real time predict
# interpreter called before, so works for both normal tflite and edge TPU
# x single sample
# x1 and x2 could be  [] 
# need for dict for multihead, mapping layer name to index
# add batch dimension
# 'serving_default_x:0' ??

# returns array of softmax
# no use of tf.
##########################################################
def TFlite_single_inference(x_test, x1_test, x2_test , interpreter, model_type):
    #  x_test is single sample
    # array of int32
    # interpreter passed as argument. interpreter = tf.lite.Interpreter(TFlite_file)
    # can also create interpreter from converted model: interpreter = tf.lite.Interpreter(model_content=tflite_model)


    #######################################################################
    # TPU works fine with CNN MNIST
    # fails with bach LSTM
    # external/org_tensorflow/tensorflow/lite/kernels/unidirectional_sequence_lstm.cc:938 
    # num_intermediate_tensors == 5 was not true.Node number 1 (UNIDIRECTIONAL_SEQUENCE_LSTM) failed to prepare. 
    # TO DO: test with MNIST LSTM TFLITE colab
    #######################################################################
    interpreter.allocate_tensors()

    input_details= interpreter.get_input_details() # input_details is a list of dictionaries
    """
    [{
        'name': 'serving_default_x:0', 
        'index': 0, 
        'shape': array([  1,  40, 483]), 
        'shape_signature': array([  1,  40, 483]), 
        'dtype': <class 'numpy.float32'>, 
        'quantization':  (0.0, 0) ,
        'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0} ,
        'sparsity_parameters': {}
        }]
    
    INT8 only
     'dtype': <class 'numpy.uint8'>, 
     'quantization': (0.003921568859368563, 0), 
     'quantization_parameters': {'scales': array([0.00392157], ...e=float32), 'zero_points': array([0]), 'quantized_dimension': 0}, 
                         scales array([0.00392157], dtype=float32)
     'sparsity_parameters': {}}
    """

    # for model 2 list of 3 dict  input_details [0] , input_details [1], ...
    # for model 1 ist with one dict is input_details[0].  

    output_details= interpreter.get_output_details() 
    """
    {
        'name': 'StatefulPartitionedCall:0', 
        'index': 48, 
        'shape': array([  1, 483]), 
        'shape_signature': array([  1, 483]), 
        'dtype': <class 'numpy.float32'>, 
        'quantization': (0.0, 0), 
        'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 
        'sparsity_parameters': {}
    }
    INT8 only

    {
        'name': 'StatefulPartitionedCall:0', 
        'index': 60, 
        'shape': array([  1, 483]), 
        'shape_signature': array([  1, 483]), 
        'dtype': <class 'numpy.uint8'>, 
        'quantization': (0.00390625, 0), 
        'quantization_parameters': {'scales': array([0.00390625], ...e=float32), 'zero_points': array([0]), 'quantized_dimension': 0}, 
        'sparsity_parameters': {}
        }

    """

    ##########################################################
    # look at input and output details
    # build dict output head name to index
    ##########################################################
    """
    0  velocity , output shape [1 5]
    1  pitch , output shape [  1 129]   array([  1,  40, 483] input for model 1
    2 duration , output shape [ 1 41]
    order in list of dict not then same as model definition 
    """
    d={}    # mapping layer name as defined in model to index used in input, output details
    # d['picth] is index    input_details[index] ['index', 'shape' , etc ..]

    
    #####################################
    # single head
    #####################################

    if model_type in [1,3,4]:

        assert input_details[0] ['dtype'] in [np.float32 , np.int32, np.int8]

         # Check if the input type is quantized, then rescale input data to uint8
        if input_details[0]['dtype'] == np.int8:
            # rescale input X_test: (100,) dtype('int32')
            input_scale, input_zero_point = input_details[0]["quantization"]
            x_test = x_test / input_scale + input_zero_point # turn to float  (100,)  dtype('float64')

        ##########################################################
        # input index always = 0, since single head
        # add last dimension to comply with signature 
        # cast to expected input type
        # reshape for CON2D
        # set input tensor
        ##########################################################

        if model_type == 4:
            # reshape from seqlen = size*size into sizexsize 
            d = x_test.shape[0] 
            d = int(math.sqrt(d))
            x_test = np.reshape(x_test, (d, d)) 

        # add extra batch dimension (1,...)  
        # can also use x_test[i:i+1] to add batch dimension
        # convert float to int8 by truncating decimal part
        input_x = np.expand_dims(x_test, axis=0).astype(input_details[0]["dtype"]) # (100,) to (1,100)
        
        # in model 3, x_test WAS array([85, 52, ... 51, 57], dtype=int16)   shape(40,) 
        # after expand (1,40) but model required 3 dimensions, as set in signature. last dim is hot size and harcoded to 1
        if model_type == 3: # add last dimension to match LSTM model signature
            input_x = np.expand_dims(input_x, axis=-1) # (1,40,1)  make sure number of dims matches, otherwize error

        # input_details[index=0 for single head] ['index']  is the input buffer
        interpreter.set_tensor(input_details[0]['index'], input_x) 


    if model_type == 2:
    ########################################
    # multi head, build dictionary of indexes
    ########################################

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
                assert input_details[i] ['dtype'] in [np.float32 , np.int32, np.int8]
        except Exception as e:
            print('PABOU: input dtype unexpected', str(e))

        
        ############################
        # get input index from dictionary
        # for all output head
        # add batch dimension and cast to rigth type
        # set input tensor
        ############################

        index = d['pitch'] 
        if input_details[index]['dtype'] == np.int8:
            #print('PABOU: pitch input are uint8')
            input_scale, input_zero_point = input_details[index]["quantization"]
            x_test = x_test / input_scale + input_zero_point
            # still original dtype. will be set to uint8 when expanding dim

        index = d['duration']
        if input_details[index]['dtype'] == np.int8:
            #print('PABOU: duration input are uint8')
            input_scale, input_zero_point = input_details[index]["quantization"]
            x1_test = x1_test / input_scale + input_zero_point
            

        if x2_test != []:  # velocity
                index = d['velocity']
                if input_details[index]['dtype'] == np.int8:
                    #print('PABOU: velocity input are uint8')
                    input_scale, input_zero_point = input_details[index]["quantization"]
                    x2_test = x2_test / input_scale + input_zero_point
        

        index = d['pitch'] 
        input_x = x_test
        input_x = np.expand_dims(input_x, axis=0).astype(input_details[index]["dtype"])
        input_x = np.expand_dims(input_x, axis=-1) # (1,40,1)
        interpreter.set_tensor(input_details[index]['index'], input_x)   
    
        index = d['duration']
        input_x = x1_test
        input_x = np.expand_dims(input_x, axis=0).astype(input_details[index]["dtype"])
        input_x = np.expand_dims(input_x, axis=-1) # (1,40,1)
        interpreter.set_tensor(input_details[index]['index'], input_x)

        if x2_test != []:  # velocity
            index = d['velocity']
            input_x = x2_test
            input_x = np.expand_dims(input_x, axis=0).astype(input_details[index]["dtype"])
            input_x = np.expand_dims(input_x, axis=-1) # (1,40,1)
            interpreter.set_tensor(input_details[index]['index'], input_x)
        # velocity



    # set input interpreter.set_tensor(input_details[index]['index'], input_x)     
    ###########################################
    # INFERENCE at least
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start # in sec
    inference_time = float(inference_time) * 1000.0 # in ms
    ###########################################

    # output_details list of dict. same order as input
    # use get_tensor to retrieve softmax from output_details[0 always for single output] ['index']

    # can also use squeeze: Remove single-dimensional entries from the shape of an array. [[ ]] to []
    #output_data = inter.get_tensor(output_details[0]['index'])   # (1, 10)
    #softmax = np.squeeze(output_data) # from array([[1.36866065e-05, to array([1.36866065e-05, 

    if model_type in [1,3,4]: #always index 0 , also use [0] or squeeze as there is only one softmax
        softmax_pi = interpreter.get_tensor(output_details[0]['index']) [0]  # only one softmax in [[]] (483,) dtype=float32)
        # array([  0,   0,   0,   1,   0,   7,   0,   0, 247,   0], dtype=int8) for TPU or array of float32 for fp32 model
        softmax_du = []
        softmax_ve = [] 

        # dequantize ie, assume int8 model are quantized float model ?
        if np.issubdtype(output_details[0]['dtype'], np.int8):
            scale, zero_point = output_details[0]['quantization'] # 
            softmax_pi = scale * (softmax_pi - zero_point) # turn softmax from array of int8 to array of float64
     
    if model_type == 2:
        ########## NEED TO DEQUANTIZE
        softmax_pi = interpreter.get_tensor(output_details[d['pitch']]['index']) [0]
        softmax_du = interpreter.get_tensor(output_details[d['duration']]['index']) [0]
        if x2_test != []:
            softmax_ve = interpreter.get_tensor(output_details[d['velocity']]['index']) [0]

    # Please note: TfLite fused Lstm kernel is stateful, so we need to reset the states.
    # Clean up internal states.
    interpreter.reset_all_variables() 


    # inference_time in msec
    return(softmax_pi, softmax_du, softmax_ve, inference_time)

#####################################################
# benchmark TFlite for ONE quantized model, passed as parameter as file name
# call TFlite_single_inference
# only iteration (vs slide)
# look at ELAPSE and ACCURACY (vs simple inference). use test set as input
# display with pretty table
# params: any string, file name of TFlite model, x test, y test, number of inferences, TFlite file size

# no use of tf.
#####################################################
def bench_lite_one_model(st,TFlite_file,x_test,y_test,x1_test, y1_test,x2_test, y2_test, nb, model_type, app, h5_size, coral=False):

    print("\nLITE: running LITE benchmark %s, for model: %s" %(st,TFlite_file))

    TFlite_file_size = round(float(os.path.getsize(TFlite_file)/(1024*1024)),1) # to print in table
    
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

    ##############################################################
    # create interpreter once for all. create from TFlite file
    # see also load_model(app, 'li', None, None)
    ##############################################################

    
    # CORAL
    # https://www.tensorflow.org/lite/guide/python
    """
    To quickly start executing TensorFlow Lite models with Python, you can install just the TensorFlow Lite interpreter, instead of all TensorFlow packages. We call this simplified Python package tflite_runtime.
    The tflite_runtime package is a fraction the size of the full tensorflow package and includes the bare minimum code required to run inferences with TensorFlow Lite—primarily the Interpreter Python class. This small package is ideal when all you want to do is execute .tflite models and avoid wasting disk space with the large TensorFlow library.
    """
    # need to install simplified Python package tflite_runtime
    # import tflite_runtime.interpreter as tflite
    # then change interpreter = tf.lite.Interpreter to interpreter = tflite.Interpreter
    if coral:
        try:
            import tflite_runtime.interpreter as tflite
        # generic_type: type "InterpreterWrapper" is already registered!
        # cannot import tensorflow and tflite_runtime at the same time
        except Exception as e:
            print('LITE: Exception cannot import coral runtime ', str(e))

        #Linux: libedgetpu.so.1
        #macOS: libedgetpu.1.dylib
        #Windows: edgetpu.dll
        #Now when you run a model that's compiled for the Edge TPU, TensorFlow Lite delegates the compiled portions of the graph to the Edge TPU.

        if sys.platform in ["win32"]:
            delegate = 'edgetpu.dll'
        if sys.platform in ['linux']:
            delegate = 'libedgetpu.so.1'
        print('LITE: using coral (vs tensorflow) interpreter, run time: %s' %delegate)

        # model file must be compiled for edgeTPU
        # cannot use standard lite interpreter module 'tensorflow._api.v2.lite' has no attribute 'load_delegate'
        interpreter = tflite.Interpreter(model_path = TFlite_file, experimental_delegates=[tflite.load_delegate(delegate)]) # 2 bong

    else:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path = TFlite_file) 
        print('LITE: using tensorflow (vs coral) interpreter')
        

    start = time.perf_counter() # sec float. timer for all unference
    # also .perfcounter_ns   nano sec return int

    for i in range(nb):  
        ############################
        # inference on x_test, also set x1 and x2 to be [] or test, depending on model
        ############################

        if x1_test == []:  # avoid out of range  for du
            x1= []
        else:
            x1 = x1_test[i]

        if x2_test== []:
            x2= []
        else:
            x2 = x2_test[i]
        
        # inference
        # extra dimention added to x_test in single_inference
        (softmax_pi, softmax_du, softmax_ve,inference_time_ms) = \
        TFlite_single_inference(x_test[i], x1, x2, interpreter, model_type)
        
        # test if Y is a softmax (bach) or an int (MNITS)
        # softmax is not a LIST but an ndarray. for INT8, array of uint8
        # y_test[1]   (483,) dim 1 dtype float32   np.sum(y_test[1]) = 1.0
        if isinstance(y_test[i],np.ndarray): # y is a one hot
            if np.argmax(softmax_pi) != np.argmax(y_test[i]): # use squeeze to get from [[ ]] to []
                error_pi = error_pi + 1
        else:
            if np.argmax(softmax_pi) != y_test[i]:  # MNIST y is integer
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

    elapse = (time.perf_counter() - start) * 1000.0 # convert to ms

    # average elapse per iteration
    elapse_per_iter = elapse / float(nb)

    # accuracies
    e1= round(float(nb - error_pi) / float(nb),2)
    e2= round(float(nb - error_du) / float(nb),2)
    e3= round(float(nb - error_ve) / float(nb),2)

    
    print("LITE: TF lite model: %s. elapse %0.5f ms, %d samples." % (TFlite_file, round(elapse_per_iter,2), nb))
    print("LITE: acc pi: %0.2f, acc du: %0.2f, acc ve: %0.2f. " % (e1,e2,e3))
    pt.add_row([os.path.basename(TFlite_file), nb, round(elapse_per_iter,2), e1, e2, e3, h5_size, TFlite_file_size ])
    print('\n', pt, '\n')
    print('\n')

    # return to print in main , as one pretty table row
    return([os.path.basename(TFlite_file), nb, round(elapse_per_iter,2), e1, e2, e3, h5_size, TFlite_file_size ])
