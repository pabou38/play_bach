#!/usr/bin/env  python

version = '1.4'

"""
TO DO
metadata for model 2
metadata for input LSTM vs image
exception pitch  4.7.11 ['4', '7', '11'] Cannot make a name out of ''  
"""

"""
install pyaudio
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
sudo apt-get install ffmpeg libav-tools
sudo pip install pyaudio 
install music21 tabulate tqdm
conda install -c anaconda pydot
"""

##################################################################
# dedicated edgtpu benchmark
# before importing tensorflow , imported in pabou
##################################################################
edge_bench = False

def bench_coral(edge_tpu_file,pick_x_y,nb,model_type,app,h5_size):
    import pickle
    #######################################################
    # Lite benchmark for edge TPU
    # last boolean indicate coral
    # tflite file in  models/edge
    # to run on coral DO NOT import tensorflow
    # need a pickle file with vectorized X, Y and edgetpu file 
    # model/app_x_y_test.pick. created after vectorization
    # edge tpu model file created when saving TFlite models
    # set edge_bench to True to run. will then exit
    #######################################################
    #import tflite_runtime.interpreter as tflite
    sys.path.insert(1, '..') # this is in above working dir
    import lite_inference

    print('CORAL: load test X and Y')
    with open (pick_x_y, 'rb') as fp:
        (x_test, y_test , x1_test, y1_test, x2_test, y2_test) = pickle.load(fp)

    #my_tflite_model = 'models/edge/mnist_TPU_lite_edgetpu.tflite' 
    my_tflite_model = edge_tpu_file 

    print('CORAL: doing EDGE benchmark for %s, with %d inferences' %(my_tflite_model,nb))
    
    try:
        # return pretty table row 
        # use specific module which does not import tensorflow
        # specific edge interpreter created in bench_lite based on coral boolean
        # need to pass h5_size
        b = lite_inference.bench_lite_one_model("TFlite coral", my_tflite_model ,x_test, y_test, x1_test, y1_test, x2_test, y2_test, nb, model_type , app, h5_size, True)
        pt.add_row(b)
        print(pt)
    except Exception as e:
        print('CORAL: exception in TFlite EDGE benchmark for %s.  %s\n'  %( my_tflite_model, str(e)))

    # call inference directly
    # create coral interpreter
    import tflite_runtime.interpreter as tflite

    average_inference_time = 0.0
    if sys.platform in ["win32"]:
            delegate = 'edgetpu.dll'
    if sys.platform in ['linux']:
        delegate = 'libedgetpu.so.1'

    # model file must be compiled for edgeTPU
    # cannot use standard lite interpreter module 'tensorflow._api.v2.lite' has no attribute 'load_delegate'
    interpreter = tflite.Interpreter(model_path = my_tflite_model, experimental_delegates=[tflite.load_delegate(delegate)])

    for i in range(nb):
        (softmax, _, _, inference_time) = lite_inference.TFlite_single_inference(x_test[i], [], [], interpreter, 1)
        average_inference_time = average_inference_time + inference_time # in msec, float
    
    average_inference_time = average_inference_time / float(nb)
    print('\nCORAL: average CORAL inference time ms %0.2f for %d inferences' %(average_inference_time,nb))


# cannot get context , CLI arguments from pabou, as it imports tensorflow. 
# hardcode for now
# edgetpu_compiler   -a to avoid  More than one subgraph is not supported

if edge_bench:
    import sys,os
    # must be created by edgetpu_compiler
    edge_tpu_file = 'models/edge/cello1_nc_so_TPU_lite_edgetpu.tflite'
    pick_x_y = os.path.join('models', 'cello1_nc_so_x_y_test.pick')
    nb = 1000
    model_type = 1
    app = 'cello1_nc_so_edge'
    h5_size = 0

    bench_coral(edge_tpu_file,pick_x_y,nb,model_type,app,h5_size) 
    sys.exit(0)


##################################
# import
#################################

print('importing ... ', sep=' ')
import logging
from tensorflow.python.training.checkpoint_management import _evaluate
logging.getLogger("tensorflow").setLevel(logging.ERROR)

"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

#from tensorflow.python.client import device_lib

import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer # multi hot, one hot
import sklearn
import music21
from music21 import converter, instrument, note, chord, stream
from music21 import corpus
#import pdb
import glob
import random
import pickle
from tabulate import tabulate
from collections import Counter
import requests
import platform
from tqdm import tqdm # progress bar
import json
import time
import logging
import datetime
import _thread # low level threading P3 _t
import threading # higher-level threading interfaces on top of the lower level _thread module
import queue # for thread
from multiprocessing import Process, Queue, cpu_count
from prettytable import PrettyTable

print('import config file' )
import config_bach # same dir

print('import model definition')
import my_model

# pabou.py is one level up
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '..') # this is in above working dir
#print ('sys path used to import modules: %s\n' % sys.path)
try:
    print ('importing pabou helper')
    import pabou # various helpers module, one dir up
    import lite_inference # single inference and benchmark
    # pip install prettytable
    print('pabou helper path: ', pabou.__file__)
    #print(pabou.__dict__)
except:
    print('cannot import pabou')
    exit(1)

print('all import done')

###################
# DONE once
# can also get MIDI file from other sources
###################

# create directory of all midi files from all bach in music21 built in corpus, done once.
from multiprocessing import queues
from numpy.lib.function_base import _place_dispatcher, place


def create_bach():
    for score in corpus.chorales.Iterator(numberingSystem='bwv', returnType='filename'):
        print ("parsing score, create MIDI file", score)
        b=music21.corpus.parse(score)
        music=music21.midi.translate.streamToMidiFile(b)
        music.open(score + ".mid", "wb")
        music.write()
        music.close()
          

# ##########################################################################################  
# Parse all MIDI files
# do not print stats. will be done later
# create dictionnaries 
# creates network input list of STRING, x and y. pitches, duration, velocity
############################################################################################  
def parse(corp): # directory
    print('PARSE: analyzing MIDI files in DIR %s and creating INTEGERS list and dictionaries' %(corp))
    corpus_notes=[] # only pitches (notes or chords normal or not) . can also contains pitches + duration concatenated
    corpus_duration=[]
    corpus_velocity=[]
    
    max_du = config_bach.max_duration # cap very long duration , cap number of entries
    
    print('normal mode ? ', config_bach.normal)

    i=0

    # source midi defined in config file
    for file in tqdm(glob.glob(corp)): # glob, use shell meta
        #print(file)
        i = i + 1

        try:
            if config_bach.chordify: 
                score=converter.parse(file).chordify() 
                #chordify: all notes in differents parts converted to chords in single line, not polyphonic
                # chordify seems to explode model. much more chords combination, specially without normal
            else:
                score=converter.parse(file)  # return music21.stream.Score 

            #score.show() 
            #score.show('text')
            #score.show('midi')
                
        except Exception as e:
            print('cannot parse score ' , file, str(e))
            sys.exit(1)
            
    
        # update list pi(all pitches) , su (all durations) , corpus_list list of tuple() notes/chord/rest and duration
        # everything is a string
        # duration capped, ignore very long. Prefix with D. stored as string
        # notes are string  C4.  # and - are b
        
        # chords are either list of element (int) , stored as string 1.2.3  NOTE, C4 and C5 are same int
        # use only string inside corpus_list and not list. cannot set otherwize
        # or list of string C4.C5.C6  to preserve octave, but explode number ie C4.D4 , C5.D4   but only one with integer
        # tried to add duration at the end of the pitch(s) . but number of labels exploded, manage 2 vocab
        # flatten , ie should see all parts, loose concept of nested streams

        for element in score.flat:  

            ###########################################
            # Note
            ###########################################
            if isinstance(element,note.Note): # <music21.note.Note G>
                # pitch is an object str convert it to D4
                # element.pitch.midi
                
                """
                #symbols for sharp or flat, (# or - respectively).
                print('octave: ' , element.pitch.octave)
                print('name: ' , element.pitch.name)
                print('name with octave: ' , element.pitch.nameWithOctave)
                print('MIDI: ' , element.pitch.midi)

                print('steps: ' , element.step) #

                print('duration type: ' , element.duration.type)
                print('duration dots: ' , element.duration.dots)
                print('duration quarter: ' , element.duration.quarterLength)
                # default is 1.0 , ie quarter

                element.show()
                """

                p= str(element.pitch) #  get C4  from <music21.pitch.Pitch G4> print or str
                # could also do element.pitch.name + element.octave 
              
                # element.duration <music21.duration.Duration 1.0> 
                # element.offset 0.0
                 
                d= element.duration.quarterLength
                if not isinstance(d,float): # class fraction  1/3
                    d1 = float(d.numerator) / float(d.denominator)   
                else:
                    d1 = float(d)

                # cap long duration
                    
                d1 = min(d1,max_du) # ignore very long, less labels 
                d1= '%0.1f' %(d1) # round to 2 decimal and convert to srt
                d1= '$' + d1  # to distingish from normal C4 C5 which is one integer

                # velocity      element.volume <music21.volume.Volume realized=0.79>   element.volume.velocity 100
                velocity = element.volume.velocity
                
                # concatenate piches and duration. $ separator . only for model 1
                if config_bach.concatenate and config_bach.model_type == 1:
                    p = p + d1

                corpus_notes.append(p)
                corpus_duration.append(d1)
                corpus_velocity.append(velocity)

                # not needed, increases at music goes
                #offset_list.append(element.offset)

            ###########################################
            # Chord
            ###########################################   
            elif isinstance(element,chord.Chord):   
                # dMaj = chord.Chord(['D', 'F#', 'A'])
                # dMaj = chord.Chord(['D3', 'F#4', 'A5'])
                # element <music21.chord.Chord C2 G2 E-3>

                # chord have a duration <music21.duration.Duration 1.0>
                # chord have no pitch pitches 
                # element.pitches (<music21.pitch.Pitch C2>, <music21.pitch.Pitch G2>, <music21.pitch.Pitch E-3>)
                # Pitch (upper P), part of chords, have no duration or volume <music21.pitch.Pitch C2>
                # Pitch have name, namewithoctave
                # chord have a volume <music21.volume.Volume realized=0.79>

                """
                C C#  D   D#  E  F  F#  G   G#  A  A#  B
                0 1   2   3   4  5   6   7   8  9  10  11

                l = [str(n) for n in element.normalOrder] #  E4 G4  ['4,'7'] '4.7'
                # decode with note.Note(4)  return note E (default octave 4)
                """
                
                 # use either normal order as in example , or C4.D4
                 
                if config_bach.normal: # represented as integer, ie 1.2.  forget octave. default is octave 4. 
                    p=[]
                    for x in element.normalOrder: # x is int representation of PITCHE. 
                        p.append(str(x))
                    p = ".".join(p) # convert to string with dot  
                    # if E4 E5 normal is [4] and no dot ?  ok later in split('.')
                         
                else:     # keep octave F5.A4.D5.D4.D3. generates much larger dictionary 3200 vs 310   E4 E5 generated [4] vs [E4,E5]
                    p=[]
                    # storing octave
                    for x in  element.pitches: 
                        # type(x) <class 'music21.pitch.Pitch'>
                        # tuple of pitch (<music21.pitch.Pitch C2>, <music21.pitch.Pitch G2>, <music21.pitch.Pitch E-3>)
                        p.append (x.nameWithOctave)                
                    p = ".".join(p) # convert to string with dot

                    # some cleanup keep # - G5.E-5.C3  bemol
                    p=p.replace('is','') #A2.A3.Cis4'
                    p=p.replace('s','')
                   
                    
                d= element.duration.quarterLength
                if not isinstance(d,float): # class fraction  1/3
                    d1 = float(d.numerator) / float(d.denominator)   
                else:
                    d1 = float(d)
                  
                d1 = min(d1,max_du)
                d1= '%0.1f' %(d1) # round to 2 decimal and convert to srt
                d1= '$' + d1  # to distingish from normal C4 C5 which is one integer
                # cannot use D. a label D something could be a note or a duration

                # concatenate piches and duration. $ separator
                if config_bach.concatenate :
                    p = p + d1
            
                velocity = element.volume.velocity

                corpus_notes.append(p)
                corpus_duration.append(d1)
                corpus_velocity.append(velocity)
                
             
            ###########################################
            # Rest
            ###########################################      
            elif isinstance(element,note.Rest):
                d = element.duration.quarterLength
                d1 = float(d)   
                d1 = min(d1,max_du)
                d1= '%0.1f' %(d1) # round to 2 decimal and convert to srt
                d1= '$' + d1  # to distingish from normal C4 C5 which is one integer
                #print('rest ', d1)
                p='R' # means rest

                # concatenate piches and duration. $ separator
                if config_bach.concatenate :
                    p = p + d1

                velocity = 0 # rest has no velocity

                corpus_notes.append(p) # encode rest
                corpus_duration.append(d1)
                corpus_velocity.append(velocity)
                
            else:
                pass
                
                """
                <class 'music21.meter.TimeSignature'>
                <class 'music21.key.Key'>
                <class 'music21.tempo.MetronomeMark'>
                instruments, parts, 
                """

    # FOR FILE

    print ("ALL MIDI file parsed. corpus_notes, corpus_duration, corpus_velocity created. STRINGS\n") 

    corpus_len = len(corpus_notes)     
    print('pitches in corpus_notes: %d, duration in corpus_duration: %d' %(corpus_len, len(corpus_duration)))
    # should be the same
    assert (corpus_len == len(corpus_duration))
    assert (corpus_len == len(corpus_notes))
    assert (corpus_len == len(corpus_velocity))

    print('corpus example, ie ALL elements')
    print('notes ', corpus_notes[:10])
    print('duration ', corpus_duration[:10])
    print('velocity ', corpus_velocity[:10])
    
    # unique pitches and duration
    unique_pi_list = sorted(list(set(corpus_notes))) 
    unique_du_list = sorted(list(set(corpus_duration)))  
    unique_ve_list = sorted(list(set(corpus_velocity)))  

    unique_pi_size = len(unique_pi_list)
    unique_du_size = len(unique_du_list)
    unique_ve_size = len(unique_ve_list)

    print('\nunique pitches %d , unique duration %d, unique velocity %d' %(unique_pi_size, unique_du_size, unique_ve_size))

    # use dict to convert each unique pitches or duration to INTEGER
    pi_to_int = dict((p,n) for n,p in enumerate(unique_pi_list))
    du_to_int = dict((p,n) for n,p in enumerate(unique_du_list))
    ve_to_int = dict((p,n) for n,p in enumerate(unique_ve_list))

    #pi_to_int dict {'0': 0, '0.1': 1, '0.1.2.4.6.8.9': 2, '0.1.3': 3, '0.1.3.4.6.8': 4, '0.1.3.4.6.8.9': 5
    #du_to_int dict {'X0.1': 0, 'X0.2': 1, 'X0.3': 2, 'X0.4': 3, 'X0.5': 4, 'X0.7': 5, 'X0.8': 6, 'X1.0': 7, 'X1.2': 8, 

    # convenience naming dict one way integer to string
    int_to_pi = unique_pi_list
    int_to_du = unique_du_list
    int_to_ve = unique_ve_list

    print('example "dictionary", int_to_pi , aka unique_pi_list (list): ' , int_to_pi[:5])
    print('example "dictionay", int_to_du  aka unique_du_list (list): ' , int_to_du[:5])
    print('example "dictionay", int_to_ve  aka unique_ve_list (list): ' , int_to_ve[:5])

    # get a list from dict
    pi_to_int_items = pi_to_int.items()
    du_to_int_items = du_to_int.items()
    ve_to_int_items = ve_to_int.items()

    print('example dictionary, pi_to_int (list from dict) ', list(pi_to_int_items)[:5])
    print('example dictionary, du_to_int (list from dict) ', list(du_to_int_items)[:5])
    print('example dictionary, ve_to_int (list from dict) ', list(ve_to_int_items)[:5])

    unique_pi_size_after_low_removal = unique_pi_size # will be updated later after removing low occurence
    
    """
    # OPTIMIZATION: smaller softmax by not trying to predict low occurence. 
    # see the least used pitches
    exclude = config_bach.low_occurence # is less than occurences
    
    pi_to_exclude = []
    for i in unique_pi_list: # strings
        print ("occurence %s %d" %(i, most[i]))
        if most[i] < exclude:
            pi_to_exclude.append(i)         
    print('low occurence list: ', pi_to_exclude)
    print ('total unique pi from corpus %d, ignored for low occurence %d' %(corpus_pi_size, len(pi_to_exclude)))
    
    unique_pi_size_after_low_removal = unique_pi_size - len(pi_to_exclude)
    # in that config, softmax should be new unique_pi_size
    # but dictionary stay the same. we will see low occurence in X
    # here we just created pi_to_exclude
    """ 

    print("create input list and output list for TRAINING. list of seqlen of INTEGERS for X and list of one INTEGER for Y")  
    

    ####################################################
    # network input and output list. to vectorized later
    # type 1:
    #   input: list of list of seqlen integer index (either pitches, our pitches$duration)
    #   ouput list of integer index 

    # will be deleted after training

    # vectorization:
    #   type 1:
    #       input: integer index converted to one hot
    #       output: integer index converted to one hot
    # 
    #####################################################

    # model 2: manage THREE input, output list separatly for pitches and duration

    # INTEGERS
    training_input_list_int_pi=[] # to be used for X. n samples  
    training_output_list_int_pi=[] # Y next notes

    # for model 2 , we need a similar structure
    training_input_list_int_du=[] # to be used for X. n samples  
    training_output_list_int_du=[] # Y next notes

    training_input_list_int_ve=[] # to be used for X. n samples  
    training_output_list_int_ve=[] # Y next notes
    
    # BUG a=b=[]. Y.append also append in X 
    
    ign = 0 # counter of low occurence Y ignored
    print('LSTM sequence lenth %d '% (config_bach.seqlen))

    print('do not exclude low occurence')
    pi_to_exclude = [] # stubb do not use this optimization


    #############################################
    # create training data
    #############################################
    for i in tqdm(range(0,corpus_len-config_bach.seqlen,1)):  # large number, all sequence of seq len in corpus
        
        # handle one seqlen. 

        # X . model 1 only uses one_seq_notes. model 2 heads uses both
        # strings
        one_seq_notes = corpus_notes[i:i+config_bach.seqlen] # one sequence of pitches or pitches$duration
        one_seq_duration = corpus_duration[i:i+config_bach.seqlen]
        one_seq_velocity = corpus_velocity[i:i+config_bach.seqlen] # one sequence of $duration

        # Y str
        target_pi = corpus_notes[i+config_bach.seqlen]
        target_du = corpus_duration[i+config_bach.seqlen]
        target_ve = corpus_velocity[i+config_bach.seqlen]
        
        if target_pi in pi_to_exclude: # hardcode to null if not used
            print ('excluding low occurence %s' %(target_pi))
            ign = ign + 1
            # do not include this seqlen > low occurence Y 

        else:

            #################################### 
            # convert strings to integer indexes 
            ####################################

            seqlen_of_pitches_int = [] # list of seqlen integers, one sequence
            seqlen_of_duration_int = [] # list of seqlen integers, one sequence
            seqlen_of_velocity_int = []

            for pattern in one_seq_notes:
                # convert to int             
                p = pi_to_int[pattern]        
                seqlen_of_pitches_int.append(p) 

            for pattern in one_seq_duration:
                # convert to int             
                p = du_to_int[pattern]        
                seqlen_of_duration_int.append(p)

            for pattern in one_seq_velocity:
                # convert to int             
                p = ve_to_int[pattern]        
                seqlen_of_velocity_int.append(p) 
            
            # manage TWO input, output list separatly for pitches and duration

            training_input_list_int_pi.append(seqlen_of_pitches_int) # 
            training_input_list_int_du.append(seqlen_of_duration_int)
            training_input_list_int_ve.append(seqlen_of_velocity_int) #   
        
            target_pi  = corpus_notes[i+config_bach.seqlen]
            target_du  = corpus_duration[i+config_bach.seqlen]
            target_ve  = corpus_velocity[i+config_bach.seqlen]

            # convert to int as well for consistency with X, use in embeding
            training_output_list_int_pi.append(pi_to_int[target_pi])
            training_output_list_int_du.append(du_to_int[target_du])
            training_output_list_int_ve.append(ve_to_int[target_ve])
        
    # next sequence of seqlen in corpus . ie one sample
       
    print ('\nprocessed all sequences of %d len in corpus. %d sequences, ignored %d on low occurence ' %(config_bach.seqlen, i+1, ign))
    
    pabou.inspect('training_input_list_int ' , training_input_list_int_pi) # training_input_list_int  is a LIST. len 32447, type <class 'list'>
    pabou.inspect('training_output_list_int ' , training_output_list_int_pi) # training_output_list_int  is a LIST. len 32447, type <class 'int'>

    # model 2 training_input_list_int[0] len 2
    #[[109, 128, 128, 51, 19, 3, 19, 51, 19, ...], [2, 40, 40, 2, 2, 2, 2, 2, 2, ...]]

    # training_ouput_list_int 
    #[[45, 2], [81, 2], [45, 2], [81, 2], [109, 2], [89, 2], [45, 2],

    pabou.inspect('training_input_list_int_pi[0]' , training_input_list_int_pi[0]) # training_input_list_int_pi[0] is a LIST. len 40, type <class 'int'>
    pabou.inspect('training_output_list_int_pi[0] ' , training_output_list_int_pi[0]) # training_output_list_int_pi[0] :  type <class 'int'>

    # various checks
    assert (len(training_input_list_int_pi) == (len(corpus_notes) - config_bach.seqlen - ign) )
    assert (len(training_input_list_int_du) == (len(corpus_notes) - config_bach.seqlen - ign) )
    assert (len(training_input_list_int_ve) == (len(corpus_notes) - config_bach.seqlen - ign) )
    
    #training_output_list_int is not related to seqlen

    assert (len(training_input_list_int_pi[0]) == config_bach.seqlen)
    assert (len(training_input_list_int_du[0]) == config_bach.seqlen)
    assert (len(training_input_list_int_ve[0]) == config_bach.seqlen)
       
    for x in training_output_list_int_pi:
        assert x not in pi_to_exclude
   
    hot_size = unique_pi_size_after_low_removal
    print('pitches hot size, ie softmax size' , hot_size)
    assert hot_size + ign == unique_pi_size

    print('duration hot size, ie softmax size' , len(unique_du_list))
    print('velocity hot size, ie softmax size' , len(unique_ve_list))
    print ("PARSE DONE.  all MIDI files analyzed. created corpus, dictionaries and training_input_list_int\n") 

    # list for du, ve will be ignored for model 1
    # list are of integer index
    return (training_input_list_int_pi, training_output_list_int_pi, \
        training_input_list_int_du, training_output_list_int_du,\
        training_input_list_int_ve, training_output_list_int_ve,\
        corpus_notes, corpus_duration, corpus_velocity, \
        unique_pi_list, pi_to_int, \
        unique_du_list, du_to_int, \
        unique_ve_list, ve_to_int, \
        )
# end parse


#################################################################################################### 
# VECTORIZE ie convert list of int to np array
# create x_train x_val x_test , y_ttrain, y_val, y_test 
# and same for 2 and 3 (duration and velocity)
# ###################################################################################################

"""
model 1:
X
  seqlen of one hot
  seqlen of one hot
  ....

array dim (batch, seqlen, hot size)

Y
   One hot
   ...

array dim (batch, hot size)

when converted to one hot, 1 become hot sizes, for either pitches or pitches$duration

model 3  X   array of seqlen of int (because embedding)
Y array of one hot

model 2:
X
 [seqlen of int]  [seqlen of int ]  [seqlen of int ] 1st list is pitches, 2nd list is duration. 3rd velocity . do not convert to one hot. embedding expect integer indexes
 [seqlen of int]  [seqlen of int ]  [seqlen of int ]
 ...

Y
 one hot one hot one hot
 ....
 
"""
    
def vectorize \
(training_input_list_int_pi, training_output_list_int_pi, \
        training_input_list_int_du, training_output_list_int_du,\
        training_input_list_int_ve, training_output_list_int_ve,\
        model_type) :

    print('\n!!!! vectorize input and output integer list. model_type: %d ' %(model_type)) 
    
    # input list: integer index. could convert to float (bad) or one hot (better) or left as it for embedding
    # output list: integer, to be one hot encoded
    
    # or do shuffle in fit
    # WARNING: look like training does not work if we not do that
    print('shuffle ALL network lists of Int')
    (training_input_list_int_pi, training_output_list_int_pi, training_input_list_int_du, training_output_list_int_du, training_input_list_int_ve, training_output_list_int_ve, ) = \
    shuffle(training_input_list_int_pi, training_output_list_int_pi, training_input_list_int_du, training_output_list_int_du, training_input_list_int_ve, training_output_list_int_ve ,\
        random_state=0)


    print("vectorize INPUT list into np array")  # list, array of integer indexes 

    # training, validation, test split. already shuffled
    split_val = int(nb_sample*0.7)
    split_test = int(nb_sample*0.9)

    # vectorize all, even if only x,y used for model 1

    ###################################
    # X list of list of integers index 
    # X pitches, X1 duration, X2 velocity
    # convert in np array 
    # input tensor for embedding layer defined as .... default, ie float
    ################################### 
    # input data is array like, list, tuples. type infered  

    # model 1. int16 converted to float32 when converted to one hot

    #NOTE !!!!!!!!!!!!!!!!! MAKE SURE the dtype is consistent with model input definition 
    # or could vertorize after creating model, and cast to model input dtype

    X = np.asarray(training_input_list_int_pi, dtype = 'int32') # int32 by default

    # delete object as soon as not needed to save ram
    del(training_input_list_int_pi)

    # use one hot vs index. index may imply ranking
    pabou.inspect("INSPECT: X (pitch)", X) # dtype int16 min 0 max 94

    X1 = np.asarray(training_input_list_int_du, dtype = 'int32')
    del(training_input_list_int_du)
    pabou.inspect("INSPECT: X1 (duration)", X1) 

    X2 = np.asarray(training_input_list_int_ve, dtype = 'int32')
    del(training_input_list_int_ve)
    pabou.inspect("INSPECT: X2 (velocity)", X2) 

    
    #################################################
    # Y, int to one hot. for both model 1, model 2 and model 3
    #################################################   
    Y = np.asarray(training_output_list_int_pi) # int32 by default
    del(training_output_list_int_pi)  
    Y = tf.keras.utils.to_categorical(Y) # convert int to one hot    float32 by default
    #num_classes	total number of classes. If None, this would be inferred as the (largest number in y) + 1.
   
    pabou.inspect("INSPECT: Y (one hot)", Y) # nb_sample, nb_unique_notes   float32 min 0 max 1 
    
    Y1 = np.asarray(training_output_list_int_du)
    del(training_output_list_int_du)  
    Y1 = tf.keras.utils.to_categorical(Y1) # convert int to one hot    float32
    pabou.inspect("INSPECT: Y1 (one hot)", Y1)

    Y2 = np.asarray(training_output_list_int_ve)
    del(training_output_list_int_ve)  
    Y2 = tf.keras.utils.to_categorical(Y2) # convert int to one hot    float32
    pabou.inspect("INSPECT: Y2 (one hot)", Y2)


    ##################### Model 1 ####################################
    # only pitches, ie X and Y are used
    if model_type == 1: 
        print('model 1, convert X into one hot ', X.shape, X.dtype, X.ndim)
        #########################################################
        # danger zone . WTF
        # call to inspect freeze with tf 2.4.1
        ###########################################################
        # convert X , ie pitch into one hot. integer representation ,  do not used normalize, ie between 0 and 1 as this may imply ranking
        
        # int16 index converted to float hot    (32447, 40) int16 2  TO (32447, 40, 95) float32 
        X = tf.keras.utils.to_categorical(X)
        print('model 1, converted X into one hot ', X.shape, X.dtype, X.ndim)

        #pabou.inspect("INSPECT: X converted to one hot", X)  # WTF freeze mouse

        feature_dim = X.shape[2]
        print('model 1: feature dim, X_shape[2], aka one hot size: ', feature_dim)

        """
        # add batch dimension. feature dim was one for index encoding
        # that does not seem to change anything to X , still the same shape
        # not needed for training, does not change anything, as batch dimension is already there.
        # needed for inference. add batch = 1
        X=np.reshape(X, (len(training_input_list_int) , config_bach.seqlen, feature_dim) ) # shape (xxx,50,dim)
        pabou.inspect('INSPECT: X after reshape to batch dimension: ', X)
        """ 

     ##################### Model 3 ####################################
    if model_type == 3: 
        pass # already integer, ready for embedding 
        # vectorized int list to ???? int32, int16 , float32. 

    if model_type == 4: # X (32387, 100) dtype('int32')
        # normalize X Integer list
        print('model 4 , normalize X')
        print('X mean ' , X.mean())
        print("X std ", X.std())

        config_bach.model4_mean = X.mean()
        config_bach.model4_std = X.std()

        X = X - X.mean(axis=0)
        X = X / X.std(axis=0)
        X = X.astype('float32') # was float64
        
        pabou.inspect("INSPECT: X (normalized)", X) 
        

    ##################### Model 2 ####################################
    # X is already integer index. can used in embedding layer
    if model_type == 2:
        pass # already integer, ready for embedding

    #####################################################################
    # SPLIT in training, validation and test
    #####################################################################

    # Use np.split to chop our data into three parts.
    # The second argument to np.split is an array of indices where the data will be
    # split. We provide two indices, so the data will be divided into three chunks.

    # pitches
    x_train, x_val, x_test = np.split(X, [split_val, split_test])
    del(X)
    y_train, y_val, y_test = np.split(Y, [split_val, split_test])
    del(Y)

    # duration
    x1_train, x1_val, x1_test = np.split(X1, [split_val, split_test])
    del(X1)
    y1_train, y1_val, y1_test = np.split(Y1, [split_val, split_test])
    del(Y1)

    # velocity
    x2_train, x2_val, x2_test = np.split(X2, [split_val, split_test])
    del(X2)
    y2_train, y2_val, y2_test = np.split(Y2, [split_val, split_test])
    del(Y2)


    # Double check that our splits add up correctly
    assert (x1_train.shape[0] + x1_val.shape[0] + x1_test.shape[0]) ==  nb_sample
    assert (y1_train.shape[0] + y1_val.shape[0] + y1_test.shape[0]) ==  nb_sample

    assert (x2_train.shape[0] + x2_val.shape[0] + x2_test.shape[0]) ==  nb_sample
    assert (y2_train.shape[0] + y2_val.shape[0] + y2_test.shape[0]) ==  nb_sample

    assert (x_train.shape[0] + x_val.shape[0] + x_test.shape[0]) ==  nb_sample
    assert (y_train.shape[0] + y_val.shape[0] + y_test.shape[0]) ==  nb_sample

    print("SPLIT done: train %d, validate %d, test %d, out of %d" %(x_train.shape[0], x_val.shape[0], x_test.shape[0], nb_sample))
    """
    # freeze again
    pabou.inspect('INSPECT: x_train (pitch): ', x_train)
    pabou.inspect('INSPECT: y_train (pitch): ', y_train)
    """

    print("X and Y Vectorize done for train, val and test, and for picth, duration and velocity\n")
    return (x_train, y_train, x_val, y_val, x_test, y_test, \
        x1_train, y1_train, x1_val, y1_val, x1_test, y1_test , \
        x2_train, y2_train, x2_val, y2_val, x2_test, y2_test )
    
    # for model 2 training data is a list, ie [x_train, x1_train] [y_train, y1_train]

# end of vectorize 

"""
#https://ai.stackexchange.com/questions/3156/how-to-select-number-of-hidden-layers-and-number-of-memory-cells-in-lstm
# upper bound on number of hidden neuron = number of sample in training / (number of input neuron + number of output neuron) * 2 to 10
#440000 input 100, output 300    440000  / 400 * 2   500 to 200

256 input neurons 300 output neurons    600   600x2  to 600x10   1200 to 6000
number of samples in training set 340k
340k/1200   340k/6000   280 to 56  upper bound to number of hidden neuron that won't result in over-fitting
256 LSTM (dense is not hidden)
"""

###############################################################################################################  
# FIT . 
# test set not used there. 
# which will call keras model.fit .
#  will print some stats
################################################################################################################  

def fit(model, x_train, y_train, x_val, y_val, x1_train, y1_train, x1_val, y1_val, x2_train, y2_train, x2_val, y2_val, checkpoint_path):

    # x1, y1 : duration. only for model 2
    # x2, y2 : velocity. only for model 2

    # x and y : pitch. model 1,3,4
    
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print ('FIT: checkpoint_dir: ', checkpoint_dir)
    print ('FIT: checkpoint_path: ', checkpoint_path) # a file filepath can contain named formatting options, which will be filled the value of epoch and keys in logs (passed in on_epoch_end).


    """
    early stopping, reduce LR on plateau need val_loss
    model checkpoint need val_acc
    
    Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy
    #Reduce LR on plateau conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy,lr
    #WARNING:tensorflow:Can save best model only with val_acc available, skipping.
    #WARNING:tensorflow:Method (on_train_batch_end) is slow compared to the batch update (0.217596). Check your callbacks.
    increase batch size. 
    """
    
    """
    # pb with tensorboard in colab. and can see previous run to see progress
    files = glob.glob(config_bach.tensorboard_log_dir+'/*')
    print('removing previous tensorboard log file %s' %(files))
    try:
        for f in files:
            os.remove(f)
    except Exception as e:
        print ('exception removing tensorboard logs file ' + str(e))
    """
    if config_bach.model_type in[1,3,4]:
            what_to_monitor = 'val_accuracy'
    if config_bach.model_type == 2:
        what_to_monitor = 'val_pitch_output_accuracy'

    print('define call back list. monitoring: ' , what_to_monitor)

    callbacks_list = [
            
            # stop when val_accuracy not longer improving, no longer means 1e-2 for 4 epochs. can use baseline
            tf.keras.callbacks.EarlyStopping(monitor=what_to_monitor, mode= 'auto' , min_delta=1e-2, restore_best_weights=True, patience=8, verbose=1),
            
            # factor: factor by which the learning rate will be reduced. new_lr = lr * factor
            # min_lr: lower bound on the learning rate.
            tf.keras.callbacks.ReduceLROnPlateau(monitor=what_to_monitor,factor=0.1,patience=8, min_lr=0.001),
            
            # checkpoint dir created if needed
            # can save full model
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor=what_to_monitor, mode = 'auto', save_best_only=True, verbose=1, save_weights_only = True, save_freq = 'epoch', period=1),
            #Epoch 00001: val_loss improved from inf to 0.25330, saving model to mymodel_1.h5
            #stores the weights to a collection of checkpoint-formatted files that contain only the trained weights in a binary format. Checkpoints contain: * One or more shards that contain your model's weights. * An index file that indicates which weights are stored in a which shard.
            #If you are only training a model on a single machine, you'll have one shard with the suffix: .data-00000-of-00001

            
            # histogram frequency in epoch. at which to compute activation and weight histograms for the layers
            # write_images: whether to write model weights to visualize as image in TensorBoard.
            # update_freq: 'batch' or 'epoch' or integer. When using 'batch', writes the losses and metrics to TensorBoard after each batch.
            # write_graph: whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True.
            tf.keras.callbacks.TensorBoard(log_dir=config_bach.tensorboard_log_dir , update_freq='epoch', histogram_freq=1, write_graph=False, write_images = True)

            ]
    
    # xtrain 337911 batch 256  => 1319,9 steps tensorboard batch accucay displays steps. epoch accuracy displays every 337921

    # patience:  delay to the trigger in terms of the number of epochs on which we would like to see no improvement
    # baseline: only stop training if performance stays above or below a given threshold or baseline
    # min_delta: By default, any change in the performance measure, no matter how fractional, will be considered an improvement. You may want to consider an improvement that is a specific increment, such as 1 unit for mean squared error or 1% for accuracy.
    # mode By default, mode is set to ‘auto‘ and knows that you want to minimize loss or maximize accuracy.
    
    # tensorboard --logdir=./tensor_log
    # on windows tensorboard --logdir ./nameofdir
    #TensorBoard 1.13.1 at http://omen:6006  //localhost:6006

    print('FIT: callback list: ' , callbacks_list)
    print('FIT: Tensorboard log dir: %s. should be the same in Colab' %config_bach.tensorboard_log_dir)
    
    # if command line, use that
    if parsed_args['epochs']:
        epochs = parsed_args['epochs']
        print('FIT: fit. epochs %d. from command line' %(epochs))
    else:
        epochs = config_bach.epochs
        print('FIT: fit. epochs %d. from config file' %(epochs))
    

    #x	Vector, matrix, or array of training data (or list if the model has multiple inputs)
    #y	Vector, matrix, or array of target (label) data (or list if the model has multiple outputs)

    ###############################################################################
    # call to KERAS model fit
    ###############################################################################

    if config_bach.model_type in [1,3]:
        print('\n--------------------  FIT for model 1,3.  %d epochs. %s ------------------ '%(epochs,config_bach.app))
        start_time = time.time()   
        # one head
        history = model.fit(x= x_train, y = y_train,\
             batch_size=config_bach.batch,epochs=epochs, callbacks=callbacks_list, \
             validation_data = (x_val,y_val), shuffle = False, verbose =0)
        elapse = time.time() - start_time

    if config_bach.model_type == 4:
        print('\n--------------------  FIT for model 4.  %d epochs. %s ------------------ '%(epochs,config_bach.app))
        # reshape seqlen = size*size into a sizexsize matrix ie (22670, 100) to (22670, 10,10)
        x_train = np.reshape(x_train, (x_train.shape[0], config_bach.size, config_bach.size))
        x_val = np.reshape(x_val, (x_val.shape[0], config_bach.size, config_bach.size))

        start_time = time.time()   
        history = model.fit(x= x_train, y = y_train,\
             batch_size=config_bach.batch,epochs=epochs, callbacks=callbacks_list, \
             validation_data = (x_val,y_val), shuffle = False, verbose =0)
        elapse = time.time() - start_time

    if config_bach.model_type == 2:
        # multi head, multi input. 
        print('\n------------------------  FIT for model 2. %d epochs. %s ----------------------- ' %(epochs,config_bach.app))
        
        # predict velocity or not . pass 2 or 3 array
        # DEPEND ON how the model was created. it will expect a list of 2 or 3 inputs
        if config_bach.predict_velocity:

            print('will predict velocity')
            x = [x_train, x1_train, x2_train]
            y = [y_train, y1_train, y2_train]
            x_val = [x_val, x1_val, x2_val]
            y_val = [y_val, y1_val, y2_val]

        else:

            print('will NOT predict velocity')
            x = [x_train, x1_train]
            y = [y_train, y1_train]
            x_val = [x_val, x1_val]
            y_val = [y_val, y1_val]

        # NOTE: the code is generic, but must use the rigth model

        start_time = time.time()  
        history = model.fit(x= x,  y = y, \
            batch_size=config_bach.batch, epochs=epochs, callbacks=callbacks_list ,\
            validation_data = (x_val , y_val), \
            shuffle = True, verbose = 0)
        elapse = time.time() - start_time

    ########################## KERAS model.fit done ########################################

    print ('FIT: fit ended. took: %d sec' %(elapse))

    # epoch could be less because of early stop. also best epoch could be prior last epoch
    actual_epoch = len(history.history['loss'])
    
    print("FIT: model fitted in %d sec for %d configured epochs , and %d actual epochs" % (elapse, epochs, actual_epoch))
    print('FIT: per epoch sec: %0.2f' %(float(elapse/ actual_epoch)))
    
    print('FIT: metrics available in history after fit: ' , end = ' ')
    for key in history.history:
        print(key, end = ', ')
    print('\n')
    
    # on colab metrics available in history after fit loss acc val_loss val_acc lr.  acc vs accuracy
    
    # latest checkpoint
    try:
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print ('FIT: latest checkpoint when done fitting:', latest)
    except:
        print('error no latest checkpoint')
    
    # save history in case further processing
    history_dict = history.history # list of successive metrics
    print('FIT: serialize training history to pickle file: ' , history_file)
    with open(history_file, 'wb') as fp:
        pickle.dump(history_dict, fp)

    """
    to load
    with open (history_file, 'rb') as fp:
        history_dict = pickle.load(fp)
        print('predict. load training history from: ', history_file) # not used
    """

    print('FIT: DONE.. return history\n')
    return(history)  # saved model done in main

# end of fit  

#####################################
# predict forever THREAD.  created by flask handler for /play
# get 3 seed str as input parameter
#  convert str to int
#  predict using model full or tflite
#  use temperature
#  convert prediction to pcm audio sample
#  write pcm audio sample to queue
# rollover str
#####################################
stop_predict_thread = False # set to True in calling thread 

def predict_forever_thread(seed_pitch_str, seed_duration_str, seed_velocity_str, queue_samps):
    print('predict forever thread running')
    # corpus is global 
    global stop_predict_thread
    global lite, model, interpreter

    """
        this code used to be inline in flask
        made a thread to parallelize, and make sure flask gets timely audio sample
        loop 'forever' and push predicted audio samples in a queue
        on the other side of the queue, flask thread gets audio sample and yield them
        can be started as soon as the seed is defined, so start prediction and fill queue in parallel of playing seed 
    """

    ################################################
    # initialize input_list_*_str with seed
    ################################################
    input_list_pitch_str = seed_pitch_str # will be updated after each predict
    input_list_duration_str = seed_duration_str
    input_list_velocity_str = seed_velocity_str

    nb=0 # of inferences
    inference_time = 0.0
    concatenate = False # boolean to flag is we concanenated, and at the end roll seqlen of pi$du str

    ######################################################
    # predict 'forever' , ie 2GB
    ######################################################

    # one prediction and samps generated at each loop

    for i in range (config_bach.predict_forever):

        # check global variable at each prediction loop. 
        if stop_predict_thread:
            print('===== > predict forever thread is told to stop')
            stop_predict_thread = False
            sys.exit(0)

        # set at any time using GUI
        instrument = my_remi.instrument # to change instrument on the fly
        
        # input_list_*_str are initialized with seed, and rolled over at the end of this loop

        ###########################################################
        # create X,X1,X2 integer lists from input_list_*_str 
        # convert str to int 
        # X are ONE input, a seqlen
        ###########################################################
        X = [] # list of seqlen integers
        for pattern in input_list_pitch_str:               
            p = pi_to_int[pattern] # convert string to INT for network input, use dictionary            
            X.append(p)

        X1 = [] # not used for model 1
        for pattern in input_list_duration_str:               
            p = du_to_int[pattern] # convert string to INT for network input, use dictionary            
            X1.append(p)

        X2 = [] # not used for model 1
        for pattern in input_list_velocity_str:               
            v = ve_to_int[pattern] # convert string to INT for network input, use dictionary            
            X2.append(v)

        # list of int
        path_to_tflite_file = os.path.join('models', my_remi.tfmodel, '.tflite')

        # inference timing
        start_time = time.time()

        ############################################
        # MODEL 1 or 3
        # vectorize X and convert to one hot and add batch dimension
        # cast
        # predict FULL and LITE
        ############################################
        if config_bach.model_type in [1,3,4]:
    
            X = np.asarray(X) # array of int
            # cast to expected input
            # model return tf type, convert to numpy
            X = X.astype(model.inputs[0].dtype.as_numpy_dtype)
            if config_bach == 1:
                X = tf.keras.utils.to_categorical(X, num_classes=len(pi_to_int)) # array of one hot    float32
                # add batch dimension. feature dim is one for index encoding , and hot size for one hot
                X=np.reshape(X, (1 , config_bach.seqlen, len(pi_to_int)))  # shape (xxx,50,dim)
            else:
                pass # embedding

            # lite global, set in handler for /play
            if lite == False: # TFfull
                # number of input need to be adapted to model, here 1
                # X np array. one SEQLEN of one hot
                prediction = model.predict(X,verbose=0)  # verbose 1  1/1 [==============================] - 0s 38ms/step
                softmax_pi = prediction[0] # prediction[0] is an array of softmax , ie hotsize float32, nothing else in prediction
            else:  # TFlite
                ###############################################
                # !!!!!!!!!!!!!  create lite interpreter
                ################################################
                # pass 3 X  to make procedure generic , for model 1, only X is used
                # unused are []
                (softmax_pi, softmax_du, softmax_ve, inference_ms) = pabou.TFlite_single_inference(X,[],[], interpreter, config_bach.model_type)

            # accumulate total inference time
            inference_time = inference_time + time.time() - start_time
            nb = nb + 1 # to show inference time from time to time
            
            
            # modify proba, higher temp, increase surprise.
            # dynamic temperature using remi
            softmax_pi = pabou.get_temperature_pred(softmax_pi, my_remi.temperature)
        
            index = np.argmax(softmax_pi) # just take highest proba in softmax float32 np.sum(prediction) = 1.000001
            target_pi = unique_pi_list[index] # index from softmax converted to str label
            # target_pi is str prediction
            
            # need to set duration
            x = target_pi

            if '$' in x: # duration is encoded there
                concatenate = True # to roll over with pi$du

                duration = x.split('$')[1]
                duration.replace('$','')
                duration_in_quarter=float(duration)

                target_pi = x.split('$')[0] # str
                
            else: # random or fixed duration. fixed look better for model 1
                concatenate = False # to roll over with pi only

                # choices for default duration
                #duration = random.choice(common_duration)[0]
                #duration = duration.replace('$', '')
                #duration_in_quarter = float(duration)

                duration_in_quarter = float(config_bach.d1)  # float

            # since model 1 does not predict velocity, use default

            velocity = config_bach.velocity

            ###############################################################################
            # generates samps, ie PCM
            # predicted pitch, duration in quarter is set
            # velocity is from config file (model 1)
            # d2 is from config all the time
            ###############################################################################
            samps = my_audio.get_audio_sample_from_one_pattern(target_pi, duration_in_quarter, velocity, fluid, instrument, sfid)
            
            # ROLLOVER
            # rolling is done on input list str , not X directly , either int list or array 
            if concatenate: 
                input_list_pitch_str.append(target_pi+'$'+str(duration_in_quarter))
                input_list_pitch_str = input_list_pitch_str[1:]
            else:
                input_list_pitch_str.append(target_pi)
                input_list_pitch_str = input_list_pitch_str[1:]

        # end model 1  

        ############################################
        # MODEL 2
        ############################################

        if config_bach.model_type == 2: ################# STREAM PREDICTION FULL MODEL 2 
            if my_remi.tfmodel == 'Full': # predict using FULL model
                
                X = np.array(X)
                X = X.astype(model.inputs[0].dtype.as_numpy_dtype)
                X1 = np.array(X1)
                X1 = X1.astype(model.inputs[1].dtype.as_numpy_dtype)
                X2 = np.array(X2)
                X2 = X2.astype(model.inputs[2].dtype.as_numpy_dtype)

                if config_bach.predict_velocity:
                    # number of input need to be adapted to model
                    (softmax_pi, softmax_du, softmax_ve) = model.predict([X,X1,X2],verbose=1) # verbose 1  1/1 [==============================] - 0s 38ms/step
                else:
                    (softmax_pi, softmax_du) = model.predict([X,X1],verbose=1) # verbose 1  1/1 [==============================] - 0s 38ms/step

            else:  ################# STREAM PREDICTION LITE MODEL 2 

            # pass all X (int list) to make procedure generic 
            # X are int list.

                ###############################################
                # !!!!!!!!!!!!!  create lite interpreter
                ################################################

                if config_bach.predict_velocity:
                    (softmax_pi, softmax_du, softmax_ve, inference_ms)  = pabou.TFlite_single_inference(X,X1,X2,interpreter, config_bach.model_type)
                else:
                    (softmax_pi, softmax_du, softmax_ve, inference_ms)  = pabou.TFlite_single_inference(X,X1,[],interpreter, config_bach.model_type)

            inference_time = inference_time + time.time() - start_time # to compute average
            nb = nb + 1

            # array of one array of float32. size = hot_size prediction is a numpy array (1, 32) ndim 2 len 1 
            #print('prediction is a numpy array %s ndim %d len %d ' %(prediction.shape, prediction.ndim,  len(prediction)))
            
            """
            # GDL
            # take first softmax
            prediction_pitch = prediction_picth[0] 
            prediction_duration = prediction_duration[0] # do not expect duration to be encoded in pitch

            if config_bach.predict_velocity:
                prediction_velocity = prediction_velocity[0]
            """

            # modify proba, higher temp, increase surprise.
            # dynamic temperature using remi
            softmax_pi = pabou.get_temperature_pred(softmax_pi, my_remi.temperature)
            softmax_du = pabou.get_temperature_pred(softmax_du, my_remi.temperature)
            if config_bach.predict_velocity:
                softmax_ve = pabou.get_temperature_pred(softmax_ve, my_remi.temperature)

            # convert softmax to predicted 

            # argmax of softmax, converted to string
            index = np.argmax(softmax_pi) # just take highest proba in softmax float32 np.sum(prediction) = 1.000001
            target_pi = unique_pi_list[index] # index from softmax converted to str label
            # target_pi is str prediction

            index = np.argmax(softmax_du) # just take highest proba in softmax float32 np.sum(prediction) = 1.000001
            target_du = unique_du_list[index] # index from softmax converted to str label
            target_du1 = target_du.replace('$', '')
            duration_in_quarter = float(target_du1)
            # float

            if config_bach.predict_velocity:
                index = np.argmax(softmax_ve)
                target_ve = unique_ve_list[index]
            else:
                target_ve = None
            # integer or None
                                
            #############################################################################
            # generate samps 
            # d2 is from config all the time
            # pass target_pi(str), duration in quarter(float), target_ve (int or None)
            #############################################################################
            samps = my_audio.get_audio_sample_from_one_pattern(target_pi, duration_in_quarter, target_ve, fluid, instrument, sfid)
            
            # roll over is done on input list str, not X directly (messy)
            input_list_pitch_str.append(target_pi)
            input_list_pitch_str = input_list_pitch_str[1:]
           
            input_list_duration_str.append(target_du)
            input_list_duration_str = input_list_duration_str[1:]

            if config_bach.predict_velocity:
                input_list_velocity_str.append(target_ve)
                input_list_velocity_str = input_list_velocity_str[1:]
        # end model 2
        
        # write audio sample to queue (vs just yielding if not in a separate thread
        # will play in browser
        queue_samps.put(samps, block=True)
        #print('p', end = ' ')

        # print average inference time every n 
        if nb % 100 == 0:
            average = 1000.0 * inference_time / float(nb)
            print('%s average inference time so far ms: %0.2f ' % (my_remi.tfmodel, average))

    # for forever
    print('end of predict forever thread')
    
# end of predict forever thread



#########################################
# Flask handler
#########################################

# need to create server before route
from flask import Flask, request , render_template, make_response, Response
#from werkzeug import secure_filename
#import flask_monitoringdashboard as dashboard # only if not colab

server=Flask(__name__)  # create server before defining route 

#############################
## test. simple static text string
#############################
@server.route("/", methods=['POST', 'GET'])
def handler_root():
        print ("\nflask handler for /")
        logging.info(str(datetime.datetime.now())+ ' Flask: /' )
        resp = "use /ping, /audio or /play"
        r=make_response(resp)
        r.headers['Content-Type'] = 'text/plain'
        return (r) 

#############################
## test. simple static text string
#############################
@server.route("/ping", methods=['POST', 'GET'])
def handler_test():
        print ("\nflask handler, simple text string")
        logging.info(str(datetime.datetime.now())+ ' loopback test' )
        resp = "/ping. play bach est la"
        r=make_response(resp)
        r.headers['Content-Type'] = 'text/plain'
        return (r) 
 
#############################################
## test play notes locally on server using pyaudio
# instrument is updated as string in my_remi.instrument
#############################################
@server.route("/audio", methods=['POST', 'GET'])
def handler_local():    
    print ("\nflask handler. play audio locally on server using instrument ", my_remi.instrument)
    logging.info(str(datetime.datetime.now())+ ' local play' )
    resp = "/audio: playing few notes locally on server using fluid and pyaudio: " + my_remi.instrument
    r=make_response(resp)
    r.headers['Content-Type'] = 'text/plain'
    
    for note in ['a4','b4','c4','d4','a5','b5','c5','d5', 'e6', 'f6', 'e7', 'f7']:
        # get audio sample from one note as a string
        # play locally via pyaudio. returns text
        samps = my_audio.get_audio_sample_from_one_note(fluid, note, config_bach.d1, config_bach.d2,my_remi.instrument,sfid)
        # play with pyaudio
        pa_strm.write(samps)  
    return (r) 


#############################################
## FLASK is STREAMING there, ie yielding pcm to browser
# handler for /play
# queue already created when Flask instanciated
# init: load corpus 
# play() : 
#   play a4 and wave header
#   get random seed, convert to pcm and yield all notes to browser
#   create predict thread, with seed and queue as parameter  
#   get pcm audio (prediction) from queue and yield to browser
#############################################
@server.route("/play", methods=['POST', 'GET'])
def handler_play():  

    print('\nflask handler for /play') 
    # in case of corpus change in GUI, need to restart play url, to get there and load new corpus.
    # stopping predict thread. not sure if needed.
     
    global stop_predict_thread
    global model
    global t
    global lite

    # from pickle, corpus, unique and to_int needed in predict thread

    global training_input_list_int_pi, training_output_list_int_pi, \
           training_input_list_int_du, training_output_list_int_du , \
           training_input_list_int_ve, training_output_list_int_ve , \
          corpus_notes, corpus_duration, corpus_velocity, \
            unique_pi_list, pi_to_int, unique_du_list, du_to_int, unique_ve_list, ve_to_int
    
    ###############################################
    # load new h5 model and new corpus.pick if not the current one
    # based on variable set in GUI
    ###############################################

    #######################################################
    # stop any predict thread that may be already running.
    ########################################################
    
    try:
        if t.is_alive(): # exception is normal (t not defined) if thread not already started
            stop_predict_thread = True # tested by predict thread
            print ('asking predict thread to stop, as corpus has changed')
            t.join(5.0)  # wait until it stops
            print ('join predict thread has stopped') # but still some audio in browser pipeline. 
    except Exception as e:
        print('Exception predict thread is alive ', str(e))

    stop_predict_thread = False # so that the next one will run

    ###############################################
    # load new model and new corpus.pick if not the current one
    # can be Full or Lite model
    # based on variable set in GUI
    # set global variable lite
    ###############################################
    if my_remi.load_new_corpus : # set by GUI if corpus actually changed
        print('changed corpus to: %s. need to load new corpus pick file and  model' %my_remi.corpus)

        # interpreter, model and lite are available in predict thread
        
        # check if new corpus is full or lite
        if my_remi.corpus.split('_')[-1] == 'lite':
            interpreteur = pabou.load_model(my_remi.corpus ,'h5', None, None) 
            print('this is a TFLite model')
            lite = True
        else:
            model = pabou.load_model(my_remi.corpus ,'li', None, checkpoint_path) 
            print('this is a h5 Full model')
            lite = False

        # load corpus as well. needed for seed, dictionaries etc ..
        x = my_remi.corpus + '_' + config_bach.pick
        pick = os.path.join(path, x)
        with open(pick, "rb") as fp:

            (training_input_list_int_pi, training_output_list_int_pi, \
            training_input_list_int_du, training_output_list_int_du,\
            training_input_list_int_ve, training_output_list_int_ve,\
            corpus_notes, corpus_duration, corpus_velocity, \
            unique_pi_list, pi_to_int, \
            unique_du_list, du_to_int, \
            unique_ve_list, ve_to_int, \
            ) = pickle.load(fp)
        print("loaded new corpus from saved pickle file: " , pick)


        # use last char of corpus to encode model 1 or 2
        print('loaded model type: ' , end = ' ')
        if my_remi.corpus[-1] == '2':
            config_bach.model_type = 2 # needed for X for inference
            print ('2')
        else:
            config_bach.model_type = 1
            print ('1')
    # new corpus
    else:
        print('/play start with existing model and corpus: ' , my_remi.corpus)

    my_remi.load_new_corpus = False # 


    #########################
    # create seed , play header and seed
    # start predict forever thread
    # while TRUE get audio sample from queue and yield to browser
    #########################
    
    def play():
        print('\nstart to stream using: %s\n' %my_remi.instrument)
        logging.info(str(datetime.datetime.now())+ ' starting to stream' )
        global t

        # queue already created
        # SEED. get a ramdom list of pattern from corpus . could start at measure
        start = random.randint(0,len(corpus_notes)-config_bach.seqlen)
        
        print ('seed starts at random %d within %d' %(start, len(corpus_notes)))

        # create random seeds
        seed_pitch = corpus_notes[start:start+config_bach.seqlen] # list of notes/chords C4 , 1.2 or C4 C6
        seed_duration = corpus_duration[start:start+config_bach.seqlen] 
        seed_velocity = corpus_velocity[start:start+config_bach.seqlen] 

        #print('seed pitch: ', seed_pitch, len(seed_pitch), type(seed_pitch[0]))
        #print('seed duration: ', seed_pitch, len(seed_duration), type(seed_duration[0]))
        #print('seed velocity: ', seed_velocity, len(seed_velocity), type(seed_velocity[0]))

        # play (yield) single a4 and wave header
        
        print('play wav header and a single note a4')
        # prefix with wave header
        wav_header = my_audio.genHeader(config_bach.sampleRate, config_bach.bitsPerSample, config_bach.channels)

        # send header and one first note , just for fun
        samps = wav_header + my_audio.get_audio_sample_from_one_note(fluid, 'a4', config_bach.d1, config_bach.d2,my_remi.instrument, sfid)
        yield(samps)
        time.sleep(1)

        
        # play (yield) all notes in seed 
        # use duration encoded in pitch , or from duration corpus (should be the same)
        # use velocity from corpus
        
        for idx, pitch in enumerate(seed_pitch):

            if '$' in pitch: # duration is encoded there
                x = pitch.split('$')[0] 
                duration = pitch.split('$')[1]  
                duration = duration.replace('$','')
                pitch = x
                
            else:
                # leave pitch alone, get actual duration from corpus
                duration = seed_duration[idx]
                duration = duration.replace('$','')
                
            duration = float(duration)  # was str before
            
            # seed velocity
            velocity = seed_velocity[idx] # int

            # convert note or chord string to audio sample
            # sfid is global.  to set instrument
            # instrument is a string
            
            # convert one note/chord from seed to pcm audio sample
            # yield pcm to browser for play as wave
            
            # d2 is from config all the time
            # pass target_pi(str), duration in quarter(float), target_ve (int)

            samps = my_audio.get_audio_sample_from_one_pattern(pitch, duration, velocity, fluid, my_remi.instrument, sfid)
            yield(samps)
            
            time.sleep(0.05)
            
        # seed music have been yielded to brower
        print('all seed music have been yieled to browser')

        ###################################
        # start predict forever thread
        # use HL threading
        ###################################
        # predict thread will generate audio sample from running inferences and write to queue
        # for some reason, starting the thread before yielding the seed create blocage. 
        # a lot of pg pg , but sometime ppppp gggg pp gg

        # start predict forever thread as soon as possible. will start filling the queue while playing seed 
        # us e HL threading instead of low level thread
        # 
        try:
            #_thread.start_new_thread(predict_forever_thread, (seed_pitch, seed_duration, seed_velocity, queue_samps) ) # need tuple
            t = threading.Thread(target=predict_forever_thread, args=(seed_pitch, seed_duration, seed_velocity, queue_samps))
            t.start()
            print('start predict thread. Alive ? ' , t.is_alive())
            logging.info(str(datetime.datetime.now())+ ' starting predict forever thread' )
            
        except Exception as e:
            print ("exception in starting predict forever thread " , str(e))
            logging.error(str(datetime.datetime.now())+ ' error starting predict forever thread: %s' %(str(e)) )
            exit(1)
        
        ################################################
        # get audio sample from queue from predict thread
        # predict thread is the one which generates samps with get audio sample from one pattern
        # send (yield) audio sample to browser
        ################################################

        print('keep getting audio sample from queue, until /play is invoqued again')
        while (True):
            samps = queue_samps.get(block=True) # get prediction
            #print('g', end = ' ')
            yield(samps) # play to browser
            time.sleep(0.001)
        print('!!!!!!!!!!!!!!!! should never get there')
    # def play():

    return Response(play(), mimetype='audio/x-wav')
# def Flask handler for /play


#############################################################################
#############################################################################
##  MAIN
#############################################################################
#############################################################################

print('\n\n====================== Play Bach %s. %s ==========================\n\n' %(version, config_bach.app))

print('corpus: ', config_bach.app)
print('model type: ', config_bach.model_type)
print('working directory: ' , os.getcwd() )
os.environ["PYTHONIOENCODING"] = "utf8"
print('default encoding', sys.getdefaultencoding())
print('file system encoding', sys.getfilesystemencoding())

# parse CLI arg. see .vscode/launch.json for vscode
print('model type (hot, heads..) %d' %(config_bach.model_type))
parsed_args = pabou.parse_arg(config_bach.model_type) # return namespace converted to dict


"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # TF logging. disable AUTOGRAPH warning when saving as SavedModel 
# before importing tf
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""

# prevent spitting too much info ?
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# prevent spitting too much info ?
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# disable GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# env variable set in .vscode/launch.json
try:
    print('author from env variable ', os.environ['author'])
except:
    pass  # env variable not defined in colab

# 2.3 does not work on my compute capability. 2.4 is needed for RNN TFlite
assert float(tf.__version__[:3]) >= 2.3

# debug, info, warning, error, critical
log_file = config_bach.log_file
print ("logging to:  " , log_file)
logging.basicConfig(filename=log_file,level=logging.INFO)
logging.info(str(datetime.datetime.now())+ ' ---- play bach starting ----' )


# print various info on installation
(os_name, plat, machi) = pabou.print_tf_info()

"""
for subdir, dirs, files in os.walk('./'):
    for file in files:
        print (file)
"""

# myhost = os.uname()[1] does not work on windows
# checkpoint is always a serie of file in a dir. update at the end of epoch or so many epoch or nb of samples
# For example: if filepath is weights.{epoch:02d}-{val_loss:.2f}.hdf5, then the model checkpoints will be saved with the epoch number and the validation loss in the filename.
# checkpoint_path = "training_1/cp.ckpt" This creates a single collection of TensorFlow checkpoint files that are updated at the end of each epoch:
# content of directory : checkpoint           cp.ckpt.data-00001-of-00002 cp.ckpt.data-00000-of-00002  cp.ckpt.index

if sys.platform in ["win32"]:
    print("running on WINDOWS on system: " , platform.node())
    colab=False # proceed from training to real time audio
 
elif sys.platform in ['linux']:
    print("running on Colab on system: ", platform.node())
    # os: posix, sys.platform: linux, platform.machine: x86_64
    colab = True # to stop at the end of training, and not go in audio, remi, flask 

else: # PI
    print ('where I am running ?')
    exit(1)

# vscode start in music21


########################
# various run time file names
########################

root = os.getcwd()
print("starting dir: ", root)
checkpoint_path = os.path.join( root, 'checkpoint', 'cp-{epoch:03d}.ckpt')
print('save checkpoint (formatted) with checkpoint path: %s' %(checkpoint_path))

bach_files = os.path.join(root,config_bach.bach_files)
print('bach MIDI sources files: ', bach_files)

# prefix everything with apps/corpus. pickle for corpus needed at inference time
# note in saved model , pass app = corpus , to create cello_    h5. _ managed in pabou save/load
# at training time, model and pickle are prefixed with config_bach.app and _.
# only at inference time corpus from remi can be used to reload new model/pickle
# make sure corpus pickle is set  music21 > music21/models/cello_<file>

path = os.path.join(root,  'models' ) 

# pickle for corpus needed at inference time
x = config_bach.app + '_' + config_bach.pick
pick = os.path.join(path, x)

x = config_bach.app + '_' + config_bach.plot_file
plot_file = os.path.join(path, x)

x = config_bach.app + '_' + config_bach.history_file
history_file = os.path.join(path, x) 

x = config_bach.app + '_' +  config_bach.training_file
training_file = os.path.join(path, x)

x = config_bach.app + config_bach.midi_file
midi_file = os.path.join(path, x)

x = config_bach.app + pabou.label_txt 
label_file = os.path.join(path, x)

# test set already vectorized. needed for edgetpu benchmark
x = config_bach.app + '_' + config_bach.pick_x_y
pick_x_y = os.path.join(path, x)


###########################################
# LOAD parsed corpus from PICKLE, if pickle file exists . 
# or parse and create pickle
###########################################

if os.path.isfile(pick):
    
    # load pickle
    print("Pickle file exists. load existing corpus from saved pickle file: " , pick)
    with open(pick, "rb") as fp:

        (training_input_list_int_pi, training_output_list_int_pi, \
        training_input_list_int_du, training_output_list_int_du,\
        training_input_list_int_ve, training_output_list_int_ve,\
        corpus_notes, corpus_duration, corpus_velocity, \
        unique_pi_list, pi_to_int, \
        unique_du_list, du_to_int, \
        unique_ve_list, ve_to_int, \
        ) = pickle.load(fp)

else:
    ######################
    # no pickle
    # PARSE  , SAVE pickl and create label file
    ######################

    print("no saved pickle. parse MIDI files from directory %s" %(bach_files))
    # only the corpus is stored as str. training list are int and later vectorized
        
    (training_input_list_int_pi, training_output_list_int_pi, \
    training_input_list_int_du, training_output_list_int_du,\
    training_input_list_int_ve, training_output_list_int_ve,\
    corpus_notes, corpus_duration, corpus_velocity, \
    unique_pi_list, pi_to_int, \
    unique_du_list, du_to_int, \
    unique_ve_list, ve_to_int, \
    )  =  parse(bach_files)

    # parse print few stats on corpus

    """
    unique_pi_list , aka int_to_pi
    ['A2', 'A2.F#3', 'A2.F3', 'A3', 'A3.B3', 'A3.F#3', 'A4', 'A4.G#4', 'B-2', 'B-2.E3', 'B-2.G#3', 'B-2.G3', 'B-3', 'B-3.D3', ...]
    pi_to_int 
    {'A2': 0, 'A2.F#3': 1, 'A2.F3': 2, 'A3': 3, 'A3.B3': 4, 'A3.F#3': 5, 'A4': 6, 'A4.G#4': 7, 'B-2': 8, 'B-2.E3': 9, 'B-2.G#3': 10, 'B-2.G3': 11, 'B-3': 12, 'B-3.D3': 13, ...}
    """

    # save pickle
    with open(pick, "wb") as fp:
        print('save corpus in pickle: ', pick)

        pickle.dump ( \
        (training_input_list_int_pi, training_output_list_int_pi, \
        training_input_list_int_du, training_output_list_int_du,\
        training_input_list_int_ve, training_output_list_int_ve,\
        corpus_notes, corpus_duration, corpus_velocity, \
        unique_pi_list, pi_to_int, \
        unique_du_list, du_to_int, \
        unique_ve_list, ve_to_int, \
        ) , fp)


        
###################################
# labels for tflite models metadata
###################################
d_pi = {}
d_du = {}
d_ve = {}

for i , pi in enumerate(unique_pi_list):
        d_pi[i] = pi

for i , du in enumerate(unique_du_list):
        d_du[i] = du

for i , ve in enumerate(unique_ve_list):
        d_ve[i] = ve

l = [d_pi, d_du, d_ve]

with open(label_file, "w") as fp:   # if wb need bytes , if w need str write() argument must be str, not list
    fp.write(str(l))
print('label for metadata', l) 
#  [{0: 'A2', 1: 'A2.F#3', 2: 'A2.F3', , 127: 'G5', 128: 'R'}, {0: '$0.0',
#  38: '$7.7', 39: '$7.8', 40: '$8.0'}, {0: 0, 1: 1, 2: 70, 3: 85, 4: 100}]

############################################################
# create corpus and dictionary .cc file, for microcontroler
############################################################
print('create cc corpus file for microcontroler')
corpus_int = [] # convert corpus from str to int
for x in corpus_notes:
    corpus_int.append(pi_to_int[x])  # convert str to int

# create C array SOURCE file. meant to be compiled 
# .h file, containing actual variable definition (not only declaration)
# include .h file in C source. no need for any .cpp file
pabou.create_corpus_cc(config_bach.app, corpus_int)
pabou.create_dict_cc(config_bach.app, unique_pi_list)

print('parse done. pickle created or loaded. corpus, training integer list and dictionary available')
print('created .h file for micro')


###################################################
# look at corpus
# either from parse or from load pickle
###################################################
# corpus is string
print('\n')
print('corpus, ie all notes, after parse or pickle load')
print('notes ', corpus_notes[:10])
print('duration ', corpus_duration[:10])
print('velocity ', corpus_velocity[:10])

c = Counter(corpus_notes) # type counter most['R'] is 41388
print(tabulate(c.most_common(10), headers =["top n pitches"]))
print(tabulate(c.most_common()[:-10-1:-1], headers =["least n pitches"]))

c = Counter(corpus_duration) 
print(tabulate(c.most_common(10), headers =["top n durations"]))
print(tabulate(c.most_common()[:-10-1:-1], headers =["least n duration"]))

# define most common duration
common_duration = c.most_common(8) # if too large, random will jump over the place
print ('common duration ', common_duration)
# common duration  [('X0.2', 14069), ('X0.5', 10750), ('X1.0', 2907), ('X0.0', 1290), ('X0.3', 714)]

c = Counter(corpus_velocity) 
print(tabulate(c.most_common(10), headers =["top n velocity"]))
print(tabulate(c.most_common()[:-10-1:-1], headers =["least n  velocity"]))


nb_sample = len(training_input_list_int_pi) # all seqlen

assert nb_sample == len(training_output_list_int_pi)
assert nb_sample == len(training_output_list_int_du)
assert nb_sample == len(training_input_list_int_du)
assert nb_sample == len(training_output_list_int_ve)
assert nb_sample == len(training_input_list_int_ve)

print ('corpus size, ie number of notes, chords, ..: ', len(corpus_notes))
print ('nb seqlen sample for vectorization: ' , nb_sample)  # corpus - seqlen
print ('seqlen: ' , config_bach.seqlen)

hot_size_pi = len(unique_pi_list)
hot_size_du = len(unique_du_list)
hot_size_ve = len(unique_ve_list)
print('softmax size pi:', hot_size_pi)
print('softmax size du:', hot_size_du)
print('softmax size ve:', hot_size_ve)
    
pabou.inspect('INSPECT: training_input_list_int_pi, ie all samples, for vectorization ', training_input_list_int_pi)
pabou.inspect('INSPECT: training_output_list_int_pi, for vectorization ', training_output_list_int_pi)

pabou.inspect('INSPECT: training_input_list_int_pi[0], ie one seqlen,  for vectorization ', training_input_list_int_pi[0])
pabou.inspect('INSPECT: training_output_list_int_pi[0], for vectorization ', training_output_list_int_pi[0])


########################################################
# vectorize in any case (ie not only if fit). 
# we need some x_train for TFlite representative generator
# also need test set to evaluate a loaded lodel

# creates train, val , test for x and y , for 0,1,2
# vectorize pitch, duration and velocity in all case. depending on the model, some may not be used
# from network input list and network output list as integer

# to predict MIDI file. vectorization of seed done with prediction
# #######################################################

print('vectorize integer input list into train, validate, test , 0, 1 ,2 ') 

# x is pitch, x1 duration, x2 velocity
(x_train, y_train, x_val, y_val, x_test, y_test, \
x1_train, y1_train, x1_val, y1_val, x1_test, y1_test , \
x2_train, y2_train, x2_val, y2_val, x2_test, y2_test ) = vectorize (training_input_list_int_pi, training_output_list_int_pi, \
training_input_list_int_du, training_output_list_int_du,\
training_input_list_int_ve, training_output_list_int_ve,\
config_bach.model_type )
print('vectorized done. train, val, test, x,y, 1,2,3 created')

# coral benchmark runs standalone, so get a ready vectorized test set
with open (pick_x_y, 'wb') as fp:
    print('create pickle %s with X test and Y test. for coral benchmark' %pick_x_y)
    pickle.dump((x_test, y_test , x1_test, y1_test, x2_test, y2_test),fp)


############# create or load Model ##########################
# if l=1 load existing h5 model (could tf, h5 or checkpoints). can be used to resume training
# if l=0 force create new model from parsed corpus

# fit needs either any empty model or a saved one
# predict need a saved model
###############################################################

if parsed_args['load_model'] == 1: # INTEGER.     load saved model. used in prodiction
    
    ########################################################################################
    # LOAD EXISTING h5 model and evaluate model with test set 
    # in remi: cello, filename: cello_ ,  prefix done in pabou
    # corpus in remi is initialized with config_bach, then dynamic
    ########################################################################################

    print('load existing h5 model. could also load tf model')

    import my_remi # need to import. assume we use colab only for training, so colab should not see that
    print('model to load :', my_remi.corpus)

    model = pabou.load_model(my_remi.corpus ,'h5', None, checkpoint_path)
    if model is None:
        print ('failed loading h5 model. exit')
        sys.exit(1)

    # evaluate model.   vectorization already done, so x_test is defined
    if config_bach.model_type in [1,3]:
        x= x_test
        y = y_test

    if config_bach.model_type == 2 and config_bach.predict_velocity == True:
        x = [x_test, x1_test, x2_test]
        y = [y_test, y1_test, y2_test]

    if config_bach.model_type == 2 and config_bach.predict_velocity == False:
        x = [x_test, x1_test]
        y = [y_test, y1_test]

    if config_bach.model_type in [4]: # (3239, 100) to (3239, 10, 10)
        x = np.reshape(x_test,(x_test.shape[0],config_bach.size, config_bach.size)) # keep batch dim, split 36 in 6x6
        y = y_test

    # evaluate model just loaded from disk. good info
    (test_pitch_accuracy, test_duration_accuracy, test_velocity_accuracy ) = pabou.model_evaluate(model, config_bach.model_type, x , y)
    # pretty table is done in this routine
    
    """
    # load tf model
    model = pabou.load_model(my_remi.corpus,'tf', None, checkpoint_path)
    if model is None:
        print ('failed loading tf model. exit')
        sys.exit(1)
    
    (empty, nb_params) = create_model_1(hot_size)
    model = pabou.load_model('bach', 'cp', empty, checkpoint_path)
    if model is None:
        print ('failed loading cp model. exit')
        sys.exit(1)
    """
    
    print("full h5 model loaded and evaluated: parameters: %d, metrics %s " % (model.count_params(), model.metrics_names))

else:
    
    #########################################################################################
    # CREATE new MODEL 
    # could be model 1 (1 head), or model 2 (2 or 3 heads) or model 3
    ##########################################################################################
    print('create model. model_type: %s' %  (config_bach.model_type))

    # single output head
    if config_bach.model_type == 1:
        # hot_size is size of last dense 
        (model, nb_params) = my_model.create_model_1(hot_size_pi, plot_file)

    elif config_bach.model_type == 3:
        # hot_size is size of last dense 
        # last boolean is True for smaller models for microcontrollers
        (model, nb_params) = my_model.create_model_3(hot_size_pi, plot_file, False)

    elif config_bach.model_type == 4:
        # hot_size is size of last dense 
        (model, nb_params) = my_model.create_model_4(hot_size_pi, plot_file)

    # multi heads
    elif config_bach.model_type == 2:
        if config_bach.predict_velocity:
            # with velocity
            (model, nb_params) = my_model.create_model_2(hot_size_pi, hot_size_du, hot_size_ve, plot_file)
        else:
            # no velocity
            (model, nb_params) = my_model.create_model_2_nv(hot_size_pi, hot_size_du, plot_file)
    else:
        print('model type %d not supported' %config_bach.model_type)
        
   

# created or loaded model

#############################  
# MODEL FIT  
# model keras object must already exists
# if existing model was loaded, this means we continue training
############################

if parsed_args['fit'] == False: 
    print('DO NOT FIT, SAVE and EVALUATE.')

else:
    print('will fit and evaluate (to start or resume training)')
    
    hot_size_pi = y_train.shape[1]
    print('hot size pitches, ie size of softmax is: ' , hot_size_pi)

    hot_size_du = y1_train.shape[1]
    print('hot size duration, ie size of softmax is: ' , hot_size_du)

    hot_size_ve = y2_train.shape[1]
    print('hot size velocity, ie size of softmax is: ' , hot_size_ve)
    
    ###############################
    # MODEL FIT 
    ###############################
    print('fit model. checkpoint_path: %s'  %(checkpoint_path))

    #x	Vector, matrix, or array of training data (or list if the model has multiple inputs)
    #y	Vector, matrix, or array of target (label) data (or list if the model has multiple outputs)
    # x1, y1 passed , but not used for model 1

    # my fit routine. no need to pass test set. x1 , y1 will not be used in model 1. x2, y2 only if we use velocity
    history = fit ( model, \
        x_train, y_train, x_val, y_val,  \
        x1_train, y1_train, x1_val, y1_val, \
        x2_train, y2_train, x2_val, y2_val, \
        checkpoint_path)      
    

    ####################################################################
    # SAVE model in tf, h5 format. save checkpoints as 999. save json model. checkpoint_path is managed in main
    #####################################################################
        
    print('Save FULL model in various format (tf, h5, weigths from checkpoint, json)')
    # save is done using app as in config bach

    (tf_dir, h5_file, h5_size) = pabou.save_full_model(config_bach.app, model, checkpoint_path)
    
    
    ###################################################
    # MODEL evaluate on test data
    ####################################################
    
    if config_bach.model_type in [1,3]:
        x = x_test
        y = y_test

    if config_bach.model_type in [4]: # (3239, 100) to (3239, 10, 10)
        x = np.reshape(x_test,(x_test.shape[0],config_bach.size, config_bach.size)) # keep batch dim, split 36 in 6x6
        y = y_test

    if config_bach.model_type == 2 and config_bach.predict_velocity:
        x = [x_test, x1_test, x2_test]
        y = [y_test, y1_test, y2_test]

    if config_bach.model_type == 2 and config_bach.predict_velocity == False:
        x = [x_test, x1_test]
        y = [y_test, y1_test]
    
    # pretty table is done in this routine. some value may be zero, depending on model
    test_pitch_accuracy, test_duration_accuracy, test_velocity_accuracy  = pabou.model_evaluate(model, config_bach.model_type, x , y)

    # see training plots
    pabou.see_training_result(history, [test_pitch_accuracy, test_duration_accuracy, test_velocity_accuracy], training_file, config_bach.model_type)
    
    """
    #NEED MORE WORK
    # accuracy from any dataset
    print('model accuracy from any dataset')
    # pass array 
    pabou.see_model_accuracy(model, x, y)
    """

# end of fit , save and evaluate   


#############################
# MIDI file creation
# only using Full Model
# TFlite model only used for streaming
#############################
if parsed_args['predict_midi'] == True:
    import my_midi
    print('run prediction ON FULL MODEL to create MIDI files')
    
    # model already loaded. 
    # to predict only use -l=1 to force load saved model. -nf to skip fit

    
    # PREDICT MIDI LIST , create list of str
    # for model 1, only first list is not None
    (list_of_predicted_pitch, list_of_predicted_duration, list_of_predicted_velocity)  =  \
        my_midi.create_predicted_MIDI_list  (model, config_bach.nb_predict, \
        corpus_notes, corpus_duration, corpus_velocity, pi_to_int, du_to_int, ve_to_int, unique_pi_list, unique_du_list, unique_ve_list)

    # those list is seed (aka real music) and all predictions
    pabou.inspect ("predicted list of pitches:  ", list_of_predicted_pitch)
    pabou.inspect ("predicted list of duration:  ", list_of_predicted_duration)
    pabou.inspect ("predicted list of velocity:  ", list_of_predicted_velocity)

    # CREATE MIDI file from list of strings
    # duration, offset is set there
    print ("create output midi file %s from predicted list of strings" %(midi_file))
    my_midi.create_MIDI_file(list_of_predicted_pitch, list_of_predicted_duration, list_of_predicted_velocity, midi_file)
else:
    print('NO PREDICTION AND MIDI FILE CREATION')


####################################################################
# MODEL CONVERT in TFlite format
# name of model file are defined in PABOU , both full and tflite
#####################################################################
if parsed_args['savelite'] == True: # default false

    # https://www.tensorflow.org/lite/guide/ops_select 
    # https://www.tensorflow.org/lite/convert/rnn 

    """
        Some of the operators in the model are not supported by the standard TensorFlow Lite runtime. 
        If those are native TensorFlow operators, you might be able to use the extended runtime by 
        passing --enable_select_tf_ops, or by setting target_ops=TFLITE_BUILTINS,SELECT_TF_OPS when calling tf.lite.TFLiteConverter(). 
        Otherwise, if you have a custom implementation for them you can disable this error with --allow_custom_ops, or by setting allow_custom_ops=True 
        when calling tf.lite.TFLiteConverter(). Here is a list of builtin operators you are using: 
        ADD, FILL, FULLY_CONNECTED, LEAKY_RELU, MUL, PACK, RESHAPE, SHAPE, SOFTMAX, STRIDED_SLICE, TRANSPOSE. 
        Here is a list of operators for which you will need custom implementations: 
        TensorListFromTensor, TensorListReserve, TensorListStack, While.
    """

    if float(tf.__version__[:3]) >= 2.3:
        # need tf 2.3 for RNN but tf2.3 does not work on compute capability 5, ie my omen. so move totf2.4 
        print('\n\nconvert to TFlite. ALL quantization. From keras model, or h5 or SavedModel.') 


        # prepare representative dataset. use test set 
        
        if config_bach.model_type in [1,3,4]:
            x = x_test
        if config_bach.model_type == 2 and config_bach.predict_velocity == True:
            x = [x_test, x1_test, x2_test]
        if config_bach.model_type == 2 and config_bach.predict_velocity == False:
            x = [x_test, x1_test]
        
        try:
            pabou.save_all_tflite_models(config_bach.app, x, config_bach.model_type, hot_size_pi, model) 

            # x is for representative dataset gen. uses test set. input as numpy array
            # will generate various quantization
            # metadata only implemented for type 1 for now 
            # hot size is for metadata only. size of output softmax
            # model to get input structure 
            
        except Exception as e:
            print('Exception saving TFlite: ', str(e))
            sys.exit(1)
    else:
        print ('tf version %s. cannot save to TFlite for the RNN models with tf < 2.3' %tf.__version__)

else:
    print('NO TFLITE MODEL CONVERSION')
    

#########################################################################
# benchmarks full model
#########################################################################
if parsed_args['benchmark_full'] == True: # default false
    nb = 100
    print('doing benchmark for full model on %d inference' %(nb))
    
    # use the instanciated keras model
    """
    # load model from file
    print ('load TF full model from %s for inferences' %(full_model_SavedModel_dir))
    model = tf.keras.models.load_model(full_model_SavedModel_dir) # dir

    # load h5 full model. 3rd argument empty model if needed, 4th checkpoint path
    model = pabou.load_model('mnist_','h5', None, None)
    """

    # h5_size is set in fit, so not set if we do not fit, but loaded a model
    h5_size = pabou.get_h5_size(config_bach.app)

    if config_bach.model_type in [1,3,4]:
        pabou.bench_full("FULL model", model, x_test, y_test, [], [], [], [] , nb, h5_size , config_bach.model_type)
    else:
        if config_bach.predict_velocity:
            pabou.bench_full("FULL model2 with velocity", model, x_test, y_test, x1_test, y1_test, x2_test, y2_test, nb, h5_size, config_bach.model_type )
        else:
            pabou.bench_full("FULL model2 without velocity", model, x_test, y_test, x1_test, y1_test, [], [], nb, h5_size , config_bach.model_type)

else:
    print('NO BENCHMARK ON FULL MODEL')


###############################################
# benchmark TFlite for ALL quantization
# use test set . test set depend on model type
###############################################
if parsed_args['benchmark_lite'] == True: # default false
    nb = 100
    h5_size = pabou.get_h5_size(config_bach.app)
    print('doing benchmark for all TFLite models with %d inference' %(nb))
 
    pt = PrettyTable()
    pt.field_names = ["model", "iters", "infer (ms)",  "acc pi", "acc du", "acc ve",  "h5 size", 'TFlite size']

    for x in pabou.all_tflite_file:
        # create full path for TFlite file
        # append _fp32_lite.tflite
        x = config_bach.app +  x
        tflite_file = os.path.join('models', x)
        print('doing TFlite benchmark for: ', tflite_file)

        # False means do not use Coral run time, but tensorflow run time
        
        try:
            if config_bach.model_type in [1,3,4]:
                b= lite_inference.bench_lite_one_model("TFlite model 1,3,4: ", tflite_file ,x_test, y_test, [], [], [], [], nb,  config_bach.model_type, config_bach.app, h5_size, False)
            else:
                if config_bach.predict_velocity:
                    b= lite_inference.bench_lite_one_model("TFlite model 2, velocity: ", tflite_file ,x_test, y_test, x1_test, y1_test, x2_test, y2_test, nb,  config_bach.model_type, config_bach.app, h5_size, False)
                else:
                    b= lite_inference.bench_lite_one_model("TFlite model 2, no velocity: ", tflite_file ,x_test, y_test, x1_test, y1_test, [], [],  nb, config_bach.model_type, config_bach.app, h5_size, False)
           
            pt.add_row(b) # 
        except Exception as e:
            print('exception TFlite benchmark %s. model file: %s' %(str(e), tflite_file))

    print(pt)
else:
    print('NO BENCHMARK ON LITE MODELS')

#################################################################################
#################################################################################
# STREAMING 
# only include audio , flask and remi from there. so not executed when running on colab 
# ################################################################################
##################################################################################
# do no run audio streaming on colab
if colab:
    print('running on colab, so do not stream. exit')
    sys.exit(0)

if parsed_args['stream'] == False:
    print('stream not configured. exit')
    sys.exit(0)

print('import my audio')
# pyaudio and fluidsynth imported in my_audio
import my_audio
    
print('import remi gui' , end = '. ')
import my_remi
print('instrument %s , temperature %d' %(my_remi.instrument, my_remi.temperature))

import flask_monitoringdashboard as dashboard

sound_font = config_bach.sound_font
print ("init pyaudio and fluid synth " , sound_font)
# pyaudio stream that can be written
(pa_strm, fluid, sfid)=my_audio.init_audio(sound_font)
# pyaudio stream, fluid .. are global

##########################################################
### FLASK and REMI
##########################################################

# get local IP. only works on linux; add -I
try:
    import subprocess
    address = subprocess.check_output(['hostname']) # returns bytes
    address = address.decode('utf-8') 
    address=address[:-1] # nl at the end
    print('local IP: ', address)
except Exception as e:
    pass

with open('remi.json' ,'r') as fp:
	data = json.load(fp)
print('remi json file: ',data)

flask_port = (data['flask_port'])
remi_port = (data['remi_port'])
print('flask port %d , remi port %d' %(flask_port, remi_port))

############################## 
# Remi thread 
##############################
def remi_thread(remi_port): # get tuple
    my_remi.start_remi_app(remi_port)

try:
    print('\n==== starting REMI thread on ', remi_port)
    _thread.start_new_thread(remi_thread, (remi_port,) ) # need tuple
    logging.info(str(datetime.datetime.now())+ ' starting remi thread on port %d' %remi_port)
except Exception as e:
    print ("exception in starting remi thread " , str(e))
    logging.error(str(datetime.datetime.now())+ ' error starting remi thread: %s' %(str(e)) )
    exit(1)


############################## 
# FLASK server in main  thread
##############################
"""
https://medium.com/better-programming/how-to-use-flask-wtforms-faab71d5a034
https://medium.com/flask-monitoringdashboard-turtorial/monitor-your-flask-web-application-automatically-with-flask-monitoring-dashboard-d8990676ce83
"""

# Flask dashboard
print('start flask dashboard. connect to /dashboard to get statistics')
dashboard.config.init_from(file=config_bach.dashboard_config)
dashboard.bind(server)

print('create queue between main (flask) and predict thread') # multi process #q1=Queue() . using multi thread instead
queue_samps= queue.Queue(maxsize=10) # main predict and write. flask hander read and renders

print ("\n==== starting flask server in main on port: " , flask_port) 
logging.info(str(datetime.datetime.now())+ ' starting flask on port %d' % flask_port )

# 0.0.0.0 seen from outside local host
try:
    server.run(host="0.0.0.0", port=flask_port, debug=False)
except Exception as e:
    print('error starting flask ', str(e))
    logging.error(str(datetime.datetime.now())+ ' error starting flask: %s' %(str(e)) )
    exit(1)



 


