
import music21
from music21 import converter, instrument, note, chord, stream
from music21 import corpus

import random
import time
import requests
import numpy as np
import tensorflow as tf

import config_bach

try:
    print ('MIDI: importing pabou helper')
    import pabou # various helpers module, one dir up
    # pip install prettytable
    print('pabou helper path: ', pabou.__file__)
    #print(pabou.__dict__)
except:
    print('MIDI: cannot import pabou')
    exit(1)



#########################################
# offset for MIDI file depend on pressure
#########################################
def get_weather():
    r = requests.get('http://api.openweathermap.org/data/2.5/weather?q=Sunnyvale&APPID=2aad759893e33a56984688d1f338700d')
    # r is response (200)
    j = r.json()
    pressure = j['main'] ['pressure']
    p = (pressure - 1000.0) / (1050.0-1000.0)
    offset = p *(0.8-0.3) + 0.3
    return(offset)


#########################################################################################
# MIDI ONLY; CREATE LIST OF PREDICTION, from random seed. returns list of strings pattern
# random seed, return list of pattern , C4 or 1.2 of len nb_predict. not real time
# get corpus
# assess use of temperature from config 
# random seed 
# create list 
# vectorize, predict using temperature
# return list of string pattern, seed plus predicted
###########################################################################################

def create_predicted_MIDI_list(model, nb_predict, corpus_notes, corpus_duration, corpus_velocity, pi_to_int, du_to_int, ve_to_int, unique_pi_list, unique_du_list, unique_ve_list):

    temp = config_bach.temperature_midi

    # get random start in corpus. 
    start = random.randint(0,len(corpus_notes)-config_bach.seqlen-1)
    print ('MIDI: predict list of %d elements. start at random %d within %d. model type: %d' %(nb_predict, start, len(corpus_notes), config_bach.model_type))
    
    # SEED . list of seqlen str from corpus.
    real_pitch_seq = corpus_notes[start:start+config_bach.seqlen] 
    real_duration_seq = corpus_duration[start:start+config_bach.seqlen] 
    real_velocity_seq = corpus_velocity[start:start+config_bach.seqlen]

    # GROUND THRUTH. next str from corpus
    ground_truth_pitch = corpus_notes[start+config_bach.seqlen:start+config_bach.seqlen+nb_predict]
    ground_truth_duration = corpus_duration[start+config_bach.seqlen:start+config_bach.seqlen+nb_predict]
    ground_truth_velocity = corpus_velocity[start+config_bach.seqlen:start+config_bach.seqlen+nb_predict]

    # LIST of INTEGER index. for vectorization
    # convert input list from str to int, then vectorize
    pitch_input_list_int = [] # ONE list of seqlen int indexes
    duration_input_list_int = [] # ONE list of seqlen int indexes
    velocity_input_list_int = [] # ONE list of seqlen int indexes

    # store PREDICTION STR. as many elements as we want to predict
    # will be populated with predicted str. contains only prediction. will append with seed for midi file
    list_of_predicted_pitch_str = []
    list_of_predicted_duration_str = [] # could stay empty if do not predict duration
    list_of_predicted_velocity_str = [] # could stay empty if do not predict velocity
    
    # pi_to_int(str) gives int index.    unique_pi_list(index) gives a str

    #####################################################################
    # initialize *_input_list_int with seqlen int. comes from SEED, converted from str to int
    # creates 3 list of seqlen INTEGER indexes
    ##########################################################################

    for pattern in real_pitch_seq: # # str F3, G#5 B-3
        p = pi_to_int[pattern] # convert string to INT for network input
        assert unique_pi_list[p] == pattern
        pitch_input_list_int.append(p) # accumulate sequence. ONE list of SEQLEN integer indexes   

    for pattern in real_duration_seq: # 
        p = du_to_int[pattern] # convert string to INT for network input
        assert unique_du_list[p] == pattern
        duration_input_list_int.append(p) # accumulate sequence. ONE list of SEQLEN integer indexes   

    for pattern in real_velocity_seq: # 
        p = ve_to_int[pattern] # convert string to INT for network input
        assert unique_ve_list[p] == pattern
        velocity_input_list_int.append(p) # accumulate sequence. ONE list of SEQLEN integer indexes   

    ################################################
    # VECTORIZE *_input_list_int
    # for model 1 creates X, one hot
    # for model 2 created X,X1,X2. keep as int
    ################################################

    if config_bach.model_type in [1,3]:
        # only use pitch
        # network_input_list is a list of seqlen integer indexes
        X = np.asarray(pitch_input_list_int) # shape (40,) int32

        # cast to expected input type
        # dtype from keras model is tf.int32, need to convert to numpy
        X = X.astype(model.inputs[0].dtype.as_numpy_dtype)

        if config_bach.model_type == 1:
            # convert to one hot as in training
            X = tf.keras.utils.to_categorical(X, num_classes=len(pi_to_int) ) #  one hot as in training . 
            # WARNING. if numclass not set, hot may be less than expected, depending on the content of X
            # shape(30,193) float32
            # add batch dimension
            X=np.reshape(X, (1 , config_bach.seqlen, len(pi_to_int)) ) # shape (nb_sample = 1,seqlen,feature dim)
        else:
            pass # do not one hot. expect integers

    if config_bach.model_type == 4:
        X = np.asarray(pitch_input_list_int) # shape (100,) int32
        # normalize
        X = X - config_bach.model4_mean
        X = X / config_bach.model4_std

        X = np.reshape(X, (config_bach.size, config_bach.size))
        X = X.astype(model.inputs[0].dtype.as_numpy_dtype)
        # at this point (6,6) conv2D expect 4 dims
        X = np.expand_dims(X,axis=0) # add batch
        X = np.expand_dims(X,axis=-1) # (1,6,6,1)

    if config_bach.model_type == 2:

        # vectorize and cast to expected input type
        # network_input_list is a list of seqlen integer indexes
        X = np.asarray(pitch_input_list_int)  #(30,)
        X = X.astype(model.inputs[0].dtype.as_numpy_dtype) # input vs inputs

        X1= np.asarray(duration_input_list_int) # (30,)
        X1 = X1.astype(model.inputs[1].dtype.as_numpy_dtype)

        X2= np.asarray(velocity_input_list_int) # (30,) # vectorize X2 even if possibly not used
        X2 = X2.astype(model.inputs[2].dtype.as_numpy_dtype)
        # cannot reshape array of size 30 into shape (1,30,129) , so reshape with feature dim = 1
        # for embedding no need to reshape


    # for model 1, only X is used, for model 2 [X,X1] or [X,X1,X2]

    good = 0
    inference_time = 0.0 # total inference time divided by nb_predict

    #############################################
    ############### PREDICTION ##################
    #############################################


    for i in range(nb_predict):
        #https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/
        # verbose 1, show timing

        ############################ 
        # model.predict, 
        # get softmax, 
        # temperature, 
        # argmax index, 
        # convert to string, 
        # store predicted string, 
        # check vs truth, 
        # update int input list, 
        # vectorize
        ############################

        # only use Full model for MIDI file (this is complex enough)
        
        if config_bach.model_type in [1,3,4] :

            start_time = time.time()
            prediction = model.predict(X,verbose=0) #  one head 
            inference_time = inference_time + time.time() - start_time # accumulate

            prediction_pitch = prediction[0] # prediction[0] is an array of softmax of len unique pi. only ONE array

            # modify proba, higher temp, increase surprise.
            # dynamic temperature using remi
            prediction_pitch = pabou.get_temperature_pred(prediction_pitch, temp)

            index = np.argmax(prediction_pitch) # just take highest proba in softmax float32 np.sum(prediction) = 1.000001
            target_pi = unique_pi_list[index] # index from softmax converted to str label
            # target_pi is str prediction

            # check predict vs ground truth
            truth_pitch = corpus_notes[start+config_bach.seqlen+i]
            
            #print ('%d: ground truth %s, prediction %s, max softmax %0.4f, softmax index %d' %(i, truth_pitch , target_pi,  prediction_pitch[index], index ) )

            if target_pi == truth_pitch:
                good = good + 1

            # update list of predicted string pitches 
            # use later MIDI file generation
            list_of_predicted_pitch_str.append(target_pi) # append string label, ready to be converted to MIDI

            #####################################################
            # rollover
            # updated the SEQLEN integer input for next iteration
            # predict rolling is done on integer input list , not X directly (messy for me) 
            # X.shape(1,50,127)
            # input list is integer, convert from str to int firt
            pitch_input_list_int.append(pi_to_int[target_pi])
            pitch_input_list_int = pitch_input_list_int[1:]

            # re vectorize for next iteration
            X = np.asarray(pitch_input_list_int) # shape (50,) int32
            X = X.astype(model.inputs[0].dtype.as_numpy_dtype)

            if config_bach.model_type == 1:
                X = tf.keras.utils.to_categorical(X, num_classes = len(pi_to_int)) #  one hot as in training. SET num classes 
                X=np.reshape(X, (1 , config_bach.seqlen, len(pi_to_int)) ) # shape (nb_sample = 1,seqlen,feature dim)   1,50,1 

            if config_bach.model_type == 4:
                X = X - config_bach.model4_mean
                X = X / config_bach.model4_std

                X = np.reshape(X, (config_bach.size, config_bach.size))
                X = np.expand_dims(X,axis=0) # add batch
                X = np.expand_dims(X,axis=-1) # (1,6,6,1)
        
            ######################################################

        # end for model 1 or 3

        ##########################################################################
        # Model 2
        # model.predict, get softmax, temperature, argmax index, convert to string, 
        # store predicted string, check vs truth, update int input list, vectorize
        ###########################################################################

        #prediction_pitch is an array of 30 softmax , vs expected 1; likewize for duration
        #GDL seems to consider only first one ????????
        #related for return = True in last layer
        
        ############################

        if config_bach.model_type == 2:

            if config_bach.predict_velocity: 
                x = [X,X1,X2]
            else:
                x = [X,X1]

            # return multiple predictions
            start_time = time.time()

            pred = model.predict(x=x, verbose=0) # multi head heads  Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
            
            inference_time = inference_time + time.time() - start_time

            # return multiple prediction array, 2 or 3
            if config_bach.predict_velocity: 
                (prediction_pitch, prediction_duration, prediction_velocity) = pred
                prediction_pitch = prediction_pitch[0]
                prediction_duration = prediction_duration[0]
                prediction_velocity = prediction_velocity[0]

            else:

                (prediction_pitch, prediction_duration) = pred
                prediction_pitch = prediction_pitch[0]
                prediction_duration = prediction_duration[0]
                prediction_velocity = None

            if config_bach.temperature != 0:
                # modify proba, higher temp, increase surprise.
                # dynamic temperature using remi
                prediction_pitch = pabou.get_temperature_pred(prediction_pitch, temp)
                prediction_duration = pabou.get_temperature_pred(prediction_duration, temp)
                if config_bach.predict_velocity:
                    prediction_velocity = pabou.get_temperature_pred(prediction_velocity, temp)
 
            # get predicted pitch
            index = np.argmax(prediction_pitch) # just take highest proba in softmax float32 np.sum(prediction) = 1.000001
            target_pi = unique_pi_list[index] # index from softmax converted to str label
            truth_pitch = corpus_notes[start+config_bach.seqlen+i]
            # target_pi is str prediction

            # get predicted duration
            index = np.argmax(prediction_duration) # just take highest proba in softmax float32 np.sum(prediction) = 1.000001
            target_du = unique_du_list[index] # index from softmax converted to str label
            truth_duration = corpus_duration[start+config_bach.seqlen+i]

            if config_bach.predict_velocity:
                # get predicted velocity
                index = np.argmax(prediction_velocity) # just take highest proba in softmax float32 np.sum(prediction) = 1.000001
                target_ve = unique_ve_list[index] # index from softmax converted to str label
                truth_velocity = corpus_velocity[start+config_bach.seqlen+i]
            
            #print ('%d: ground truth %s, prediction %s, max softmax %0.4f, softmax index %d' %(i, truth_pitch , target_pi,  prediction_pitch[index], index ) )

            # check vs truth
            # did we get it right for first 2
            if target_du == truth_duration and target_pi == truth_pitch :
                good = good + 1

            # accumulate list of str 
            # for MIDI file generation
            list_of_predicted_pitch_str.append(target_pi) # append string label, ready to be converted to MIDI
            list_of_predicted_duration_str.append(target_du) # append string label, ready to be converted to MIDI
            if config_bach.predict_velocity:
                list_of_predicted_velocity_str.append(target_ve) # append string label, ready to be converted to MIDI

            #############################################################################
            # ROLLOVER
            # update input list (integer) , so convert from str prediction to int first
            # then vectorize

            pitch_input_list_int.append(pi_to_int[target_pi])
            pitch_input_list_int = pitch_input_list_int[1:]

            duration_input_list_int.append(du_to_int[target_du])
            duration_input_list_int = duration_input_list_int[1:]

            if config_bach.predict_velocity:
                velocity_input_list_int.append(ve_to_int[target_ve])
                velocity_input_list_int = velocity_input_list_int[1:]

            # for model 2, no need to add batch dimension and convert to one hot
            X = np.asarray(pitch_input_list_int) 
            X1= np.asarray(duration_input_list_int)
            if config_bach.predict_velocity:
                X2= np.asarray(velocity_input_list_int)
            ###############################################################################
        
        # end model 2
          
    #  for range   
    

    ##############################################################
    # all predict done 
    # list of predicted strings available
    ###############################################################

    inference_time = inference_time / nb_predict
    print('MIDI: inference time average ms: %0.2f '  %(inference_time*1000.0))

    print('MIDI: predicted pitch  : ', list_of_predicted_pitch_str)
    print('MIDI: ground truth pitch: ', ground_truth_pitch)

    print('MIDI: ratio of good prediction (model 1 pi, model 2, pi and du): %0.1f' %(100.0 * good/nb_predict))
    # return list of strings, seqlen + nb_predic#

    assert config_bach.seqlen + config_bach.nb_predict == len(real_pitch_seq) + len(list_of_predicted_pitch_str)

    if config_bach.model_type in [1,3,4]:   
        return(real_pitch_seq + list_of_predicted_pitch_str, None, None) # list of str from proba to str labels

    if config_bach.model_type == 2: # 3 list real plus predicted. str
        return(real_pitch_seq + list_of_predicted_pitch_str, real_duration_seq + list_of_predicted_duration_str, real_velocity_seq + list_of_predicted_velocity_str) 
        # list of str from proba to str labels
    
    # list of predicted_velocity could be empty list

# end of create_predicted_midi_list 


    
#######################################################
# CREATE MIDI FILE
# LIST are STR (or NONE).
# includes seed to predictions
# set instrument from config. decode chords into notes. 
# create list of music21 objects
# compute offset, duration 
# set duration
# convert list of music21 object to MIDI file
#######################################################

def create_MIDI_file(list_of_pitch, list_of_duration, list_of_velocity, file):

    # list of duration and velocity is None for model 1
    # list of velocity is [] for model 2 and no predict velocity
    # list are seqlen+nb_predict, strings

    # file is name of MIDI file

    my_instrument = config_bach.my_instrument
    print('MIDI: create MIDI file with instrument %s. MIDI file %s: ' %(my_instrument, file))

    ####################################################################################
    # offset is either a fixed offset from config file, or a variable offset depending on sunnyvale pressure
    # or based on duration. offset prevents stacking
    ####################################################################################

    if config_bach.offset_config == 'd':
        fixed_offset_increment = config_bach.fixed_midi_offset_increment
    else:
        fixed_offset_increment = get_weather() # offset between notes depend on pressure

    if config_bach.use_fixed_offset_midi:
        print('MIDI: offset in MIDI file is fix ', fixed_offset_increment)
    else:
        print('MIDI: offset in MIDI file depend on duration')
    
        # DURATION: can use random from common duration or fixed duration from config_bach.fixed_duration_midi_file or predicted
        # BEWARE of duration zero

    #[('$0.2', 14069), ('$0.5', 10750), ('$1.0', 2907), ('$0.0', 1290), ('$0.3', 714)]

    music21_output_list=[] # list of music21 objects , notes, rest and chord objects, later streamed to MIDI file
    offset =0 # start of offset. 

    # use idx to step thru pitch and other 2 string lists 
    for idx, x in enumerate(list_of_pitch):  # for seed + predict

        #####################################################
        # set pi, duratyion and velocity
        # set pi as string 
        # set duration_in_quarter as float, from encoded in pitch, or predicted , or static config, or random from most common duration
        # set velocity as int, hardcoded or predicted
        ######################################################

        ### MODEL 1
        # set pi string for further processing into notes, chord and rest
        # set duration_in_quarter as float, from encoded in pitch, or static config, 
        # set velocity as int, hardcoded to some default

        if config_bach.model_type in [1,3,4]: 
            # check if $ exist; if yes, duration is encoded there
            if '$' in x:
                # duration as encoded in pitch
                duration = x.split('$')[1]
                duration.replace('$','')
                duration_in_quarter=float(duration) # str to float
                pi = x.split('$')[0] # can be a note or chord
            else: # duration not encoded
                pi=x # R or D3
                # set per notes duration at random
                # common duration  [('X0.2', 14069), ('X0.5', 10750), ('X1.0', 2907), ('X0.0', 1290), ('X0.3', 714)]
                #duration = random.choice(common_duration)[0]
                #duration = duration.replace('$', '')
                duration = config_bach.fixed_duration_midi_file # use config duration
                duration_in_quarter = float(duration)

            # fixed velocity
            velocity = 100

        ### MODEL 2
        # set pi for further processing into notes, chord and rest
        # set duration as predicted
        # set velocity if predicted, else hardcode

        if config_bach.model_type == 2:
            pi = x
            duration = list_of_duration[idx] # predicted duration STR at same index as pitch
            duration = duration.replace('$', '')
            duration_in_quarter = float(duration)

            if config_bach.predict_velocity:
                velocity = list_of_velocity[idx] # predicted velocity at same index as pitch
                velocity = int(velocity)
            else:
                velocity = 100

        ###########################################################
        # create music21 object based on pi, duration and velocity
        # object cabn be note, chord or rest
        # append this element to output list of music21 objects
        ###########################################################

        # pi str,  duration_in_quarter float, and velocity int 
        # process notes, chords, rest
        # use above velocity and duration_in_quarter for notes, and notes in chords
        # https://stackoverflow.com/questions/55170188/how-to-make-midi-file-from-notes-with-flute-instrument-in-python-music21-librar
           
        # pi is a CHORD  can be 1.2 or A4.A5
        if ('.' in pi): # chords, multiples notes at the same time
            notes_in_chord = pi.split('.') # 11.2 or F4.G5  depending on normal
            # '4'.split('.') return 4. no need at add extra dots. will generate empty string in split
            
            nn = [] # list of notes objects              
            #note.Note(index or 'C3') returns note object
            
            try: 
                for p in notes_in_chord:  # list of all notes in chord
                    if config_bach.normal:  
                        n=note.Note(int(p))  # create Note object integer to notes object 
                    else:
                        # create note from A4
                        n=note.Note(p) # exception raise PitchException("Cannot make a step out of '%s'" % usrStr)
                    
                    n.duration.quarterLength = duration_in_quarter # invididual duration notes, but will be the same for all notes in chords
                    n.volume.velocity = velocity
                    nn.append(n) # all notes object in Chord

            except Exception as e:
                print ('MIDI: exception chord error %s pi %s note_in_chord %s p %s' %( str(e) , pi, notes_in_chord, p ))
                #Cannot make a step out of 'R' R$0.2 ['R$0', '2'] R$0

            c = chord.Chord(nn) # create chord object from list of music21 notes objects
            c.offset = offset 
            music21_output_list.append(my_instrument)
            music21_output_list.append(c) # append this chord
            
        # pi is a REST
        elif (pi == 'R'): # rest
            r = note.Rest()
            r.duration.quarterLength = duration_in_quarter
            r.offset = offset # note offset
            music21_output_list.append(r) # append this rest
        
        # pi is a NOTE
        else: # this is note 'C3' or 2
            try: 
                # create note with pitch C4 gives n.pitch <music21.pitch.Pitch C4> and n.octave 4. can also see with n.nameWithOctave
                n = note.Note(pi) # convert string or int to note object, used to create stream. octave is stored
                n.offset = offset # note offset
                n.duration.quarterLength = duration_in_quarter   # 0.3 <music21.duration.Duration 3/10>
                n.volume.velocity = velocity
                # https://stackoverflow.com/questions/55170188/how-to-make-midi-file-from-notes-with-flute-instrument-in-python-music21-librar
                music21_output_list.append(my_instrument) # use given instrument
                music21_output_list.append(n) # append this note
            except Exception as e:
                print ('MIDI: Exception note', str(e) , pi)
            
        # Offset is (roughly) the length of time from the start of the piece. 
        # Duration is the time the note is held. 
        # The offset of a note will only be the sum of the previous durations if there are no rests (silences) in the piece and there are no cases where two notes sound together.
        # offset: a floating point value, generally in quarter lengths, specifying the position of the object in a site.

        if config_bach.use_fixed_offset_midi:
            offset = offset + fixed_offset_increment # else, notes stacked. sound better than adding duration
        else:
            offset = offset + duration_in_quarter # else, notes stacked 
            # should also cover REST

    # done with seed and all predictions

    pabou.inspect("MIDI: music21_output_list ", music21_output_list)

    ########################################################
    # convert list of music21 object to midi file
    ########################################################
    print("MIDI: convert list of music21 object to MIDI file: ", file)
    midi_stream = stream.Stream(music21_output_list)
    midi_stream.write('midi', fp = file)
    print('MIDI: ==== > MIDI file %s created' %file)
    