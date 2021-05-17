#!/usr/bin/env  python

##########################################
# play_bach configuration file 
##########################################

from music21 import instrument
import os

###########################################
# debug mode (not used, just crank down epoch)
###########################################
debug=False # True = load small corpus, few epochs. also configurable with command line -d
# !!! to use a small corpus, need to parse MIDI file. so delete pick file first
debug_epoch = 5
debug_file_number = 5 # number file to read

###########################################
# file names
###########################################
bach_files = os.path.join('training' , '*.mid') # all midi files to use for training

pick = "corpus.pick"
plot_file = "model_plot.png"
training_file = "training_result.png"
history_file = 'history.pick'

midi_file = "generated.mid"  # will be prefixed with model name

tensorboard_log_dir = 'tensorboard_log_dir'
log_file = 'log_play_bach.log' # for flask inference
dashboard_config = 'dashboard.conf'

###########################################
# corpus parsing
###########################################
concatenate = False # only for model 1. True to have duration encoded as pitch$duration. need to delete corpus.pick.  True generates larger corpus
predict_velocity = True # only for model 2. if true 3 heads, else 2 heads
normal = False  # True use normal order 1.2 for chords, ie forget octave (default octave 4) or False uses eg F5.A4.D5.D4.D3. False generates larger corpus

max_duration = 8.0  # cap duration to reduce corpus size when concatenating
low_occurence = 5 # not used yet

chordify = False # True: all notes in differents parts converted to chords in single line, not polyphonic

###########################################
# hyperparameters and model definition
###########################################
# upper bound on number of hidden cells = number of samples / alpha *(number of input neurons + number of output neurons)
# alpha from 2 to 10, def 5?

seqlen=40 # length of input sequence
epochs=70 # max number. training will stop earlier if accuracy not improving. can be overwritten by command line -e  60
batch=64 # power of 2

model_type=1 # = use model type 1 or 2 for last char of app/corpus. retrieve model type dynamically
#1: one hot, pitch OR pitch$duration.  chord pitch either normal or full. single softmax. can only predict seen combinations
#2: embedding, stacked RNN with attention. input integer indexes. multiple head , multiple softmax. can predict unseen combinations

###########################################
# training set and model names
###########################################
app = 'goldberg'  # base name. should be synched with content of training directory
# TRAINING: app/corpus. prefix for model saved file, and pickle corpus. models/app_<>
# init remi corpus gui , to be able to reload new corpus
# modify app/corpus name based on config

# keras h5 model file name:
# app1_c for concatenate, app1_nc for non concatenate
# app2_v for velocity app2_nv for non velocity  app2_v
# ends with _mo if normal (ie multiple octave ) else _so

if model_type == 1:
    app = app + '1'
else:
    app = app + '2'

if model_type == 1 and  concatenate == False:
    app = app + '_nc'
if model_type == 1 and  concatenate == True:
    app = app + '_c'

if model_type == 2 and  predict_velocity == False:
    app = app + '_nv'
if model_type == 2 and  predict_velocity == True:
    app = app + '_v'

if normal == True:
    app = app + '_so' # simple octave
else:
    app = app + '_mo' # multiple octave
	
if model_type == 2:
    concatenate = False # create R$0.2 and model 2 believes this is a chord because there is a .

# can be modified by GUI
tfmodel = 'Full' # inference type , Full model or one of the TFlite. init remi and default inference


###################################
# static MIDI file only
###################################
nb_predict=100 # for MIDI file, number of notes to predict and write to file
offset_config='d'  # 'd' for default 'w' for weather based
use_fixed_offset_midi = False # True: increment offset with fixed offset False: use duration (fixed or predicted)
fixed_midi_offset_increment = 0.5 # MIDI ONLY. fixed offset if config is 'd' , otherwize offset is from weather.
fixed_duration_midi_file = 0.3  # MIDI ONLY. in quarter length. in case duration is not predicted. could look at top duration to get an idea 
#my_instrument = instrument.Clavichord()
#my_instrument = instrument.Trumpet()
my_instrument = instrument.Flute()
temperature_midi = 0  # ie do not use temp

###################################
# real time synth only
###################################
temperature = 0 # to init remi slider. slider between 0 and 1. 0 means does not use temp. can be modified via GUI.
predict_forever = 5000 # notes predicted for stream. 
d1=0.3 # default fixed duration in QUARTER for streaming prediction. could also use random from most common duration OR predicted OR duration in concatenate
d2=0.1 # for streaming. off time IN SEC. 
velocity = 90 # ie sound volume

sound_font = "./FluidR3_GM.sf2"
sampleRate = 44100
bitsPerSample = 16
channels = 2
BPM=120 # to init remi slider cast to float when used

#instrument = 'Harsichord' # to init remi
instrument = 'Church Organ' # to init remi

time_signature_beat_per_measure = 4 # 4 beats per measure . not used I guess
time_signature_beat_type = 4 # beat is quarter note. changing this is the same as changing BPM ?


"""
instruments_dict = {
"1":"Acoustic Piano",
"2":"BrtAcou Piano",
"3":"ElecGrand Piano",
"4":"Honky Tonk Piano",
"5":"Elec.Piano 1",
"6":"Elec.Piano 2",
"7":"Harsichord",
"8":"Clavichord",

"9":"Celesta",
"10":"Glockenspiel",
"11":"Music Box",
"12":"Vibraphone",
"13":"Marimba",
"14":"Xylophone",
"15":"Tubular Bells",
"16":"Dulcimer",

"17":"Drawbar Organ",
"18":"Perc. Organ",
"19":"Rock Organ",
"20":"Church Organ",
"21":"Reed Organ",
"22":"Accordian",
"23":"Harmonica",
"24":"Tango Accordian",

"25":"Acoustic Guitar",
"26":"SteelAcous. Guitar",
"27":"El.Jazz Guitar",
"28":"Electric Guitar",
"29":"El. Muted Guitar",
"30":"Overdriven Guitar",
"31":"Distortion Guitar",
"32":"Guitar Harmonic",

"33":"Acoustic Bass",
"34":"El.Bass Finger",
"35":"El.Bass Pick",
"36":"Fretless Bass",
"37":"Slap Bass 1",
"38":"Slap Bass 2",
"39":"Synth Bass 1",
"40":"Synth Bass 2",

"41":"Violin",
"42": "Viola",
"43":"Cello",
"44":"Contra Bass",
"45":"Tremelo Strings",
"46":"Pizz. Strings",
"47":"Orch. Strings",
"48":"Timpani",

"49":"String Ens.1",
"50":"String Ens.2",
"51":"Synth.Strings 1",
"52":"Synth.Strings 2",
"53":"Choir Aahs",
"54": "Voice Oohs",
"55": "Synth Voice",
"56":"Orchestra Hit",

"57":"Trumpet",
"58":"Trombone",
"59":"Tuba",
"60":"Muted Trumpet",
"61":"French Horn",
"62":"Brass Section",
"63":"Synth Brass 1",
"64":"Synth Brass 2",
 
"65":"Soprano Sax",
"66":"Alto Sax",
"67":"Tenor Sax",
"68":"Baritone Sax",
"69": "Oboe",
"70":"English Horn",
"71":"Bassoon",
"72":"Clarinet",

"73":"Piccolo",
"74":"Flute",
"75":"Recorder",
"76":"Pan Flute",
"77":"Blown Bottle",
"78":"Shakuhachi",
"79":"Whistle",
"80":"Ocarina",

"81":"Lead1 Square",
"82":"Lead2 Sawtooth",
"83":"Lead3 Calliope",
"84":"Lead4 Chiff",
"85":"Lead5 Charang",
"86":"Lead6 Voice",
"87":"Lead7 Fifths",
"88":"Lead8 Bass Ld",

"89":"Pad1 New Age",
"90":"Pad2 Warm",
"91":"Pad3 Polysynth",
"92":"Pad4 Choir",
"93":"Pad5 Bowed",
"94":"Pad6 Metallic",
"95":"Pad7 Halo",
"96":"Pad8 Sweep",

"97":"FX1 Rain",
"98":"FX2 Soundtrack",
"99":"FX3 Crystal",
"100":"FX4 Atmosphere",
"101":"FX5 Brightness",
"102":"FX6 Goblins",
"103":"FX7 Echoes",
"104":"FX8 Sci-Fi",

"105":"Sitar",
"106":"Banjo",
"107":"Shamisen",
"108":"oto",
"109":"Kalimba",
"110": "Bagpipe",
"111": "Fiddle",
"112": "Shanai",

"113":"TinkerBell",
"114":"Agogo",
"115":"SteelDrums",
"116":"Woodblock",
"117":"TaikoDrum",
"118":"Melodic Tom",
"119":"SynthDrum",
"120":"Reverse Cymbal",

"121":"Guitar Fret Noise",
"122": "Breath Noise",
"123":"Seashore",
"124":"BirdTweet",
"125":"Telephone",
"126":"Helicopter",
"127":"Applause",
"128":"Gunshot"
}

"""