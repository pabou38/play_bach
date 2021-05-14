#!/usr/bin/env  python 

# 26 jan

#https://ksvi.mff.cuni.cz/~dingle/2019/prog_1/python_music.html

#Installed c:\users\pboud\appdata\local\programs\python\python38\lib\site-packages\pyfluidsynth-1.2.5-py3.8.egg
#c:\users\pboud\anaconda3\envs\tf21-gpu\lib\site-packages\pyfluidsynth-1.2.5-py3.7.egg

#####################################################
# various audio related 
#####################################################

import pyaudio
import fluidsynth
import time
import numpy as np
import music21
import config_bach

# to access BPM
import my_remi

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

instrument_str_to_int = dict(map(reversed, instruments_dict.items()))
#print(instrument_str_to_int)

#############################################
# init audio, pyaudio and fluid
# play some notes with various method as test
# return stream and fluid and sfid as global variable in main
#############################################

# create pyaudio and fluidsynth
def init_audio(sound_font):

    print('init pyaudio')
    pa = pyaudio.PyAudio()
    pa_strm = pa.open(
        format = pyaudio.paInt16,
        channels = 2, 
        rate = 44100, 
        output = True)
    
    print('init fluidsynth')
    fluid = fluidsynth.Synth()  

    print('play with fuild and noteon')
    # call start
    fluid.start(driver = 'dsound')  # use DirectSound driver starts audio output in a separate thread.  
    
    sfid = fluid.sfload(sound_font)
    # track, soundfontid, banknum, presetnum
    #https://www.noterepeat.com/articles/how-to/213-midi-basics-common-terms-explained
    # instrument 20 Churn Organ
    fluid.program_select(0, sfid, 0, 20)

    #noteon(track, midinum, velocity)

    # play with noteon. 

    for i in range(3):
        fluid.noteon(0, 60+i, config_bach.velocity)
        fluid.noteon(0, 63+i, config_bach.velocity)
        fluid.noteon(0, 66+i, config_bach.velocity)

        time.sleep(0.2)

        fluid.noteoff(0, 60+i)
        fluid.noteoff(0, 63+i)
        fluid.noteoff(0, 66+i)

        time.sleep(0.1)

    fluid.delete()

    #other method is to use fluid to generate PCM samples, and manage IO
    print('play with audio samples')

    fluid = fluidsynth.Synth()

    sfid = fluid.sfload(r'c:\users\pboud\DEEP\music21\FluidR3_GM.sf2')
    # instrument 19
    fluid.program_select(0, sfid, 0, 19)

    for i in range(3):
        s=[]
        # Initial silence is 1 second
        # get_samples (len) to generate next chunck of data, len number of audio samples
        s = np.append(s, fluid.get_samples(int(44100 * 1))) 

        fluid.noteon(0, 80+i, config_bach.velocity)
        fluid.noteon(0, 83+i, config_bach.velocity)
        fluid.noteon(0, 86+i, config_bach.velocity)

        # Chord is held for 2 seconds
        s = np.append(s, fluid.get_samples(int(44100*0.3)))

        fluid.noteoff(0, 80+i)
        fluid.noteoff(0, 83+i)
        fluid.noteoff(0, 86+i)

        # Decay of chord is held for 
        #  cast to int
        s = np.append(s, fluid.get_samples(int(44100*0.1)))

        #convert an array of samples into a string of bytes suitable for sending to the soundcard, use
        samps = fluidsynth.raw_audio_string(s) # bytes

        pa_strm.write(samps) 

    return(pa_strm, fluid, sfid)
    # fluid and pyaudio streams are global in main


#################################
# .wav file header for streaming
#################################

# Generates the .wav file header for a given set of samples and specs
#def genHeader(sampleRate, bitsPerSample, channels, samples):
def genHeader(sampleRate, bitsPerSample, channels) -> bytes:
    #datasize = len(samples) * channels * bitsPerSample // 8
    datasize = 2000*10**6 # 2Gigabytes
    o = bytes("RIFF",'ascii')                                               # (4byte) Marks file as RIFF
    o += (datasize + 36).to_bytes(4,'little')                               # (4byte) File size in bytes excluding this and RIFF marker
    o += bytes("WAVE",'ascii')                                              # (4byte) File type
    o += bytes("fmt ",'ascii')                                              # (4byte) Format Chunk Marker
    o += (16).to_bytes(4,'little')                                          # (4byte) Length of above format data
    o += (1).to_bytes(2,'little')                                           # (2byte) Format type (1 - PCM)
    o += (channels).to_bytes(2,'little')                                    # (2byte)
    o += (sampleRate).to_bytes(4,'little')                                  # (4byte)
    o += (sampleRate * channels * bitsPerSample // 8).to_bytes(4,'little')  # (4byte)
    o += (channels * bitsPerSample // 8).to_bytes(2,'little')               # (2byte)
    o += (bitsPerSample).to_bytes(2,'little')                               # (2byte)
    o += bytes("data",'ascii')                                              # (4byte) Data Chunk Marker
    o += (datasize).to_bytes(4,'little')                                    # (4byte) Data size in bytes
    return o


###################################################
# number of quarter notes to second
# use BPM from remi, and time signature from config
###################################################
def quarter_to_time(dur) -> float: # dur in quarter note , str
    # returns duration in sec 
    # input number of quarter notes
    # 4/4 4 beat/measure  beat = one quarter
    # 120 BPM one beat = 1/2 sec
    # one quarter = 1/2 sec
    # second = nb of quarter / 2
    # 1 beat = 60/BPM sec . 1 beat = 1/b notes.  1 note = 60*b/BPM sec.  1/4 note = 60*b/BPM*4 sec
    a = config_bach.time_signature_beat_per_measure # !!!!!! NOT USED
    b = config_bach.time_signature_beat_type

    # beat type 1/2 note and BPM halved, same

    quarter_in_sec = (60.0 * float(b)) / (float(my_remi.BPM) * 4.0) #1 quarter = 0.5 sec with BMP = 120 AND 4/4
    duration_in_sec = float(dur) * quarter_in_sec
    return (duration_in_sec)

#######################################################################
# convert single note as string to MIDI number
# input single note as string eg 'C4' , return midi number used for synth
# use event list.  could also go thru m21 objects

#n=music21.note.Note(pi)  # str to notes object               
#(n.pitch.midi) # get midi number
######################################################################

def note_string_to_midi_number(single_note):
    
    #c = music21.chord.Chord(['c3', 'g#4', 'b5'])

    # use chord ... 
    c = music21.chord.Chord([single_note])

    # velocity 1 to 127
    c.volume = music21.volume.Volume(velocity=config_bach.velocity)
    c.volume.velocityIsRelative = False

    #Translates a Chord object to a list of base.DeltaTime and base.MidiEvents objects.
    eventList = music21.midi.translate.chordToMidiEvents(c)

    """
    [<MidiEvent DeltaTime, t=0, track=None, channel=None>,
    <MidiEvent NOTE_ON, t=0, track=None, channel=1, pitch=48, velocity=90>,
    <MidiEvent DeltaTime, t=0, track=None, channel=None>,
    <MidiEvent NOTE_ON, t=0, track=None, channel=1, pitch=68, velocity=90>,

    """
    # single note, so pitch is there
    midi = eventList[1].pitch
    
    print ("single note string to MIDI number ", single_note , midi) 
    return(midi)


########################################################
# use to play locally with pyaudio. single notes
# create audio samples from single note, ie does not handle chords
# uses fluidsynth, note on/off
# input note as string 'C4', return audio samples, ready to be written to pyaudio
# note on for d1 duration in quarter, off for d2
# velocity from config file
# get instrument as string. use reverse dict to set in fluid
########################################################

def get_audio_sample_from_one_note(fluid,note,d1,d2,instrument,sfid): 
    # note is string  
    # d1 is on time, d2 is off time

    # set instrument integer, using reverse dict and fuild select
    instrument_int = int(instrument_str_to_int[instrument])
    fluid.program_select(0, sfid, 0, instrument_int)

    s=[] # audio samples

    midi = note_string_to_midi_number(note)

    #noteon(track, midinum, velocity)
     
    fluid.noteon(0, midi, config_bach.velocity)
    s = np.append(s, fluid.get_samples(int(44100 * d1)))
    fluid.noteoff(0, midi)
    s = np.append(s, fluid.get_samples(int(44100 * d2))) 

    samps = fluidsynth.raw_audio_string(s)
    # return array of audio samples
    return(samps)



##############################################################
# creates audio sample 
# real time synthesis. call by flask
# input one note/chord/rest pattern, one duration in quarter, one velocity
# create list of MIDI numbers (for chords), and one duration
# velocity
# music21 object are used to convert note str into midi number
# duration in sec, converted from quarter (float) passed as argument
# instrument is passed as string. convert to music21 object but NOT NEEDED
# convert instrument from string to instrument as int to use in fluid select
# note off duration d2 from config file
# return audio sample
# use fluid note (midi) one and np array of samples
##############################################################
i=0
def get_audio_sample_from_one_pattern(pi, duration, velocity, fluid, instrument, sfid):
    global i

    # if velocity was not predicted, use default
    if velocity is None:
        velocity = config_bach.velocity

    midi_list = ([],None)  # list of one or more MIDI number and time in sec, converted from duration in quarter
    #ONE midi list of ([midi number,....], duration in sec) . multiple MIDI number if chord

    """
    #m21_instrument = music21.instrument(instrumentName=instrument) 
    # NOT NEEDED. instrument is set in fluid preset , not m21 object
    """

    # set instrument integer, using reverse dict, to set fuild select
    instrument_int = int(instrument_str_to_int[instrument])

    fluid.program_select(0, sfid, 0, instrument_int) 

    # get duration in quarter

    # check if $ exist; if yes, duration is encoded there, and not in passed argument
    if '$' in pi:

        # ignore duration argument 
        # set per notes duration
        duration = pi.split('$')[1]
        duration.replace('$','') 
        duration_in_quarter=float(duration)

        pi = pi.split('$')[0]
        
    else: # use duration passed as argument

        duration_in_quarter = duration 
        # no need to touch pi


    ###############  chord            
    #if ('.' in pi): # chords, multiples notes at the same time
    if pi.find('.') != -1:

        notes_in_chord = pi.split('.') # 11.2 or F4.G5  depending on normal
        # '4'.split('.') return 4. no need at add extra dots. will generate empty string in split
            
        if config_bach.normal: # 1.2
            try:
                c=[]
                for p in notes_in_chord:
                    #note.Note(index or 'C3') returns note object
                    n=music21.note.Note(int(p))  # integer to notes object
                    #n.storedInstrument = m21_instrument  # individual note instrument
                    c.append(n.pitch.midi) # get midi number

                # create MIDI list; convert quarter to sec
                midi_list = (c, quarter_to_time(duration_in_quarter)) #    # tuple ([],float) convert quarter to sec, based on BMP
            except Exception as e:
                print ('exception pitch ', pi, notes_in_chord, str(e))
                    
        else: # C4.D3 not normal
            try:
                c=[]
                for p in notes_in_chord:                
                #note.Note(index or 'C3') returns note object
                    n = music21.note.Note(p) # convert to note object   # exception raise PitchException("Cannot make a step out of '%s'" % usrStr)
                    #n.storedInstrument = m21_instrument  # individual note instrument
                    c.append(n.pitch.midi)

                # create MIDI list; convert quarter to sec
                midi_list = (c, quarter_to_time(duration_in_quarter)) # convert quarter to sec, based on BMP
            except Exception as e:
                print ('exception pitch ', pi, notes_in_chord, str(e))

    ###############  rest                  
    elif (pi == 'R'): # rest
        #r = note.Rest()
        # create MIDI list; convert quarter to sec
        midi_list = ([-1],quarter_to_time(duration_in_quarter)) # midi number = -1 means REST
        
    ###############  note                
    else: # this is note 'C3'
        try:
            n = music21.note.Note(pi) # convert string to note object, used to create stream
            #n.storedInstrument = instrument # instrument

            # create MIDI list; convert quarter to sec
            midi_list = ([n.pitch.midi],quarter_to_time(duration_in_quarter)) # get midi number for note object  
        except Exception as e:
            print ('exception pitch ', pi, str(e))

    ###############################           
    # convert midi to audio sample
    ###############################
    #print('MIDI',i, midi_list, pi); i = i+1
    
    # midi list is created list of [[midi number,], duration in sec]
    # create audio sample from list of one or more midi number
    # 44.1 K samples per secondes 
    s=[] # audio sample

    duration_sec = midi_list[1]

    if midi_list[0][0] == -1: # rest
            # use duration in sec there
            s = np.append(s, fluid.get_samples(int(config_bach.sampleRate * duration_sec))) # 
            
    else:
        
        # midi_list  ([one or more midi number],duration in sec)
        # note on
        # set VELOCITY
        for m in midi_list[0]:
            fluid.noteon(0, m, velocity)   # predicted velocity no longer from config file
        # use duration in sec here
        s = np.append(s, fluid.get_samples(int(config_bach.sampleRate * duration_sec))) # chord is held for x seconds
        
        # note off
        for m in midi_list[0]:
            fluid.noteoff(0, m)   
        s = np.append(s, fluid.get_samples(int(config_bach.sampleRate * config_bach.d2)))    # Decay of chord is held for x second

    #https://github.com/FluidSynth/fluidsynth/wiki/UserManual
    #https://github.com/nwhitehead/pyfluidsynth

    samps = fluidsynth.raw_audio_string(s)
    return(samps)