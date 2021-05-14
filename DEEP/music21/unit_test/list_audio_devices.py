#!/usr/bin/python

# fluidsynth on windows
#https://ksvi.mff.cuni.cz/~dingle/2019/prog_1/python_music.html

# python binding
#https://github.com/SpotlightKid/pyfluidsynth

import pyaudio


#fluidsynth FluidR3_GM.sf2 book1-prelude01.mid

#########################################################################
# audio configuration
#########################################################################

print('list audio devices')

p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

print ("num device ", numdevices)


print ("\ninfo by host api:")
#for each audio device, determine if is an input or an output and add it to the appropriate list and dictionary
for i in range (0,numdevices):
        if p.get_device_info_by_host_api_device_index(0,i).get('maxInputChannels')>0:
                print ("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0,i).get('name'))

        if p.get_device_info_by_host_api_device_index(0,i).get('maxOutputChannels')>0:
                print ("Output Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0,i).get('name'))

print ("\ninfo by index:")
try:
	for i in range(numdevices):
		devinfo = p.get_device_info_by_index(i)
		print ("index ",i, devinfo.get('name'))
except:
	pass

print ("\nDEFAULT input")
print (p.get_default_input_device_info())
print ("\nDEFAULT output")
print (p.get_default_output_device_info())


#if p.is_format_supported(44100.0,  # Sample rate
#                         input_device=devinfo["index"],
#                         input_channels=devinfo['maxInputChannels'],
#                         input_format=pyaudio.paInt16):
#  print 'Yay!'
p.terminate()

#######################################################################################
# realtime playback of Music21 Streams as MIDI.
# Create a player for a stream that plays its midi version in realtime using pygame.
#######################################################################################

#conda install -c cogsci pygame failed python 3.5 vs 3.7
# pip install pygame

print('use music21 stream player via pygame MIDI')
import random
keyDetune = []
for i in range(127):
   keyDetune.append(random.randint(-30, 30))

from music21 import *

b = corpus.parse('bwv66.6')
# returns a score
# pitch.name, step, octave, namewithoctave frequency, midi, 

for n in b.flat.notes:
   n.pitch.microtone = keyDetune[n.pitch.midi] # modify stream

sp = midi.realtime.StreamPlayer(b)
#sp.play()


################################################################################
# play MIDI with pygame
################################################################################

print('play MIDI with pygame')
import pygame.midi
import pygame
import time

print('init pygame')
pygame.init()
pygame.midi.init()
 
port = pygame.midi.get_default_output_id()
print ('using output id %s' %(port))
# device id
player = pygame.midi.Output(port,0)
print('midi player created')

# instrument 0 grand piano, 19 church organ
player.set_instrument(0)


for i in range(10):
        player.note_on(60+i, 127) # midi 64, velocity = max
        time.sleep(0.2)
        player.note_off(60+i, 127)
        time.sleep(0.1)

player.note_on(72, 127)
player.note_on(76, 127)
player.note_on(79, 127)
time.sleep(1)
player.note_off(72, 127)
player.note_off(76, 127)
player.note_off(79, 127)
time.sleep(0.1)

del player
pygame.midi.quit()

#############################################################################
# play MIDI with fluidsynth
#############################################################################

print('use fluidsynth with MIDI notes')
import time
import fluidsynth

fs = fluidsynth.Synth()
# call start if not using audio samples
#The start() method starts audio output in a separate thread.
fs.start(driver = 'dsound')  # use DirectSound driver. 

#FluidSynth needs a file FluidR3_GM.sf2 that contains waveforms for various musical instruments.

sfid = fs.sfload(r'c:\users\pboud\DEEP\music21\FluidR3_GM.sf2')  # replace path as needed

# program_select(track or channel, soundfontid, banknum, presetnum)

# select instrument with bank and preset ?
fs.program_select(0, sfid, 0, 0) 

#noteon(track, midinum, velocity)

for i in range(10):
        fs.noteon(0, 30+i, 90)
        fs.noteon(0, 37+i, 90)
        fs.noteon(0, 36+i, 90)

        time.sleep(0.2)

        fs.noteoff(0, 30+i)
        fs.noteoff(0, 37+i)
        fs.noteoff(0, 36+i)

        time.sleep(0.1)

fs.delete()

#You can also manage audio IO yourself and just use FluidSynth to calculate the samples for the music.
print('use fluidsynt with samples')

import time
import numpy
import pyaudio
import fluidsynth

pa = pyaudio.PyAudio()
strm = pa.open(
    format = pyaudio.paInt16,
    channels = 2, 
    rate = 44100, 
    output = True)
print('pyaudio stream opened')
s = []

fl = fluidsynth.Synth()

# do not call start

# Initial silence is 1 second
# get_samples (len) to generate next chunck of data, len is number of audio samples
s = numpy.append(s, fl.get_samples(int(44100 * 1)))  

# return numpy array of audio samples stereo 2xlen

sfid = fl.sfload(r'c:\users\pboud\DEEP\music21\FluidR3_GM.sf2')
fl.program_select(0, sfid, 0, 0)


for i in range(10):

        fl.noteon(0, 80+i, 100)
        fl.noteon(0, 87+i, 100)
        fl.noteon(0, 96+i, 100)

        # Chord is held for x seconds
        s = numpy.append(s, fl.get_samples(int(44100 * 0.25)))

        fl.noteoff(0, 80+i)
        fl.noteoff(0, 87+i)
        fl.noteoff(0, 96+i)

        # Decay of chord is held for x second
        s = numpy.append(s, fl.get_samples(int(44100*0.25)))

print('np samples ' , len(s))
print('in sec ? ', len(s)/44100)

fl.delete()

#convert an array of samples into a string of bytes suitable for sending to the soundcard, use
samps = fluidsynth.raw_audio_string(s)

print ('audio samples ' , len(samps))
print ('Starting playback on pyaudio')
strm.write(samps)


