#!/usr/bin/python


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
print(instrument_str_to_int)


import pyaudio
import time
import fluidsynth

fs = fluidsynth.Synth()
# call start if not using audio samples
#The start() method starts audio output in a separate thread.
fs.start(driver = 'dsound')  # use DirectSound driver. 

#FluidSynth needs a file FluidR3_GM.sf2 that contains waveforms for various musical instruments.

sfid = fs.sfload(r'c:\users\pboud\DEEP\music21\FluidR3_GM.sf2')  # replace path as needed

# program_select(track or channel, soundfontid, banknum, presetnum)

# bank 0 , preset are instrument
print('scan instrument')
for i in range(128):
        fs.program_select(0, sfid, 0, i) 
        print(i+1, instruments_dict[str(i+1)])

        fs.noteon(0, 60, 90)
        fs.noteon(0, 66, 90)
        fs.noteon(0, 69, 90)
        time.sleep(0.5)
        fs.noteoff(0, 60)
        fs.noteoff(0, 66)
        fs.noteoff(0, 69)
        time.sleep(0.1)


fs.delete()


""" 
Pour rester compatible avec la norme MIDI, les emplacements mémoire (ROM ou RAM) sont “découpés” en banques de 
128 programmes/presets.

One Bank is a virtual collection of 128 patches (instruments). Virtual means that it is not a memory cell, 
sector on the disk, directory, flash-drive, or anything physical. It’s just a group of 128 programs. 
They may be somehow logically related, e.g. one Bank has Piano sounds, another Horns, and yet another Strings, 
but this is not a rule.
"""