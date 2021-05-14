import config_bach
import music21
from music21 import converter, instrument, note, chord, stream
from music21 import corpus

print('test midi')

my_instrument = config_bach.my_instrument
music21_output_list=[]
offset = 0

file = 'test_midi.mid'
duration_in_quarter = 0.5  # actual len of note
offset_increment = 0.5 # spacing, if > duration, blank
velocity = 100

for pi in range(20):
    n = note.Note(60+pi)
    n.offset = offset # note offset
    n.storedInstrument = my_instrument 
    n.duration.quarterLength = duration_in_quarter  
    n.volume.velocity = velocity
    music21_output_list.append(n) # list of notes object

    #offset = offset + offset_increment # else, notes stacked. sound better than adding duration
    offset = offset + duration_in_quarter # else notes stacked, ie all played at once

print("convert list of music21 object to MIDI file: ", file)
midi_stream = stream.Stream(music21_output_list)
midi_stream.write('midi', fp = file)
print('==== > MIDI file created')

# Offset is (roughly) the length of time from the start of the piece. 
# Duration is the time the note is held. 
# The offset of a note will only be the sum of the previous durations if there are no rests (silences) in the piece and there are no cases where two notes sound together.
# offset: a floating point value, generally in quarter lengths, specifying the position of the object in a site.



