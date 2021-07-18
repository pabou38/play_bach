#!/usr/bin/python3

"""
on windows, use connect vs do
cannot define sub containers
"""

#https://remi.readthedocs.io/en/latest/remi.html

instruments = {
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

instrument_name_list=[]
for x in instruments:
	#print(instruments[x])
	instrument_name_list.append(instruments[x])

"""
you can define the size of widgets as percentage instead of pixels.
widget = gui.Widget (width="50%", height ="20%")
This allows to auto size widget
to make the position change also you should use VBox or HBox as containers, and the contained widget must have the style parameter position as relative
"""

from time import sleep
from types import coroutine
import remi.gui as gui
from remi import start, App, Widget

import os, sys
import glob
import json
import re

# use for remi user,  password and ports
with open('remi.json' ,'r') as fp:
	data = json.load(fp)
print('my_remi.py: remi.json ', data)

title= 'play bach'

import config_bach

# init remi with value from config file. then set by remi
# instrument, corpus, BMP, temp, model


instrument  = config_bach.instrument
#temperature = config_bach.temperature/100.0 # label display 
temperature = config_bach.temperature  # init to 0, value between 0 and 1
BPM = config_bach.BPM
corpus = config_bach.app # to init label

# set to true when corpus is changed in the GUI., check in handler for /play
load_new_corpus = False 

corpus_list = [] # a list will initialize the drop down
tf_list = []

# build list of existing models based on h5 files (full models) or tflite
# remove suffix. NEED to be consistent with pabou.py (or import pabou.py)
p = os.path.join(os.getcwd(), 'models', '*.h5')
for file in glob.glob(p):
	file = os.path.basename(file)
	file = re.sub('_full_model.h5', '', file) # on file system, name is cello1_c_mo_full_model.h5
	corpus_list.append(file) 
	# when passed to main, pabou.load_model() will happend  h5_file = app + '_' +  full_model_h5_file 
	# full_model_h5_file = 'full_model_h5.h5' # will be saved as .h5py

# build list of existing lite models based on .tflite files    cello1_c_mo_fp32_lite.tflite
p = os.path.join(os.getcwd(), 'models', '*.tflite')
for file in glob.glob(p):
	file = os.path.basename(file)
	file = re.sub('.tflite', '', file)
	corpus_list.append(file)

#file = ''.join(file)  # convert list to string
#file = file.split('.') [0] # ignore tflite
#if file not in tf_list: # multiple corpus have same TFfile name
#	tf_list.append(file)

print('my_remi.py: corpus_list ', corpus_list)

# size of root container, ie the screen
# try and error. does not resize with mouse
		
w=320 # galaxy S9 in portrait
h=500


class remi_app(App):
	def __init__(self, *args):
		res_path = os.path.join(os.path.dirname(__file__), 'res')
		print('res path!: ', res_path)
		#super(remi_app, self).__init__(*args, static_file_path={'res':res_path})
		super(remi_app, self).__init__(*args)
		

	def main(self):
		self.colors = {'white' : '#ffffff', 'black' : '#000000', 'gray' : '#919191', 'x' : '#ff0000', 'red' : '#ff0000', 'green' :'#00ff00', 'blue' : '#0000ff'}

		#verticalContainer = gui.Container(width=w, height=h, margin='0px auto', style={'display': 'block', 'overflow': 'hidden', 'text-align':'center'})
		# AttributeError: module 'remi.gui' has no attribute 'Container'  on WIN10

		"""
		The purpose of this VBOX widget is to automatically vertically aligning the widgets that are appended to it.
		Does not permit children absolute positioning.
		In order to add children to this container, use the append(child, key) function. The key have to be numeric and
		determines the children order in the layout.
		Note: If you would absolute positioning, use the Widget container instead.
		"""

		#margin container a bit detached from left side of window

		###################
		# containers
		###################

		verticalContainer = gui.VBox(width=w, height=h, margin='2px', style={'display': 'block', 'text-align':'center'} )
		# only left margin
	
		# use Vbox instead of more generic Widget. easier to stack. when hz stacking is needed, create HBox
		
		##############################
		# menu
		##############################
		hh = 30

		self.menu = gui.Menu(width='100%', height=hh, style = { 'font-weight' : 'bold', 'text-align' : 'center', 'font-size' : '10px'})
	
		# heigth size of colored background. 
		self.menu.style['color'] = 'white'

		self.m3 = gui.MenuItem('system', width=w/3, height = '100%')
		self.m2 = gui.MenuItem('guide', width=w/3, height='100%')
		self.m1 = gui.MenuItem('about', width=w/3, height= '100%')
		# create from m1 at left
		# heigh is highlight whe hoovering 
		# 100% means highligth is height of menu
		# default is highlight is around text
		# best is same size as menu
		# color is text
		# 20 = '20px'
		self.menu.append([self.m1, self.m2, self.m3])

		self.menubar = gui.MenuBar(width='100%', height=hh, style ={'color':'red'})
		self.menubar.attributes['title'] = 'play bach'

		self.menubar.append([self.menu])

		# heigh is place the menu bar will take in container. could be larger that menu item. if so some white space below colored menu
		# style= {'color' : 'red' } is the same as   self.menubar.style['color'] = 'red'
		# if alone, will be in the middle. as other widget are defined, the 1st one will move to the top
	
		# For a reference list of HTML attributes, you can refer to https://www.w3schools.com/tags/ref_attributes.asp

		# For a reference list of style attributes, you can refer to https://www.w3schools.com/cssref/default.asp
		#self.menubar.style['color'] = 'red'


		####################
		# menu call back
		####################
		self.m1.onclick.connect(self.on_about)
		self.m2.onclick.connect(self.on_guide)
		self.m3.onclick.connect(self.on_more)

		

		####################
		# main title. static
		####################
		# set width in % so that it resize
		# will always be written on top regardless of size, unless margin managed
		self.title = gui.Label(title,  width = '100%', 
		style = { 'font-weight' : 'bold', 'text-align' : 'center', 'font-size' : '30px'} \
		)
		self.title.style['color'] = 'red'

		#self.title.style['margin-top'] = '0px'
		#self.title.style['margin-bottom'] = '0px'

		
		##############################
		# to print status line. bottom of GUI 
		# need to be defined before dropdown, as it is referenced there
		##############################

		self.label_output = gui.Label('output area', margin='0px', height=80, width='100%' ,
		style = { 'color' : self.colors['gray'], 'font-weight' : 'bold', 'text-align' : 'center', 'font-size' : '15px'} )

		
		##############################
		# drop down
		##############################

		# drop down for instrument
		# manage heigh vs font size
		# font 15px, use height = 30 to have some room around

		ww = 20
		hh = 20
		fs = '15px'
		
		# instrument dropdown
		self.instrument_label = gui.Label('Instrument',  height = hh , margin = '2px', style={'text-align': 'left', 'font-size' : fs})
		self.instrument_label.style['color'] = 'gray'
		self.instrument_label.style['width']= '30%'

		self.dropDown_instrument = gui.DropDown.new_from_list(instrument_name_list, \
		style = {  'color' : self.colors['blue'], 'font-weight' : 'bold', 'text-align' : 'center', 'font-size' : fs} )
		#self.dropDown_instrument.style['margin-left']= '50px'
		self.dropDown_instrument.style['width']= '50%'
		self.dropDown_instrument.select_by_value(instrument)
		self.dropDown_instrument.attributes['title'] = 'select instrument' # show when hoovering
		# pass label as parameter 
		self.dropDown_instrument.onchange.connect(self.on_drop_instrument, self.label_output) # pass label to be written as argument to call back

		self.instrument = gui.HBox(width=w) 
		#self.instrument.set_layout_orientation(Widget.LAYOUT_HORIZONTAL)
		
		self.instrument.append(self.instrument_label,key="")
		self.instrument.append(self.dropDown_instrument,key="")

		# drop down for corpus
		self.corpus_label = gui.Label('Corpus',  height = hh , margin = '2px', style={'text-align': 'left', 'font-size' : fs})
		self.corpus_label.style['color'] = 'gray'
		self.corpus_label.style['width']= '30%'

		self.dropDown_corpus = gui.DropDown.new_from_list(corpus_list, \
		style = { 'color' : self.colors['blue'], 'font-weight' : 'bold', 'text-align' : 'center', 'font-size' : fs} )
		#self.dropDown_corpus.style['margin-left']= '50px'
		self.dropDown_corpus.style['width']= '50%'
		self.dropDown_corpus.select_by_value(corpus)
		self.dropDown_corpus.attributes['title'] = 'select corpus' # show when hoovering
		# pass label as parameter
		self.dropDown_corpus.onchange.connect(self.on_drop_corpus, self.label_output)

		self.corpus = gui.HBox(width=w) 
		#self.instrument.set_layout_orientation(Widget.LAYOUT_HORIZONTAL)
		
		self.corpus.append(self.corpus_label,key="")
		self.corpus.append(self.dropDown_corpus,key="")

		
		##############################
		# slider
		##############################
		# create an hz container to include two widget side by side
		# because the main container is VBox

		hh=15

		# slider, temperature in %
		self.temp_label = gui.Label('Temp', height = hh , margin = '0px', style={'text-align': 'left', 'font-size' : '15px'})
		self.temp= gui.Slider(temperature,0.1,3,0.1, height=hh, margin='0px')
		self.temp.onchange.connect(self.on_temp)
		self.temp_label.set_style({'color':'gray'})
		self.temp.style['width']= '60%'
		self.temp_label.style['width']= '30%'

		self.temp_c = gui.HBox(width=w) 
		self.temp_c.attributes['title'] = 'select temperature' # show when hoovering
		#self.temp_c.set_layout_orientation(Widget.LAYOUT_HORIZONTAL)
		
		k1 = self.temp_c.append(self.temp_label,key="")
		k2 = self.temp_c.append(self.temp,key="")

		# slider, BPM
		self.BPM_label = gui.Label('BPM', height = hh ,  margin = '2px', style={'text-align': 'left', 'front-size' : '15px'})
		self.BPM= gui.Slider(BPM,40,240,5, height=hh, margin='2px')
		self.BPM.onchange.connect(self.on_BPM)
		self.BPM_label.set_style({'color':'gray'})
		self.BPM.style['width']= '60%'
		self.BPM_label.style['width']= '30%'

		self.BPM_c = gui.HBox(width=w)
		self.BPM_c.attributes['title'] = 'select BPM' # show when hoovering
		self.BPM_c.set_layout_orientation(Widget.LAYOUT_HORIZONTAL)
		
		self.BPM_c.append(self.BPM_label,key="")
		self.BPM_c.append(self.BPM,key="")


		
		##############################
		# links
		##############################
		
		#base_url = 'http://127.0.0.1:' + str(data['flask_port']) 
		base_url = data['dns'] + str(data['flask_port']) 
		stream_url = base_url + '/play'
		ping_url = base_url + '/ping'
		audio_url = base_url + '/audio'
		dashboard_url = base_url + '/dashboard'

		wl = w / 8
		hl=30
		style = { 'color' : 'green', 'font-weight' : 'bold', 'text-align' : 'left', 'font-size' : '20px'}

		self.play = gui.Link(stream_url, "PLAY", open_new_window=True, width=wl, height=hl, margin='0px', \
		style = style )
		self.play.attributes['title'] = 'to stream' # show when hoovering

		style = { 'color' : 'green', 'font-weight' : 'bold', 'text-align' : 'left', 'font-size' : '12px'}

		self.ping = gui.Link(ping_url, "server test", open_new_window=True, width=wl, height=hl, margin='0px', \
		style = style )
		self.ping.attributes['title'] = 'to test' # show when hoovering

		self.audio = gui.Link(audio_url, "audio test", open_new_window=True, width=wl, height=hl, margin='0px', \
		style = style )
		self.audio.attributes['title'] = 'to test audio on server' # show when hoovering

		self.dashboard = gui.Link(dashboard_url, "dashboard", open_new_window=True, width=wl, height=hl, margin='0px', \
		style = style )
		self.dashboard.attributes['title'] = 'monitor' # show when hoovering

		# put all links in horizontal box
		self.links = gui.HBox(width=w, height =30, margin='0px', style={'position': 'relative','display': 'block', 'overflow': 'hidden', 'text-align':'left'})
		self.links.append([self.dashboard, self.audio, self.ping])


		##############################
		# build UI
		##############################
		verticalContainer.append(self.menubar)

		verticalContainer.append(self.title) 
		verticalContainer.append(self.instrument) # instrument
		verticalContainer.append(self.corpus) # corpus
		verticalContainer.append(self.temp_c) # use hz container for label and slider
		verticalContainer.append(self.BPM_c) # 
		verticalContainer.append(self.play) # link
		verticalContainer.append(self.links) # use hz box for 3 links
		verticalContainer.append(self.label_output)
		
		return(verticalContainer)

	
	#############################################
	# call back for menu
	#############################################

	def on_about(self,widget):
		self.label_output.set_text('Meaudre Robotics. Rev 1.3. pboudalier@gmail.com')

	def on_more(self,widget):
		# create dialog vs static output string
	
		print('REMI: display more')
		self.more = gui.GenericDialog(title= 'see more by clicking on links below:', message = '', \
		initial_value = 'links' , width=w, height = h)

		# show link
		s = 'https://pboudalier.medium.com/play-bach-let-a-neural-network-play-for-you-part-1-596e54b1c912'
		self.medium = gui.Link(s, 'medium', open_new_window=True, width='100%', height=30, margin='0px', \
		style = { 'color' : 'blue', 'font-weight' : 'bold', 'text-align' : 'left', 'font-size' : '20px'} )

		s = 'https://github.com/pabou38/play_bach'
		self.git = gui.Link(s, 'github', open_new_window=True, width='100%', height=30, margin='0px', \
		style = { 'color' : 'blue', 'font-weight' : 'bold', 'text-align' : 'left', 'font-size' : '20px'} )

		# make sure the links are in dialog
		self.more.add_field('titi', self.medium)
		self.more.add_field('titu', self.git)

		print('show generic dialog')
		self.more.show(self)
		# no need to define call back for OK or cancel

	def on_guide(self,widget):

		guide = "music generated by deep neural network. \
		\n\n- Select instrument. \
		\n- Select corpus used for training. \
		\n\n- Increase 'temperature' to add randomness in notes generation. \
		\n- Change tempo, ie 'speed' of the music, with BPM (beats per minute).\
		\n\n- Play will stream music to your browser.\
		\n- server test will ping the the server. \
		\n- audio test will test audio on the server by playing some notes. \
		\n- dashboard will display activity on web server used to stream music. \
		\n\nif you change any setting while playing, you will have to wait a bit to hear the effect, due to buffering."

		print('display guide. create GenericDialog')

		self.dialog = gui.GenericDialog(title= 'quick guide', message = 'click OK or cancel to return', \
		initial_value = 'not used' , width=w, height = h)

		# widgets within generic dialog

		# dialog consist of one text input. multiple lines to get nl
		self.dtextinput = gui.TextInput(width=w, height=300, single_line=False)
		# write guide
		self.dtextinput.set_value(guide)

		#self.dtextinput.set_on_confirm_dialog.do(self.guide_confirm)
		#self.dtextinput.set_on_cancel_dialog.do(self.guide_confirm)

		# I suspect only input widget can be used there
		# add without label to get more space vs add_field_with_label
		# add this widget to generic dialog
		self.dialog.add_field('dtextinput', self.dtextinput)

		#########################################
		#  for some reason  does not work and not needed on windows
		#########################################
		#self.dialog.confirm_dialog(self.guide_confirm)

		print('show generic dialog')
		self.dialog.show(self)

	# get value from GenericDialog
	def guide_confirm(self, widget):
		print('guide input dialog was confirmed. just return to main')
		# just return to main, after displaying guide
		# no input widget

		"""
		result = self.dialog.get_field('dtextinput').get_value()
		result = self.dialog.get_field('dcheck').get_value()
		"""



	#############################################
	# call back for dropdown and sliders
	#############################################

	# set instrument as global string
	# write to passed widget label
	def on_drop_instrument(self, widget, value, label):
		global instrument
		print('REMI: drop down: ', value)
		instrument = value # string
		# write instrument to specific line
		s = ('instrument set to %s. takes a while if already playing' %instrument)
		self.label_output.set_text(s) # use dynamically in main
		

	# set corpus as global string
	# corpus is current corpus

	# model is loaded at start of mail. based on config parameters.
	# model will be reloaded by flask handler for /play before running inference from scratch

	def on_drop_corpus(self, widget, value, label):
		global corpus, load_new_corpus
		print('REMI:drop down: ', value)

		if value == corpus:
			pass # current, no change
			print('REMI: no need to load corpus ', value)
		else:
			corpus = value # string
			print('REMI: need to LOAD new corpus/model %s' %corpus)
			load_new_corpus = True # global inter module. read in handler for /play
			# set to true when corpus is changed in the GUI. checked in handler for /play, there load new h5 model , corpus.pick
			
			s = ('hit PLAY to load new corpus: %s' %corpus)
			self.label_output.set_text(s)
			# will stop current predict thread, load new model, corpus and restart from seed

			""" 
			# cannot wait here; load only happen when /play is loaded
			while load_new_corpus :
				# set to false in main when h5 model and corpus.pick are loaded
				sleep(1)
			print('new corpus loaded')
			self.label_output.set_text('new corpus %s loaded. reload /play url'%corpus)
			"""


	# set temperature as global float
	# temp = 0 means do  not use temp
	# temp and slider 0 to 1
	def on_temp(self, widget, value):
		global temperature
		print('REMI: slider temp: ', value)
		#temperature = float(value) / 100.0
		temperature = float(value)
		# write temperature to output area
		s = ('temperature set to %s. takes a while if already playing' %str(temperature))
		self.label_output.set_text(s)


	# set BPM as global string
	def on_BPM(self, widget, value):
		global BPM
		print('REMI: slider BPM: ', value)
		BPM = value
		# write BPM to output area
		s = ('beat per minute set to %s. takes a while if already playing' %str(BPM))
		self.label_output.set_text(s)

	# no call back for links

#start(remi_app, address='0.0.0.0', port=5010, multiple_instance=False, enable_file_cache=False, update_interval=0.1, start_browser=True, standalone=False)


def start_remi_app(port): # call by remi thread in main
	print('REMI: bach GUI server on port ', port)
	start(remi_app, username=data['user'], password=data['password'], address='0.0.0.0', port=port, multiple_instance=False, enable_file_cache=False, update_interval=0.1, start_browser=True, standalone=False)

if __name__ == "__main__":
	print('starting REMI in standalone test mode')
	try:
		start(remi_app, address='0.0.0.0', port=5012, multiple_instance=False, enable_file_cache=True, update_interval=0.1, start_browser=True, standalone=False)
		#start_remi_app(data['remi_port'])
	except Exception as e:
		print('cannot start REMI ' , str(e))
		sys.exit(1)




