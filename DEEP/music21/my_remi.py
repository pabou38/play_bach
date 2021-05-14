#!/usr/bin/python3

#29 jan

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

from types import coroutine
import remi.gui as gui
from remi import start, App

import os
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
tfmodel = config_bach.tfmodel # to init label

# set to true when corpus is changed in the GUI
# monitored by main 
load_new_corpus = False 

corpus_list = []
tf_list = []


# build list of existing models based on h5 files (full models)
p = os.path.join(os.getcwd(), 'models', '*.h5')

for file in glob.glob(p):
	#print (file)
	file = os.path.basename(file)
	#corpus_list.append(file.split('_')[0])
	file = re.sub('.h5', '', file)
	corpus_list.append(file)

print('my_remi.py: corpus_list ', corpus_list)

"""
celloc1_fixed-to-int.tflite 
"""

# build list of existing lite models based on .tflite files
p = os.path.join(os.getcwd(), 'models', '*.tflite')
for file in glob.glob(p):
	#print (file)
	file = os.path.basename(file)
	#file = file.split('_') [1:]  # in case multiple _ , output is list
	file = re.sub('.tfile', '', file)
	file = ''.join(file)  # convert list to string
	file = file.split('.') [0] # ignore tflite
	if file not in tf_list: # multiple corpus have same TFfile name
		tf_list.append(file)
	
tf_list.append('Full') # hardcoded. means non tflite for inference

print('my_remi.py: tf_list ', tf_list)

# size of root container, ie the screen
# try and error. does not resize with mouse
		
w=400
h=500


class remi_app(App):

	def __init__(self, *args):
		# custom style for app
		res_path = os.path.join(os.path.dirname(__file__), 'res')
		super(remi_app, self).__init__(*args, static_file_path={'res':res_path})
		# remi_app is a sub class of App

	# entry point
	def main(self):
		global instrument, temperature, BPM

		# size of root container, ie the screen
		# try and error. does not resize with mouse

		w=400
		h=500

		self.colors = {'white' : '#ffffff', 'black' : '#000000', 'gray' : '#919191', 'red':'#ff0000', 'green':'#00ff00', 'blue':'#0000ff'}

		container = gui.VBox(width=w, height =h, margin='0px auto', style={'position': 'relative','display': 'block', 'overflow': 'hidden', 'text-align':'center'})
		

		# GUI elements

		# to print status line. botton of GUI
		self.label_output = gui.Label('status', margin='1px', height='20%', width='100%' ,\
			style = { 'color' : self.colors['gray'], 'text-align' : 'left', 'font-size' : '15px'} \
			)

		#menu item

		self.m1 = gui.MenuItem('about', width = w/4)
		self.m2 = gui.MenuItem('guide',  width = w/4)
		self.m3 = gui.MenuItem('more', width = w/4)
		
		self.m1.onclick.connect(self.on_about)
		self.m2.onclick.connect(self.on_guide)
		self.m3.onclick.connect(self.on_more)

		# menu from menuitem
		self.menu = gui.Menu(width='100%')
		self.menu.append ([self.m1, self.m2, self.m3])

		# menubar from menu
		self.menubar = gui.MenuBar(width='100%', height='20%')
		self.menubar.append([self.menu])
		

		# main title. static
		# set width in % so that it resize
		self.title = gui.Label(title,  width = '100%', height='20%',  margin='5px',\
		style = { 'color' : self.colors['red'], 'font-weight' : 'bold', 'text-align' : 'center', 'font-size' : '20px'} \
		)
 

		# drop down for instrument
		# manage heigh vs font size

		# does not work to create a Hbox with a label and a drop down
	
		self.instrument_label = gui.Label('instrument', wigth = '20%', height = 15 , margin = '5px', style={'text-align': 'left', 'font-size' : '10px'})
		self.dropDown = gui.DropDown.new_from_list(instrument_name_list, widht='100%', height='15%', margin='5px', \
		style = { 'color' : self.colors['blue'], 'font-weight' : 'bold', 'text-align' : 'center', 'font-size' : '15px'} \
		)
		self.dropDown.select_by_value(instrument)
		self.dropDown.attributes['title'] = 'select instrument'
		# pass label as parameter 
		self.dropDown.onchange.connect(self.on_drop_instrument, self.label_output) # pass label to be written as argument to call back

		

		# drop down for corpus
		self.corpus_label = gui.Label('corpus', wigth = '20%', height = 15 , margin = '5px', style={'text-align': 'left', 'font-size' : '10px'})
		self.dropDown2 = gui.DropDown.new_from_list(corpus_list, widht='100%', height='15%', margin='5px', \
		style = { 'color' : self.colors['blue'], 'font-weight' : 'bold', 'text-align' : 'center', 'font-size' : '15px'} \
		)
		self.dropDown2.select_by_value(corpus)
		self.dropDown2.attributes['title'] = 'select corpus'
		# pass label as parameter
		self.dropDown2.onchange.connect(self.on_drop_corpus, self.label_output)
		

		# drop down for TFmodel type
		self.tf_label = gui.Label('tensorflow model', wigth = '20%', height = 15 , margin = '5px', style={'text-align': 'left', 'font-size' : '10px'})
		self.dropDown3 = gui.DropDown.new_from_list(tf_list, widht='100%', height='15%', margin='5px', \
		style = { 'color' : self.colors['blue'], 'font-weight' : 'bold', 'text-align' : 'center', 'font-size' : '15px'} \
		)
		self.dropDown3.select_by_value(tfmodel)
		self.dropDown3.attributes['title'] = 'select tensorflow model'
		# pass label as parameter
		self.dropDown3.onchange.connect(self.on_drop_model, self.label_output)



		# play with heigth and font size
		
		# slider, temperature in %

		self.temp_label = gui.Label('temperature', wigth = '20%', height = 15 , margin = '5px', style={'text-align': 'left', 'font-size' : '15px'})
		self.temp= gui.Slider(temperature,0,1,0.1, width = '50%' , height='10%', margin='5px')
		self.temp.onchange.connect(self.on_temp)

		# slider, BPM
		# 
		self.BPM_label = gui.Label('BPM', wigth = '20%', height = 15 ,  margin = '5px', style={'text-align': 'left', 'front-size' : '15px'})
		self.BPM= gui.Slider(BPM,40,240,5, width = '50%' , height='10%', margin='5px')
		self.BPM.onchange.connect(self.on_BPM)

		wl = w / 4
		base_url = 'http://127.0.0.1:' + str(data['flask_port']) 

		stream_url = base_url + '/play'
		ping_url = base_url + '/ping'
		audio_url = base_url + '/audio'
		dashboard_url = base_url + '/dashboard'
	

		self.play = gui.Link(stream_url, "play", open_new_window=True, width=wl, height=30, margin='1px', \
		style = { 'color' : 'blue', 'font-weight' : 'bold', 'text-align' : 'center', 'font-size' : '15px'} )

		self.ping = gui.Link(ping_url, "ping", open_new_window=True, width=wl, height=30, margin='1px', \
		style = { 'color' : 'blue', 'font-weight' : 'bold', 'text-align' : 'center', 'font-size' : '15px'} )

		self.audio = gui.Link(audio_url, "audio", open_new_window=True, width=wl, height=30, margin='1px', \
		style = { 'color' : 'blue', 'font-weight' : 'bold', 'text-align' : 'center', 'font-size' : '15px'} )

		self.dashboard = gui.Link(dashboard_url, "dashboard", open_new_window=True, width=wl, height=30, margin='1px', \
		style = { 'color' : 'blue', 'font-weight' : 'bold', 'text-align' : 'center', 'font-size' : '15px'} )

		self.links = gui.HBox(width=w, height =h, margin='0px auto', style={'position': 'relative','display': 'block', 'overflow': 'hidden', 'text-align':'center'})

		self.links.append([self.play, self.ping, self.audio, self.dashboard])


		######################################
		# build GUI layout
		######################################

		container.append(self.menubar)
		
		container.append(self.title) 
		
		container.append(self.instrument_label)
		container.append(self.dropDown) # instrument

		container.append(self.corpus_label)
		container.append(self.dropDown2) # corpus

		container.append(self.tf_label)
		container.append(self.dropDown3) # TF model

		container.append(self.temp_label) # temp
		container.append(self.temp)
		container.append(self.BPM_label) # BPM
		container.append(self.BPM)

		container.append(self.links)

		container.append(self.label_output)

		# root widget
		return(container)



	############################# 
	# call backs
	#############################

	def on_about(self,widget):
		self.label_output.set_text('version 1.3. done by pabou. ')

	def on_more(self,widget):
		# create dialog vs string
		#self.label_output.set_text('https://pboudalier.medium.com/the-new-york-times-on-epaper-every-morning-f3b3cff43f9c')
		print('REMI: display more')
		self.more = gui.GenericDialog(title= 'see more by clicking on links', message = '', \
		initial_value = 'links' , width=w, height = h)

		s = 'https://pboudalier.medium.com/the-new-york-times-on-epaper-every-morning-f3b3cff43f9c'

		self.medium = gui.Link(s, 'articles on medium', open_new_window=True, width='100%', height=30, margin='1px', \
		style = { 'color' : 'blue', 'font-weight' : 'bold', 'text-align' : 'left', 'font-size' : '20px'} )

		self.more.add_field('titi', self.medium)

		print('show generic dialog')
		self.more.show(self)

		# no ned to define call back for OK or cancel

	def on_guide(self,widget):

		guide = "Bach like music generated by deep neural network. \
		\n\n- Change instrument used to synthetize notes. \
		\n- Select corpus of music used for training. \
		\n- Select use of tensorflow full or tensorflow lite. \
		\n\n- Increase 'temperature' to add randomness in notes generation (may sound interesting, strange or weird). \
		\n- Change tempo, ie 'speed' of the music, with BPM (beat per minute).\
		\n\n- play will stream music to your browser.\
		\n- ping just test the server is active. \
		\n- audio will test audio on server. just play some notes. \
		\n- dashboard will display activity on (flask) web server. \
		\n\nif you change instrument or music type, you may have to wait a bit to hear the effect, due to buffering."

		print('display guide. create GenericDialog')

		self.dialog = gui.GenericDialog(title= 'quick guide', message = 'click OK or cancel to return', \
		initial_value = 'guide will be there' , width=w, height = h)

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




	# call back for dropdown

	# set instrument as global string
	# write to passed widget label
	def on_drop_instrument(self, widget, value, label):
		global instrument
		print('REMI: drop down: ', value)
		instrument = value # string
		# write instrument to specific line
		self.label_output.set_text(instrument) # use dynamically in main



	# set corpus as global string
	# corpus is current corpus
	def on_drop_corpus(self, widget, value, label):
		global corpus, load_new_corpus
		print('REMI:drop down: ', value)

		if value == corpus:
			pass # current, no change
		else:
			corpus = value # string
			print('REMI: LOAD new corpus %s' %corpus)
			load_new_corpus = True # global inter module. read in main
			self.label_output.set_text('corpus ', corpus)
			self.label_output.set_text('reload /play url to take effect')

	# set tfmodel gobal

	def on_drop_model(self, widget, value, label):
		global tfmodel # current model
		print('REMI: current model %s, drop down %s : ' %(tfmodel,value))
		self.label_output.set_text(tfmodel) 

		if tfmodel == value:
			return

		else:

			tfmodel = value # string
			# will be monitored in main, if not 'Full' , will do inference using TFlite and tfmodel

		
		




	# call back for sliders

	# set temperature as global float
	# temp = 0 means do  not use temp
	# temp and slider 0 to 1
	def on_temp(self, widget, value):
		global temperature
		print('REMI: slider temp: ', value)
		#temperature = float(value) / 100.0
		temperature = float(value)
		# write temperature to output area
		self.label_output.set_text('temperature: ' + str(temperature))


	# set BPM as global string
	def on_BPM(self, widget, value):
		global BPM
		print('REMI: slider BPM: ', value)
		BPM = value
		# write BPM to output area
		self.label_output.set_text('BPM: ' + str(BPM))


	

def start_remi_app(port): # call by remi thread in main
	print('REMI: bach GUI server on port ', port)
	start(remi_app, username=data['user'], password=data['password'], address='0.0.0.0', port=port, multiple_instance=False, enable_file_cache=True, update_interval=0.1, start_browser=True, standalone=False)

if __name__ == "__main__":
	
	start_remi_app(data['remi_port'])



"""
		# use TextInput to get multiline. write with set_text or set_value
		self.label_output = gui.TextInput(single_line=False, margin='10px', height=60, width = '50%' , \
		style = { 'color' : self.colors['gray'], 'text-align' : 'left', 'font-size' : '15px'} \
		)
	
	
	def on_spin(self, widget, value):
		print('spin: ', value)

	def on_listview_command (self, widget, key):
		# param is a key
		global command
		command  = widget.children[key].get_text()
		print ('listview command: ', command)

	def on_listview_action (self, widget, key):
		# param is a key
		global command
		action  = self.listView2.children[key].get_text()
		print ('listview action: ', action)

	
		self.button = gui.Button('Press!')
		# html
		self.button.attributes['title'] = 'please press me'
		self.button.style['color'] = 'red'
		# call back
		self.button.onclick.do(self.on_button,'you pressed')

		def on_button(self, widget, param1 = ''):
		self.button.set_text('ok')
		# widget is button pressed

		self.spin= gui.SpinBox(1,0,100, width = 50, height=50, margin='5px')
		self.spin.onchange.do(self.on_spin)
		
	

	
		items = ('threshold' , 'timelapse', 'snapshot')
		self.listView1 = gui.ListView.new_from_list(items, width=50, margin='10px')
		self.listView1.attributes['title'] = 'select command'
		self.listView1.onselection.connect(self.on_listview_command)
		
		items = ('read' , 'set')
		self.listView2 = gui.ListView.new_from_list(items, width=50, margin='10px')
		self.listView2.attributes['title'] = 'select action'
		self.listView2.onselection.connect(self.on_listview_action)
"""

"""
		# use TextInput to get multiline. write with set_text or set_value
		self.label_output = gui.TextInput(single_line=False, margin='10px', height=60, width = '50%' , \
		style = { 'color' : self.colors['gray'], 'text-align' : 'left', 'font-size' : '15px'} \
		)
		"""