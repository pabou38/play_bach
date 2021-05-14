#!/usr/bin/env  python

from flask import Flask
import flask_monitoringdashboard as dashboard
import time

app = Flask(__name__)
dashboard.config.init_from(file='./dashboard.conf')

dashboard.bind(app)

@app.route('/')
def hello_world():
    return 'Flask-Monitoring-Dashboard tutorial'

@app.route('/endpoint1')
def endpoint1():
    time.sleep(0.20)
    return 'Endpoint1', 400

@app.route('/endpoint2')
def endpoint2():
    time.sleep(5)
    return 'Endpoint2'

app.run(host="0.0.0.0", port=5001, debug=False)