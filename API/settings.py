from flask import Flask, jsonify, request, make_response
from flask_restful import Api, Resource
import os
from datetime import timedelta
import json
import urllib3
import base64


import warnings
warnings.filterwarnings("ignore")
urllib3.disable_warnings()

# Base = declarative_base()

#app = Flask(__name__)

#api = Api(app)
#app.permanent_session_lifetime = timedelta(seconds=8)
