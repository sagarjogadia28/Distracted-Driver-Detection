from settings import *
from test_image import *
from train_model import *

app = Flask(__name__)

api = Api(app)
app.permanent_session_lifetime = timedelta(seconds=8)
c = ['safe driving', 'texting - right', 'talking on the phone - right', 'texting - left', 'talking on the phone - left', 'operating the radio', 'drinking',  'reaching behind', 'hair and makeup', 'looking around']

@app.after_request
def after_request(response):
	response.headers.add('Access-Control-Allow-Origin', '*')
	response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
	response.headers.add('Access-Control-Allow-Methods', 'GET, PUT, POST, DELETE')
	return response


@api.resource("/")
class Root(Resource):
	"""docstring for Test"""

	def get(self):
		output = "It's working"
		return output


@api.resource("/test")
class Classify(Resource):
	#flipped_result = [0, 3, 4, 1, 2, 5, 6, 7, 8, 9]
	def get(self):
		global initialized
		if initialized:
			return jsonify({"message" : "Image classifier is ready"})
		else:
			return jsonify({"message" : "Image classifier is not ready yet. Try again later."})

	def post(self):
		global initialized

		flipped_result = [0, 3, 4, 1, 2, 5, 6, 7, 8, 9]
		data = request.get_json()
		image_data = data["image_data"]
		filename = data["filename"]
		direction = data["direction"]
		width = data["width"]
		height = data["height"]

		image_data_decoded = base64.b64decode(image_data)
		nparr = np.fromstring(image_data_decoded, dtype=np.uint8)
		img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

		# cv2.imwrite(filename, img)

		# resize
		img_cols, img_rows = 224, 224
		if img is None:
			nparr.resize((height, width), refcheck=False)
			img = cv2.resize(nparr, (img_cols, img_rows))
			print (img)
		else:
			img = cv2.resize(img, (img_cols, img_rows))

		# sbuf = StringIO()
		# sbuf.write(image_data_decoded)
		# pimg = Image.open(sbuf)
		# d = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
		# flip = False
		if direction == "right":
			img = flip_image(img)
		cv2.imwrite(filename, img)

		# number of models
		start = 1
		end = 3

		#try:
		answer = test_model_and_submit(img, start, end, img_rows, img_cols, filename)
		if type(answer) == type('string'):
			return jsonify({"filename" : filename, "distraction_type" : -1, "message" : answer})
		else:
			if direction == "right":
				answer = flipped_result[answer]
			
			print (c[answer], filename)
			return jsonify({"filename" : filename, "distraction_type" : answer})
		#except:
		#	return jsonify({"error": "There occured some error while testing image"})

@api.resource("/train")
class Train(Resource):

	def get(self):
		return jsonify({"message" : "Train is working. But, you may want to send POST request!"})

	def post(self):
		#data = request.get_json()
		#image_data = data["image_data"]
		#filename = data["filename"]
		#direction = data["direction"]

		#decode_image(image_data)
		#if direction == "right":
		#	flip_image()
		try:
			accuracy = train_model(8, 6, 0.15, '_vgg_16_2x20')
			return jsonify({"message": "The model is trained successfully with accuracy of" + accuracy})
		except:
			return jsonify({"message" : "Error occured while training the model. Try again later"})


def flip_image(img):
	return cv2.flip(img, 1) 


# def  test_image(img, img_cols, img_rows):
#	return False, ''



if __name__ == '__main__':

	port = int(os.environ.get('PORT', 8000))
	app.run(host='0.0.0.0', port=port, debug=False)
	gc.collect()
	
