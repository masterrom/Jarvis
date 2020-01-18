# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from sercurity_module import Sercurity
# from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template, jsonify, make_response, request, url_for, redirect
import threading
import argparse
import datetime
# import imutils
import cv2
import time
import numpy as np

import firebase_admin
from firebase_admin import credentials, firestore
import json

cred = credentials.Certificate('/Users/vaishvik/Desktop/uoftHacks2020/supervisor-f2f29-firebase-adminsdk-l2twy-ae836f2735.json')

default_app = firebase_admin.initialize_app(cred)

db = firestore.client()

# hardcoded
userID = 'NH17KayNX5dm0nlnPhklw3gzN7i2'



# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup

class Camera(object):
    thread = {}  # background thread that reads frames from camera
    frame = {}  # current frame is stored here by background thread
    last_access = {}  # time of last client access to the camera
    event = {}

    def __init__(self, camera_type=None, device=None):
        """Start the background camera thread if it isn't running yet."""
        self.unique_name = "{cam}_{dev}".format(cam=camera_type, dev=device)
        BaseCamera.event[self.unique_name] = CameraEvent()
        if self.unique_name not in BaseCamera.thread:
            BaseCamera.thread[self.unique_name] = None
        if BaseCamera.thread[self.unique_name] is None:
            BaseCamera.last_access[self.unique_name] = time.time()

            # start background frame thread
            BaseCamera.thread[self.unique_name] = threading.Thread(target=self._thread,
                                                                   args=(self.unique_name,))
            BaseCamera.thread[self.unique_name].start()

            # wait until frames are available
            while self.get_frame() is None:
                time.sleep(0)

    def get_frame(self):
        """Return the current camera frame."""
        BaseCamera.last_access[self.unique_name] = time.time()

        # wait for a signal from the camera thread
        BaseCamera.event[self.unique_name].wait()
        BaseCamera.event[self.unique_name].clear()

        return BaseCamera.frame[self.unique_name]

    @staticmethod
    def frames():
        """"Generator that returns frames from the camera."""
        raise RuntimeError('Must be implemented by subclasses')

    @classmethod
    def _thread(cls, unique_name):
        """Camera background thread."""
        print('Starting camera thread')
        frames_iterator = cls.frames()
        for frame in frames_iterator:
            BaseCamera.frame[unique_name] = frame
            BaseCamera.event[unique_name].set()  # send signal to clients
            time.sleep(0)

            # if there hasn't been any clients asking for frames in
            # the last 5 seconds then stop the thread
            if time.time() - BaseCamera.last_access[unique_name] > 5:
                frames_iterator.close()
                print('Stopping camera thread due to inactivity')
                break
        BaseCamera.thread[unique_name] = None

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

@app.route("/settings")
def setting():
	# return the rendered template
	return render_template("settings.html")

@app.route("/landingpage")
def landingpage():
	# return the rendered template
	return render_template("landingpage.html")

@app.route("/profile")
def profile():
	# return the rendered template
	return render_template("profile.html")

@app.route("/registration")
def registration():
	# return the rendered template
	return render_template("registration.html")

def detect_motion(frameCount, datasets_path, vs):
	# grab global references to the video stream, output frame, and
	# lock variables
	global outputFrame, lock

	sr = Sercurity(datasets_path)
	sr.load_known_face()
	total = 0
	# loop over frames from the video stream
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		new_frame = None
		frames = []
		for v in vs:
			_, frame = v.read()
			frames.append(cv2.resize(frame, (400, 400)))
		new_frame = np.concatenate((frames[0], frames[1]), axis=1)
		if total % 50 == 0:
			tlables, v_scores = sr.detect_labels(new_frame)
			dangers, danger_scores = sr.analyzer(tlables, v_scores)
			print(dangers, danger_scores)
		mo, frame_marked = sr.detect_and_show(new_frame, total, frameCount)
		if mo: 
			#[(),()]
			#locations = sr.face_detection(frame)
			face_locations, names = sr.recongnize(new_frame)	
			frame_marked = sr.display_result(frame_marked, face_locations, names)


		# acquire the lock, set the output frame, and release the
		# lock
		# print(frame)
		total += 1
		with lock:
			outputFrame = frame_marked.copy()
		
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')



@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/_view_log", methods=['POST'])
def view_log():
	doc_ref = db.collection(u'Camera').document(u'camera1')
	doc = doc_ref.get()
	print(u'Document data: {}'.format(doc.to_dict()))
	result = doc.get('Log')
	print(u'Document data: {}'.format(result))
	return jsonify(result)

@app.route("/_view_logii", methods=['POST'])
def view_logii():
	doc_ref = db.collection(u'Camera').document(u'camera2')
	doc = doc_ref.get()
	print(u'Document data: {}'.format(doc.to_dict()))
	result = doc.get('Log')
	print(u'Document data: {}'.format(result))
	return jsonify(result)

@app.route("/_view_logiii", methods=['POST'])
def view_logiii():
	doc_ref = db.collection(u'Camera').document(u'camera3')
	doc = doc_ref.get()
	print(u'Document data: {}'.format(doc.to_dict()))
	result = doc.get('Log')
	print(u'Document data: {}'.format(result))
	return jsonify(result)

@app.route("/_update_Toddler", methods=['POST'])
def update_Toddler():
	clicked=request.form['data']
	doc_ref = db.collection(u'users').document(userID)
	doc_ref.update({'Toddler': clicked})

	return jsonify("Success")


@app.route("/_update_Sharp", methods=['POST'])
def update_Sharp():
	clicked=request.form['data']
	doc_ref = db.collection(u'users').document(userID)
	doc_ref.update({'Sharp': clicked})

	return jsonify("Success")

@app.route("/_update_Shoes", methods=['POST'])
def update_Shoes():
	clicked=request.form['data']
	doc_ref = db.collection(u'users').document(userID)
	doc_ref.update({'Shoes': clicked})

	return jsonify("Success")

@app.route("/_update_Dirt", methods=['POST'])
def update_Dirt():
	clicked=request.form['data']
	doc_ref = db.collection(u'users').document(userID)
	doc_ref.update({'Dirt': clicked})

	return jsonify("Success")
	
@app.route("/_update_SMStwilio", methods=['POST'])
def update_SMStwilio():
	clicked=request.form['data']
	doc_ref = db.collection(u'users').document(userID)
	doc_ref.update({'SMStwilio': clicked})

	return jsonify("Success")


# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())
	#vs1 = VideoStream(src=4).start()
	#time.sleep(2.0)

	#vs2 = VideoStream(src=0).start()
	#time.sleep(2.0)

	vs1 = cv2.VideoCapture(0)
	vs2 = cv2.VideoCapture(1)

	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],'./datasets', [vs1,vs2]))
	t.daemon = True
	t.start()

	#t = threading.Thread(target=detect_motion, args=(
	#	args["frame_count"],'./datasets', vs2))
	#t.daemon = True
	#t.start()
	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)
	# app.run(host="138.51.9.170", port=args["port"], debug=True,
	# 	threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()
