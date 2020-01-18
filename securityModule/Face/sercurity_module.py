import face_recognition
import cv2
import numpy as np
import os
import sys
import string
import random
import imutils
import datetime
from google.cloud import vision
class Sercurity:
    def __init__(self, datasets_path, accumWeight=0.5):
        self.datasets_path = datasets_path
        self.known_face_encodings = []
        self.known_face_names = []
       	# store the accumulated weight factor
        self.accumWeight = accumWeight
        # initialize the background model
        self.bg = None

    def update(self, image):
        # if the background model is None, initialize it
        if self.bg is None:
        	self.bg = image.copy().astype("float")
        	return

        # update the background model by accumulating the weighted
        # average
        cv2.accumulateWeighted(image, self.bg, self.accumWeight)

    def load_known_face(self):
        for filename in os.listdir(self.datasets_path):
            if 'jpg' in filename or 'png' in filename:
                face_path = os.path.join(self.datasets_path, filename)
                # Load face from datasets
                face = face_recognition.load_image_file(face_path)
                face_encoding = face_recognition.face_encodings(face)[0]
                self.known_face_encodings.append(face_encoding)
                face_name = filename.replace('_', '.').split('.')[0]
                self.known_face_names.append(face_name)
                print(face_name)

    # inputs:
    # locations will be a list of tuple which contains the location of each face[(),()]
    # img will be a 3 dimentional img matrix
    # return:
    # list of face face_encodings
    def extract_face(self,locations, img):
        return locations.face_encoding(img, locations) 

    def randomString(self, stringLength=10):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(stringLength))

    def face_detection(self, image):
        client = vision.ImageAnnotatorClient()
        imageByte = vision.types.Image(content=cv2.imencode('.jpg', image)[1].tostring())

        response = client.label_detection(image=imageByte)
        labels = response.label_annotations
        print('Labels:')
        print
        for label in labels:
            print(label.description)
            
        return faceBounds
        

    def add_new_face_to_datasets(self, face_img, face_encoding):
        confirm = input('Unkonw face detected. Do you want add it to dataset?(Y/N)')
        if confirm == 'Y' or confirm == 'y':
            new_face_name = input('Name:')
            new_face_path = os.path.join(self.datasets_path, new_face_name+'_'+randomString(10)+'.jpg')
 
            cv2.imwrite(new_face_path, face_img) 
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(new_face_name)
            print('New image saved!')

    def crop_face(self, frame, face_location):
        top = face_location[0]*4
        right = face_location[1]*4
        bottom = face_location[2]*4
        left = face_location[3]*4

        face = frame[top:bottom, left:right]
        return face

    def shrink_frame(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        return small_frame[:, :, ::-1]

    def recongnize(self, frame, locations):
        locations = locations
        rgb_small_frame = self.shrink_frame(frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, locations)

        name_result = []
        for i, face_encoding in enumerate(face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = 'Unknown' 
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            name_result.append(name)
        return name_result

    def display_result(self, frame, locations, names):
        frame1 = frame
        # Display the results
        for (top, right, bottom, left), name in zip(locations, names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame1, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame1, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame1, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        return frame1
 
    def motion_detect(self, image, tVal=25):

           # compute the absolute difference between the background model                           
           # and the image passed in, then threshold the delta image                                
           delta = cv2.absdiff(self.bg.astype("uint8"), image)
           thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]                           
           
           # perform a series of erosions and dilations to remove small                             
           # blobs
           thresh = cv2.erode(thresh, None, iterations=2)
           thresh = cv2.dilate(thresh, None, iterations=2)                                          
           
           # find contours in the thresholded image and initialize the                              
           # minimum and maximum bounding box regions for motion
           cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,                                
                   cv2.CHAIN_APPROX_SIMPLE)
           cnts = imutils.grab_contours(cnts)                                                       
           (minX, minY) = (np.inf, np.inf)
           (maxX, maxY) = (-np.inf, -np.inf)                                                        
           
           # if no contours were found, return None                                                 
           if len(cnts) == 0:
                   return None                                                                      
           
           # otherwise, loop over the contours                                                      
           for c in cnts:
                   # compute the bounding box of the contour and use it to                          
                   # update the minimum and maximum bounding box regions                            
                   (x, y, w, h) = cv2.boundingRect(c)
                   (minX, minY) = (min(minX, x), min(minY, y)) 
                   (maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))                              
           
           # otherwise, return a tuple of the thresholded image along                               
           # with bounding box
           return (thresh, (minX, minY, maxX, maxY))

    def detect_and_show(self, frame, total, frameCount):
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        mo = False
        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        # if the total number of frames has reached a sufficient
        # number to construct a reasonable background model, then
        # continue to process the frame
        if total > frameCount:
            # detect motion in the image
            motion = self.motion_detect(gray)
            # cehck to see if motion was found in the frame
            if motion is not None:
            # unpack the tuple and draw the box surrounding the
            # "motion area" on the output frame
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY),
                    (0, 0, 255), 2)
                mo = True
        # update the background model and increment the total number
        # of frames read thus far
        self.update(gray)
        return mo, frame

