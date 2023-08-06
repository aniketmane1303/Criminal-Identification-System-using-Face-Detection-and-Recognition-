

from twilio.rest import Client
import numpy as np
import threading
import datetime
import pickle
import time
import cv2
import os
import imutils
from imutils.video import VideoStream
from twilio.rest import Client
def  recognition():
  class Activities:
      def __init__(self):
          # Initialize Twilio Client to send whatsapp message
          sid ='AC391245de560b66bb16e100a29323433c'
          authToken ='43c0d29f4036a52408b3f02b7f745356'
          self.client = Client(sid , authToken)
          self.details_json = {
              "aniket": {
                  "timestamp": 0,
                  "captures": 0,
                  "total_capture": 0,
                  "image_sent": False
              },
              "suraj": {
                  "timestamp": 0,
                  "captures": 0,
                  "total_capture": 0,
                  "image_sent": False
              },
              "unknown": {
                  "timestamp": 0,
                  "captures": 0,
                  "total_capture": 0,
                  "image_sent": False
              },
              "pd mundada": {
                  "timestamp": 0,
                  "captures": 0,
                  "total_capture": 0,
                  "image_sent": False
              },
              "tushar": {
                  "timestamp": 0,
                  "captures": 0,
                  "total_capture": 0,
                  "image_sent": False
              },
              "shreeyash": {
                  "timestamp": 0,
                  "captures": 0,
                  "total_capture": 0,
                  "image_sent": False
              },
              "nitin gavankar": {
                  "timestamp": 0,
                  "captures": 0,
                  "total_capture": 0,
                  "image_sent": False
              },
              "abhishek": {
                  "timestamp": 0,
                  "captures": 0,
                  "total_capture": 0,
                  "image_sent": False
              },

          }

          self.FROM_NUMBER = "whatsapp:+14155238886"
          self.TO_NUMBER = "whatsapp:+919307302391"
          self.min_confidence = 0.55
          self.max_captures = 1
          self.min_time_gap = 480
          self.show_video = True

        # Load OpenCV’s Caffe-based deep learning face detector model
          print("[EXEC] Loading face detector...")
          self.detector = cv2.dnn.readNetFromCaffe(
              "DNN_module/deploy.prototxt.txt",
		  	"DNN_module/res10_300x300_ssd_iter_140000.caffemodel")

        # Load our face embeddings
          print("[EXEC] loading face embeddings...")
          self.embedder = cv2.dnn.readNetFromTorch(
              "openface_nn4.small2.v1.t7")

        # Load the recogniser model
          print("[EXEC] loading face recognizer...")
          self.recognizer = pickle.loads(open("serialized_files/SVM_recognizer.pickle", "rb").read())

        # Load the Label encoder
          self.le = pickle.loads(open("serialized_files/label_encoder.pickle", "rb").read())

    # save the captured frame in a file
      def store_frame(self, get_name, op_frame):
          p = os.path.sep.join(["motions_caught", "{}.png".format(
              str(get_name).capitalize())])
          cv2.imwrite(p, op_frame)
          print("Whatsapp message sent")

    # send whatsapp message to specific number via TWILIO API
      def send_message(self, get_name, timestamp):
          self.client.messages.create(
              body='Criminal Name:🦹‍♂️ {} \nDate & Time:\n {}'.format(
                  str(get_name).upper(),
                  str(timestamp.strftime("%d %B %Y %I:%M%p"))),
              from_=self.FROM_NUMBER,
              to=self.TO_NUMBER
          )
          print("Frame has been saved ")


# initialse a activity object
  activity = Activities()

# Load videostream
  print("[EXEC] starting video stream...")
  cap = cv2.VideoCapture(0)
# Let the camera sensor warm-up
  time.sleep(1.0)

  frame_avg = None
  last_seen = datetime.datetime.now()

  while True:
    # Read frames
      ret, frame = cap.read()
    # Current timestamp
      curr_timestamp = datetime.datetime.now()

    # Frame resize
      frame = cv2.resize(frame, dsize=(750, 600))
    # Height and width of the frame
      (h, w) = frame.shape[:2]

    # Pre-process image by Mean subtraction, Resize and
    # scaling by some factor
      imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                        1.0, (300, 300),
                                        (104.0, 177.0, 123.0),
                                        swapRB=False, crop=False)
      activity.detector.setInput(imageBlob)
    # Detect possible face detection in image with the detector model
      detections = activity.detector.forward()

    # Now loop over each detection
      for i in range(0, detections.shape[2]):
          confidence = detections[0, 0, i, 2]
        # Proceed if detected face confidence is above min_confidence
          if confidence > activity.min_confidence:
              box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
              (startX, startY, endX, endY) = box.astype("int")
            # Extract the ROI (region of interest)
              face = frame[startY:endY, startX:endX]
              (fH, fW) = face.shape[:2]
            # Make sure ROI is sufficiently large
              if fW < 20 or fH < 20:
                  continue
            # Now we pre-process the our ROI i.e face detected
              faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                               (96, 96), (0, 0, 0),
                                               swapRB=True, crop=False)
            # Use embedder model to extract 128-d face embeddings
              activity.embedder.setInput(faceBlob)
              vec = activity.embedder.forward()

            # Now make predictions based on our recognizer model
              preds = activity.recognizer.predict_proba(vec)[0]
            # store the index prediction maximum probability
              j = np.argmax(preds)
            # store the prediction maximum probability
              proba = preds[j]
            # extract the name of the prediction
              name = activity.le.classes_[j]

              text = "{}: {:.2f}%".format(name, proba * 100)
              y = startY - 10 if startY - 10 > 10 else startY + 10
            # Draw bounding box
              cv2.rectangle(frame, (startX, startY), (endX, endY),
                            (0, 0, 255), 2)
              cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                          0.45, (0, 0, 255), 2)
              ts = curr_timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
              text = "criminal detected"
              cv2.putText(frame, "{}".format(text), (10, 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
              cv2.putText(frame, ts, (10, frame.shape[0] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

              if name == "unknown":
                  continue
                # For first frame recog or appear after 8 mins
                # send whatsapp notification
              elif activity.details_json[name]["captures"] == 0 or \
                      (activity.details_json[name]['timestamp'] -
                       curr_timestamp).total_seconds() >= \
                      activity.min_time_gap:
                # start new thread and call store_frame function

                # to keep track of captured
                  activity.details_json[name]["captures"] = \
                      activity.details_json[name]["captures"] + 1
                  threading.Thread(target=activity.send_message(
                      get_name=name, timestamp=curr_timestamp)).start()
                  threading.Thread(target=activity.store_frame(
                      op_frame=frame, get_name=name)).start()

                # change the timestamp when the frame was captured
                  activity.details_json[name]['timestamp'] = curr_timestamp

    # show the current videostram or not
      if activity.show_video:
        # display the security feed
          cv2.imshow("Output", frame)
          key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
          if key == ord("q"):
              break
  cap.release()
  cv2.destroyAllWindows()


