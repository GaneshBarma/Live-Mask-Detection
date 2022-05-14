'''
Packages used for the model
'''
import time
import numpy as np
import cv2
import imutils
from imutils.video import VideoStream
from tensorflow import keras
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input

from source.utils import preprocess_face_frame, decode_prediction, write_bb, load_cascade_detector
mask_detection_model_path = 'models/final_mask_detection.h5'

default_face_path = 'models/haarcascade_frontalface_default.xml'
frontal_face_alt2_path = 'models/frontal_face_alt2.xml'
nose_path = 'models/Nariz.xml'
mouth_path = 'models/Mouth.xml'
eyes_path = 'models/frontalEyes35x16.xml'


#Classes that can be showed. 
class_names= ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
              'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
              'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella',
              'handbag tie','suitcase','frisbee','skis','snowboard','sports ball','kite',
              'baseball','bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
              'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
              'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
              'bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
              'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors',
              'teddy bear','hair drier','toothbrush']
              
bounding_colors = np.random.uniform(0,255,size = (len(class_names),3))

labels_dict={0:'MASK',1:'NO MASK',2:'WEAR MASK PROPERLY'}
color_dict={0:(0,255,0),1:(0,0,255),2:(0,100,100)}

#model = keras.models.load_model('models/mask_mobilenet.h5')
model = keras.models.load_model(mask_detection_model_path)
face_detector = load_cascade_detector()



face_clsfr=cv2.CascadeClassifier(default_face_path)
nose_clsfr = cv2.CascadeClassifier(nose_path)
mouth_clsfr = cv2.CascadeClassifier(mouth_path)
eyes_clsfr = cv2.CascadeClassifier(eyes_path)
frames = []

size_fact = (320,320)
mean = (0,0,0)
scalefactor = 0.004

yolo_predict = []

def output_layers(net):
    layer_names = net.getLayerNames()
    layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return layers

def video_mask_detector():
    video = VideoStream(src=0).start()
    time.sleep(1.0)
    while True:
        # Capture frame-by-frame
        frame = video.read()

        frame = detect_mask_in_frame(frame)
        # Display the resulting frame
        # show the output frame
        cv2.imshow("Mask detector", frame)

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    # cleanup
    cv2.destroyAllWindows()
    video.stop()
import logging    
logging.basicConfig(filename='mask_logs.log',format='%(asctime)s %(message)s')
logger=logging.getLogger()
logger.setLevel(logging.INFO)
#def detect_face(img):
def detect_mask_in_frame(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_clsfr.detectMultiScale(img,1.3,4)
    for (x,y,w,h) in faces:
        face_img = img[y:y+h,x:x+w]
        resized = cv2.resize(face_img,(224,224))
        normalized = resized/255.0
        reshaped = np.reshape(normalized,(1,224,224,3))
#        result = mask_detection.predict(reshaped)
        result = model.predict(reshaped)
        label = np.argmax(result,axis=1)[0]
        accuracy = "{:.2f}".format(np.max(result) * 100)
        label_txt = labels_dict[label] + " " + str(accuracy) + "%"
        cv2.rectangle(img,(x,y),(x+w,y+h) ,color_dict[label],2)
        cv2.rectangle(img,(x,y-30),(x+w,y),color_dict[label],-1)
        cv2.putText(img,label_txt,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX ,0.6,(255,255,255),2)
        dict={0:'MASK',1:'NO MASK'}
        logger.info("Label of Img :- %d with accuracy %f i.e Status : %s",label,float(accuracy),dict[label])
    return img
  
if __name__ == '__main__':
    video_mask_detector()

