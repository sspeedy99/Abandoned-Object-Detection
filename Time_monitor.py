import numpy as np
import cv2
import sys
from collections import Counter, defaultdict

#Classification Libraies
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import model_from_json
import json


with open('model_in_json.json','r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights('model_weights.h5')

test_data_dir = 'Test_dir/'


def Classification():
    img_width, img_height = 224, 224
    batch_size=2

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_data = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

    test_data.reset()


    yhat = model.predict_generator(test_data,  verbose=1)
    y_classes = yhat.argmax(axis=-1)
    val = {0: '1', 1: '2', 2: '3'}

    final = y_classes[0]
    threat_val = val[final]
    return threat_val

   # print("The threat classification is: ", val[final])

 
# location of first frame
# firstframe_path ='frame0.jpg'
# firstframe = cv2.imread(firstframe_path)
# firstframe = cv2.resize(firstframe, (640,480), interpolation = cv2.INTER_AREA) 
# #print(firstframe.shape)
# #Preprocessning for edge detection
# firstframe_gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
# firstframe_blur = cv2.GaussianBlur(firstframe_gray,(21,21),0)
#Background subtracion
backSub1 = cv2.createBackgroundSubtractorMOG2()

#---------------------------------
#size the window first
#---------------------------------
#cv2.namedWindow('CannyEdgeDet',cv2.WINDOW_NORMAL)
cv2.namedWindow('Abandoned Object Detection',cv2.WINDOW_NORMAL)
#cv2.namedWindow('Morph_CLOSE',cv2.WINDOW_NORMAL) # Closing MT - It is obtained by the dilation of an image followed by an erosion.

# location of video
# print(sys.argv[1])
# file_path ='video1.avi'
file_path = sys.argv[1]

cap = cv2.VideoCapture(file_path)
success, frame0 = cap.read()
#print(frame0.shape)
if success:
    cv2.imwrite('final.jpg',frame0)

firstframe_path ='final.jpg'
firstframe = cv2.imread(firstframe_path)
#firstframe = cv2.resize(firstframe, (640,480), interpolation = cv2.INTER_AREA) 


consecutiveframe=20
flag = 1

track_temp=[]
track_master=[] #Track the num of centroid with its frame number.
track_temp2=[]

top_contour_dict = defaultdict(int)
obj_detected_dict = defaultdict(int)

frameno = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    #cv2.imshow('main',frame)
    #print(frame.shape)
    frame_temp = frame
    
    if ret==0:
        break
    
    frameno = frameno + 1
    #cv2.putText(frame,'%s%.f'%('Frameno:',frameno), (400,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray,(21,21),0)
    frame_diff = cv2.absdiff(firstframe, frame)
  
    #Canny Edge Detection
    edged = cv2.Canny(frame_diff,100,200) #any gradient between 30 and 150 are considered edges
    #cv2.imshow('CannyEdgeDet',edged)
    kernel2 = np.ones((5,5),np.uint8) #higher the kernel, eg (10,10), more will be eroded or dilated
    thresh2 = cv2.morphologyEx(edged,cv2.MORPH_CLOSE, kernel2,iterations=10)
    #cv2.imshow('Morph_Close', thresh2)
    
    #Create a copy of the thresh to find contours    
    (_,cnts, _) = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    mycnts =[] # every new frame, set to empty list. 
    # loop over the contours
    
    for c in cnts:

        #This object tracking algorithm is called centroid tracking as it relies on the Euclidean distance 
        #between (1) existing object centroids 
        #(i.e., objects the centroid tracker has already seen before) and 
        #(2) new object centroids between subsequent frames in a video
        
        # Calculate Centroid using cv2.moments
        M = cv2.moments(c)
        if M['m00'] == 0: 
            pass
        else:
            #formula for calculating centroids form image moments.
            '''Image Moment is a particular weighted average of image pixel intensities,
            with the help of which we can find some specific properties of an image, like radius, area, centroid etc.'''
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])


            #----------------------------------------------------------------
            # Set contour criteria
            #----------------------------------------------------------------
            
            if cv2.contourArea(c) < 500 or cv2.contourArea(c)>20000:
                pass
            else:
                mycnts.append(c)
                  
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                #Storing the coordinates of the contours to draw the rectange in the frame. 
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #Euclidean Distance between the centroids
                #cv2.putText(frame,'C %s,%s,%.0f'%(cx,cy,cx+cy), (cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2) 
                sumcxcy=cx+cy
                #Add the sum of centroid and its respective frame number to the tracking list
                track_temp.append([cx+cy,frameno])
                track_master.append([sumcxcy,frameno])

                #Get the unique framenumbers
                countuniqueframe = set(j for i, j in track_master)

                '''
                Monitoring the objects by the time they are left attended.
                Video is compiled with 30 frames/sec, so after the no of unique frames passes the consecutive frame i,e 30
                remove the earliest frame, that will be the frame of least frame no, labeled as miniframeno.
                Removing the miniframeno will be tough so it would be easy if we create a new list, and the frames to the new list
                '''
                if len(countuniqueframe)>consecutiveframe or False: 
                    minframeno=min(j for i, j in track_master)
                    for i, j in track_master:
                        if j != minframeno: # get a new list. omit the those with the minframeno
                            track_temp2.append([i,j])
                
                    track_master=list(track_temp2) # transfer to the master list
                    track_temp2=[]
                

                '''
                Storing the imagemoments count to track if there are any changes in the backgound,
                same sumcxcy means there is a stationary object present, we are making a counter to montior this, all count of sumcxcy will be 1
                '''
                countcxcy = Counter(i for i, j in track_master)
                #print countcxcy
                #example countcxcy : Counter({544: 1, 537: 1, 530: 1, 523: 1, 516: 1})
                #if j which is the count occurs in all the frame, store the sumcxcy in dictionary, add 1
                for i,j in countcxcy.items(): 
                    if j>=consecutiveframe:
                        top_contour_dict[i] += 1
                
                if sumcxcy in top_contour_dict:
                    #Checking for 100, if it is left for some time(approx 3 sec, 30fps, 100 = 3.33 sec)
                    if top_contour_dict[sumcxcy]>35:
                        if flag:
                            cv2.imwrite("DetectedFrame.jpg",frame)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        #cv2.putText(frame,'%s'%('Marked'), (cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                        if flag:
                            frame =frame[y:y+h,x:x+w]
                            cv2.imwrite("./Test_dir/TestIm/MarkedFrame.jpg",frame)
                            threat = Classification()
                            flag = 0
                        print ('Detected : ', sumcxcy,frameno, obj_detected_dict)
                        threat_classified = "Threat Found: Classified as {}".format(threat)
                        cv2.putText(frame,'%s'%(threat_classified), (cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                        
                        # Store those objects that are detected, and store the last frame that it happened.
                        # Need to find a way to clean the top_contour_dict, else contour will be detected after the 
                        # object is removed because the value is still in the dict.
                        # Method is to record the last frame that the object is detected with the Current Frame (frameno)
                        # if Current Frame - Last Frame detected > some big number say 100 x 3, then it means that 
                        # object may have been removed because it has not been detected for 100x3 frames.
                        
                        #Add the frame number of the detected frame with its moments values
                        obj_detected_dict[sumcxcy]=frameno


    for i, j in obj_detected_dict.items():
        if frameno - obj_detected_dict[i]>200:
            print ('PopBefore',i, obj_detected_dict[i],frameno,obj_detected_dict)
            print ('PopBefore : top_contour :',top_contour_dict)
            obj_detected_dict.pop(i)
                                    
            # Set the count for eg 448 to zero. because it has not be 'activated' for 200 frames. Likely, to have been removed.
            top_contour_dict[i]=0
            print ('PopAfter',i, obj_detected_dict[i],frameno,obj_detected_dict)
            print ('PopAfter : top_contour :',top_contour_dict)

                
                

    
    cv2.imshow('Abandoned Object Detection',frame)
    
         
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
cap.release()
cv2.destroyAllWindows()

# img_width, img_height = 224, 224
# batch_size=2

# test_datagen = ImageDataGenerator(rescale=1. / 255)

# test_data = test_datagen.flow_from_directory(
# test_data_dir,
# target_size=(img_width, img_height),
# batch_size=batch_size,
# class_mode='categorical')

# test_data.reset()


# yhat = model.predict_generator(test_data,  verbose=1)
# y_classes = yhat.argmax(axis=-1)
# val = {0: '1', 1: '2', 2: '3'}

# final = y_classes[0]

print(threat_classified)