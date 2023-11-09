import cv2
import csv
import numpy as np
import pybboxes as pbx
from utils.csvReader import read as csvRead
def annotate(file, gazeData, boundingBoxes):

    boundingBoxReader = csvRead(boundingBoxes)
    array_of_bounding_boxes = np.array(boundingBoxReader)

    # Read the video
    cap = cv2.VideoCapture(file)

    split = file.split('/')
    fileName = split[-1]

    output = cv2.VideoWriter('annotated_' + fileName, cv2.VideoWriter_fourcc(*'mp4v'), 30, (3840, 1920))

    if (cap.isOpened()== False):
        print("Error opening video stream or file")


    while(True):
        ret, frame = cap.read()
        if ret == True:

            # Print the current frame number
            print("Frame: " + str(cap.get(cv2.CAP_PROP_POS_FRAMES)))

            # Get the gaze data for the current frame
            array_of_lists = np.array(gazeData)
            filtered_list = array_of_lists[array_of_lists[:,1] == str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))]
            filtered_list.tolist()

            #get bounding box coordinates for this frame
            filtered_list_of_BB = array_of_bounding_boxes[array_of_bounding_boxes[:,0] == str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))]
            filtered_list_of_BB.tolist()
            print(filtered_list_of_BB)

            #annote bounding boxes in xywh format
            for box in filtered_list_of_BB:
                #remove non numeric characters from coordinate strings
                box[1] = ''.join(char for char in box[1] if char.isdigit() or char == '.') #x
                box[2] = ''.join(char for char in box[2] if char.isdigit() or char == '.') #y
                box[3] = ''.join(char for char in box[3] if char.isdigit() or char == '.') #w
                box[4] = ''.join(char for char in box[4] if char.isdigit() or char == '.') #h
                box[5] = ''.join(char for char in box[5] if char.isdigit() or char == '.') #id
                box[5] = 'ID: ' + box[5]
                convertedBB = int(float(box[1])), int(float(box[2])), int(float(box[3])), int(float(box[4])), box[5]
                #convertedBB = (float(box[1]), float(box[2]), float(box[3]), float(box[4]))
                #convertedBB = pbx.convert_bbox(yoloBB, from_type='yolo', to_type='voc', image_size=(3840, 1920))
                #draw rectanges with converted xyxy format
                cv2.rectangle(frame, (convertedBB[0], convertedBB[1]), (convertedBB[2], convertedBB[3]), (0, 255, 0), 2)
                cv2.putText(frame, box[5], (convertedBB[0], convertedBB[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            

            # Annotate the frame
            for gazePoint in filtered_list:
                cv2.circle(frame, (int(float(gazePoint[2])), int(float(gazePoint[3]))), 10, (0, 0, 255), -1) #image, center_coordinates, radius, color, thickness

            # Write the frame to the output video
            output.write(frame)
        else:
            break

    # Release the capture and the output
    cap.release()
    output.release()
    