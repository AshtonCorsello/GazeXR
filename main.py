from ultralytics import YOLO
import pandas as pd
import os
import numpy as np
import argparse
import csv
import tempfile
import matplotlib.pyplot as plt
from utils.csvReader import read as csvRead
from utils.annotator import annotate
from utils.intersect import intersect

class GazePoint:

    def get_frame_data(self):
        array_of_lists = np.array(self.gazeData)
        filtered_list = array_of_lists[array_of_lists[:,1] == str(self.frame)]
        filtered_list.tolist()
        return filtered_list

    def __init__(self, gazeData, frame):
        self.frame = frame
        self.gazeData = gazeData
        self.gazePoints = self.get_frame_data()
        self.x = []
        self.y = []
        for point in self.gazePoints:
            self.x.append(point[2])
            self.y.append(point[3])

        self.time = float(frame/30)

class Graph:
    def __init__(self):
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        scatter = ax.scatter([], [], marker='o', label='Gaze Points')

        # Format plot
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Box ID')
        ax.set_title('Gaze Points vs. Time')
        ax.grid(True)
        ax.legend(loc='best')

    def update(self, time, boxID):
        #make color the same for each boxID
        color = 'C' + str(int(boxID))
        self.ax.scatter(time, boxID, marker='o', c=color, label='Gaze Points')
       # self.ax.scatter(time, boxID, marker='o', label='Gaze Points')
        self.fig.canvas.draw() # update plot

    def save(self):
        self.fig.savefig('gaze_points.png')



def main(args):

    #Load model and run inference on passed video
    model = YOLO("yolov8l.pt")
    result = model.track(source=args.video, persist=True, stream=True, exist_ok=True, tracker="botsort.yaml", classes=[0])

    #gazeData is a list of lists
    gazeData = csvRead(args.gaze)

    plot = Graph() #initialize plot

    #keep track of frame number (TODO: find a better way to do this)
    frame = 1
    with tempfile.NamedTemporaryFile(mode='a', delete=False, suffix='.csv') as f:
        temp_csv_name = f.name
        with open(temp_csv_name, 'a') as csvfile:
            #for r in result:
            #    boxes = r.boxes  # Boxes object for bbox outputs
            #    masks = r.masks  # Masks object for segment masks outputs
            #    probs = r.probs  # Class probabilities for classification outputs
            #    gazePoint = GazePoint(gazeData, frame)
            #    intersect(gazePoint, boxes, plot)
            #    #print(r.boxes.id.cpu().numpy())
            #    #print(r.boxes.xywhn.cpu().numpy())
            #    for b in boxes:
            #        #id = b.id.cpu().numpy()[0]
            #        #x = b.xywhn.cpu().numpy()[0][0]
            #        #y = b.xywhn.cpu().numpy()[0][1]
            #        #w = b.xywhn.cpu().numpy()[0][2]
            #        #h = b.xywhn.cpu().numpy()[0][3]
            #        print('DATA: ', b.data)
            #        x = b.data[0][0].cpu().numpy().astype(float)
            #        y = b.data[0][1].cpu().numpy().astype(float)
            #        w = b.data[0][2].cpu().numpy().astype(float)
            #        h = b.data[0][3].cpu().numpy().astype(float)
            #        id = b.data[0][4].cpu().numpy().astype(float)
                #    print('BOX: ', b)
            #        csv.writer(csvfile).writerow([frame, x, y, w, h, id])
            #    frame += 1
            for f, r in enumerate(result):
                frame_number = f
                for bbox in r.boxes.data.cpu().numpy():
                    result_dict = {
                        'frame-num': frame_number,
                        'x1': bbox[0],
                        'y1': bbox[1],
                        'x2': bbox[2],
                        'y2': bbox[3],
                        'track-ID': int(bbox[4]),
                        'confidence': bbox[5],
                    }
                    csv.writer(csvfile).writerow([frame_number, bbox[0], bbox[1], bbox[2], bbox[3], int(bbox[4]), bbox[5]])
            #send csv file to annotate
            annotate(args.video, gazeData, temp_csv_name)
            
    
    plot.save() #save plot
    
        
def test_reader():
    gazeData = csvRead('gaze_data.csv')
    annotate('runs/detect/predict/360_video_18s.avi', gazeData)


def filter_list(gazeData, frame):
    array_of_lists = np.array(gazeData)
    filtered_list = array_of_lists[array_of_lists[:,1] == str(frame)]
    filtered_list.tolist()
    return filtered_list

def argParser():
    parser = argparse.ArgumentParser(description='Data input for gaze tracking')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--gaze', type=str, help='Path to gaze data csv file')

    args = parser.parse_args()

    if not args.video:
        print('No video file provided')
        exit()

    if not args.gaze:
        print('No gaze data file provided')
        exit()
    
    if not os.path.isfile(args.video):
        print('Video file does not exist')
        exit()
    
    if not os.path.isfile(args.gaze):
        print('Gaze data file does not exist')
        exit()
    
    return args


if __name__ == "__main__" :
    args = argParser()
    main(args)
    #test_reader()
