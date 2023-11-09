def intersect(gazePoint, boundingBoxes, graph):

    #iterate through all bounding boxes for this frame
    boxNum = 0
    for box in boundingBoxes:
        #get bounding box coordinates
        x1, y1, w, h = box.xywh[0]
        x2, y2 = x1 + w, y1 + h

        #convert to floats
        x1 = float(x1)
        x2 = float(x2)
        y1 = float(y1)
        y2 = float(y2)

        #iterate through all gaze point for this frame
        for x, y in zip(gazePoint.x, gazePoint.y):
            x = float(x)
            y = float(y)

            #check if gaze point is within bounding box
            if x1 < x and x < x2:
                if y1 < y and y < y2:
                    print("Intersection found at: " + str(x) + ", " + str(y))
                    print("Box ID: " + str(boundingBoxes[boxNum].id))
                    #graph.update(gazePoint.time, float(boundingBoxes[boxNum].id))

        #next bounding box
        boxNum += 1

    print("Time: " + str(gazePoint.time))