'''
Configuration Task
==================

Takes a video as input and generates / saves to disk an object file as output which contains
all the configuration information.

This configuration and features are used in the next task, ImageProcessingTask to track 
the microcontroller and detect LEDs

Usage:
        ConfigTask.py [<video_source>]

Keys:
        <Space Bar> - Pause the video
        c           - Clear all the marked ROI rectangles
        s           - Save the configuration file to the disk
        <Esc>       - Stop the program
'''

import numpy as np
import cv2
import pickle
import video
from collections import namedtuple
# import common

class RectSelector:
    def __init__(self, win):
        self.win = win
        cv2.setMouseCallback(win, self.onmouse)
        self.drag_start = None
        self.drag_rect = None
        self.rectangles= []
        self.circles= []
        self.cNames= []
        self.tx0,self.ty0,self.tx1,self.ty1= 0,0,0,0
        self.counter= 0
    def onmouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y]) # BUG
        #ch = cv2.waitKey(1)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        if self.drag_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                #print(flags, cv2.EVENT_FLAG_LBUTTON)
                xo, yo = self.drag_start
                x0, y0 = np.minimum([xo, yo], [x, y])
                x1, y1 = np.maximum([xo, yo], [x, y])
                self.drag_rect = None
                if x1-x0 > 0 and y1-y0 > 0:
                    self.drag_rect = (x0, y0, x1, y1)
                    #self.rectangles.append([x0, y0, x1, y1])
        if event== cv2.EVENT_LBUTTONDBLCLK:
            x, y = np.int16([x, y])
            self.circles.append([x, y])
            tempName= "LED: "+ str(len(self.circles))
            self.cNames.append(tempName)

    def draw(self, vis):
        if not self.drag_rect:
            for rect in self.rectangles:
                x0,y0,x1,y1= rect
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            for i in range(len(self.circles)):
                x, y= self.circles[i]
                cv2.circle(vis, (x, y), 7, (255,0,0), 2)
                cv2.putText(vis, self.cNames[i], (x-15, y-13),  cv2.FONT_HERSHEY_PLAIN, 1.0, (25,0,225), 2)
            return False

        x0, y0, x1, y1= self.drag_rect
        if self.tx0== x0 and self.ty0== y0 and self.tx1== x1 and self.ty1== y1:
            self.counter+= 1
            if self.counter== 150 and [x0,y0,x1,y1] not in self.rectangles:
                # Rectangle i sappended as ROI if it is unchanged for 150 frames (not in dragging position)
                self.rectangles.append([x0,y0,x1,y1])
                print("Added the rectangle with ("+str(x0)+","+str(y0)+"), ("+str(x1)+","+str(y1)+") as diagonal points to 'Region Of Interest!'")
                self.counter= 0
        else:
            self.counter= 0

        self.tx0,self.ty0,self.tx1,self.ty1= x0,y0,x1,y1
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)

        for rect in self.rectangles:
            x0,y0,x1,y1= rect
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        for i in range(len(self.circles)):
            x, y= self.circles[i]
            cv2.circle(vis, (x, y), 7, (255,0,0), 2)
            cv2.putText(vis, self.cNames[i], (x-15, y-13),  cv2.FONT_HERSHEY_PLAIN, 1.0, (25,0,225), 2)

        return True
    @property
    def dragging(self):
        return self.drag_rect is not None




PlanarTarget = namedtuple('PlanarTarget', 'rect, keypoints, descrs, data')

class FeatureDetector:
    def __init__(self, callback):
        self.detector = cv2.ORB( nfeatures = 1000)
        self.callback= callback

    def extract_features(self, image, rects, data=None):
        all_rects_points, all_rects_descs, all_rects= [], [], rects
        for rect in rects:
            x0, y0, x1, y1 = rect
            raw_points, raw_descrs = self.detect_features(image)

            points, descs = [], []

            for kp, desc in zip(raw_points, raw_descrs):
                x, y = kp.pt
                if x0 <= x <= x1 and y0 <= y <= y1:
                    points.append(kp)
                    descs.append(desc)

            all_rects_points.append(points)
            all_rects_descs.append(descs)

        self.callback(all_rects_points, all_rects_descs, all_rects)


    def detect_features(self, frame):
        '''detect_features(self, frame) -> keypoints, descrs'''
        keypoints, descrs = self.detector.detectAndCompute(frame, None)
        if descrs is None:  # detectAndCompute returns descs=None if not keypoints found
            descrs = []
        return keypoints, descrs


class ConfigApp:
    def __init__(self, src):
        self.cap = video.create_capture(src)
        self.frame = None
        self.paused = False
        #self.tracker = PlaneTracker()

        cv2.namedWindow('plane')
        self.rect_sel = RectSelector('plane')
        self.feat_det= FeatureDetector(self.save_data)

    def run(self):
        while True:
            playing = not self.paused and not self.rect_sel.dragging
            if playing or self.frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame = frame.copy()

            vis = self.frame.copy()
    
            self.rect_sel.draw(vis)
            cv2.imshow('plane', vis)

            ch = cv2.waitKey(1)
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.rect_sel.rectangles= []
                self.rect_sel.circles= []
                print("Cleared all marked Rectangles & Circles from 'Region Of Interest'!")
            if ch == ord('s'):
                #save to .obj file
                self.feat_det.extract_features(self.frame, self.rect_sel.rectangles)                  
            if ch == 27:
                break

    def save_data(self, all_rects_points, all_rects_descs, all_rects):
        file_object=  open("outputFile", "wb")
        #print(type(points), type(descs), type(rect))
        all_index= []

        for points in all_rects_points:
            index= []
            for point in points:
                temp= (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
                index.append(temp)
            all_index.append(index)

        pickle.dump([all_index, all_rects_descs, all_rects], file_object)
        file_object.close()
        print("Successfully saved selected ROIs to 'outputFile'")

if __name__ == '__main__':
    print __doc__
    import sys
    try: video_src = sys.argv[1]
    except: video_src = 0
    ConfigApp(video_src).run()
