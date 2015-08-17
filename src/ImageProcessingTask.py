'''
Image Processing Task
=====================

Usage:
        ImageProcessingTask.py [<saved/object/file> [<video_source>]]

Keys:
        <Space Bar> - Pause the video
        <Esc>       - Stop the program

------------------------------------------------------------------------------------

'''

import numpy as np
import time
import cv2
import pickle
import video
from collections import namedtuple
import colorsys

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

MIN_MATCH_COUNT = 10

'''
  image     - image to track
  rect      - tracked rectangle (x1, y1, x2, y2)
  keypoints - keypoints detected inside rect
  descrs    - their descriptors
  data      - some user-provided data
'''
PlanarTarget = namedtuple('PlaneTarget', 'rect, keypoints, descrs, data')

'''
  target - reference to PlanarTarget
  p0     - matched points coords in target image
  p1     - matched points coords in input frame
  H      - homography matrix from p0 to p1
  quad   - target bounary quad in input frame
'''
TrackedTarget = namedtuple('TrackedTarget', 'target, p0, p1, H, quad')

class PlaneTracker:
    def __init__(self):
        self.detector = cv2.ORB( nfeatures = 1000 )
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
        self.targets = []
        self.user_res= []
        self.ROI_type= 0
        #self.all_circles
        #self.all_cNames
        #self.all_cRadiuses
        self.all_circles_new= []

    def load_data(self, file_name, data=None):
        try:
            input_file= open(file_name, "r")
            #print(file_name)
        except:
            print("Unable to open the file- "+file_name+". Please re-run the program.")
        [all_index, all_rects_descs, all_rects, [self.all_circles, self.all_cRadiuses, self.all_cNames], self.user_res, self.ROI_type]= pickle.load(input_file)

        for i in range(len(all_rects)):
            index= all_index[i]
            descs= all_rects_descs[i]
            rect= all_rects[i]

            points= []
            for point in index:
                temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
                points.append(temp)

            descs = np.uint8(descs)
            self.matcher.add([descs])
            target = PlanarTarget(rect=rect, keypoints = points, descrs=descs, data=None)
            self.targets.append(target)


    def track(self, frame):
        '''Returns a list of detected TrackedTarget objects'''
        self.frame_points, self.frame_descrs = self.detect_features(frame)
        if len(self.frame_points) < MIN_MATCH_COUNT:
            return []
        matches = self.matcher.knnMatch(self.frame_descrs, k = 2)
        matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
        if len(matches) < MIN_MATCH_COUNT:
            return []
        matches_by_id = [[] for _ in xrange(len(self.targets))]
        for m in matches:
            matches_by_id[m.imgIdx].append(m)
        tracked = []
        for imgIdx, matches in enumerate(matches_by_id):
            if len(matches) < MIN_MATCH_COUNT:
                continue
            target = self.targets[imgIdx]
            p0 = [target.keypoints[m.trainIdx].pt for m in matches]
            p1 = [self.frame_points[m.queryIdx].pt for m in matches]
            p0, p1 = np.float32((p0, p1))
            H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
            status = status.ravel() != 0
            if status.sum() < MIN_MATCH_COUNT:
                continue
            p0, p1 = p0[status], p1[status]

            if self.ROI_type== 0:
                x0, y0, x1, y1 = target.rect
                quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            elif self.ROI_type== 1:
                #x0, y0, x1, y1 = target.rect
                #quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])         #for method 1 or method 2
                quad = np.float32(target.rect)

            quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)

            #transforming saved led positions to new positions!
            self.all_circles_new= []
            for circleI in range(len(self.all_circles)):
                new_point= cv2.perspectiveTransform(np.float32(self.all_circles[circleI]).reshape(1, -1, 2), H).reshape(-1,2)
                #print(self.all_circles[circleI], point)
                self.all_circles_new.append(new_point)

            track = TrackedTarget(target=target, p0=p0, p1=p1, H=H, quad=quad)
            tracked.append(track)
        tracked.sort(key = lambda t: len(t.p0), reverse=True)
        return tracked

    def detect_features(self, frame):
        '''detect_features(self, frame) -> keypoints, descrs'''
        keypoints, descrs = self.detector.detectAndCompute(frame, None)
        if descrs is None:  # detectAndCompute returns descs=None if not keypoints found
            descrs = []
        return keypoints, descrs

class ledApp:
    def __init__(self):
        #data of all the leds
        self.statuses= None     #False= off; True= on
        self.colors= None
        self.frequencies= None
        self.frame= None
        self.blur= None
        self.gray= None
        self.thresholded= None
        self.names= None
        self.radiuses= None

    def starter(self, frame, circles, names, radiuses):
        #start process with remaining functions
        #print(len(circles))
        self.frame= frame
        self.blur = cv2.GaussianBlur(self.frame,(5,5),0)
        self.gray= cv2.cvtColor(self.blur, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("plane2", self.gray)
        #cv2.imshow("blur", self.blur)

        self.radiuses= radiuses
        self.names= names
        self.statuses= []
        self.colors= []
        self.frequencies= []

        for i in range(len(circles)):
            print
            print("circle: "+ names[i])
            temp_status, temp_color= self.get_status_color(circles[i][0], self.radiuses[i], self.names[i])
            print(temp_status, temp_color)



    def get_status_color(self, circle, radius, name):
        ret= []
        x, y= circle
        y,x= int(x), int(y)

        #status detector
        '''change with respectively to radius obtained'''
        th, self.thresholded= cv2.threshold(self.gray, 240, 255, cv2.THRESH_BINARY)
        #cv2.imshow("plane3", self.thresholded)

        area_sum= sum(sum(self.thresholded[x-3:x+4, y-3:y+4])) #max= 7*255= 1785        #threshold value = 240
        #print(self.thresholded[x-3:x+4, y-3:y+4])
        print(area_sum)

        if(area_sum>= 1000):
            ret.append(True)
        else:
            ret.append(False)

        

        #color detector         values in BGR format
        #print(ret[0])
        if(ret[0]== True):      #only if the status is on!
            rgb_small= self.blur[x-radius-2:x+radius+3, y-radius-2:y+radius+3]
            threshold_small= self.thresholded[x-radius-2:x+radius+3, y-radius-2:y+radius+3]
            hsv_small= cv2.cvtColor(rgb_small, cv2.COLOR_BGR2HSV)
            #cv2.imshow(name+ " rgb", rgb_small)
            #cv2.imshow(name+ " threshold", threshold_small)
            #cv2.imshow(name+ " hsv", hsv_small)
            
            color_names= ["red", "yellow", "green", "cyan", "blue"]
            color_pixels= [0, 0, 0, 0, 0]

            for i in range(len(rgb_small)):
                for j in range(len(rgb_small[0])):
                    if threshold_small[i][j]== 255:
                        rgb_small[i][j]= ([255, 255, 255])
                        #print(hsv_small[i][j])
                    else:
                        #change it to hsv, detect color, add to array,
                        temp_h= hsv_small[i][j][0]
                        if temp_h<= 15 or temp_h> 135:
                            # red
                            color_pixels[0]+= 1
                        elif temp_h<= 45:
                            # yellow
                            color_pixels[1]+= 1
                        elif temp_h<= 75:
                            # green
                            color_pixels[2]+= 1
                        elif temp_h<= 105:
                            # cyan
                            color_pixels[3]+= 1
                        elif temp_h<= 135:
                            # blue
                            color_pixels[4]+= 1

            ret.append(color_names[color_pixels.index(max(color_pixels))])

            cv2.imshow(name+ " rgb modified", rgb_small)
        else:
            ret.append(None)
        
        return ret

    def get_frequency(self, circle):
        return 0

class ImageProcessionApp:
    def __init__(self, file_name, src):
        self.cap = video.create_capture(src)
        self.frame = None
        self.paused = False
        self.fps= self.cap.get(5)
        self.file_name= file_name
        self.tracker = PlaneTracker()
        self.ledModifier= ledApp()

        cv2.namedWindow('plane')


    def run(self):
        self.tracker.load_data(self.file_name)

        while True:
            playing = not self.paused
            if playing or self.frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame = frame.copy()
            

            print
            print("Another frame................")
            self.frame= cv2.resize(self.frame, (self.tracker.user_res[1], self.tracker.user_res[0]))
            vis = self.frame.copy()
            
            tracked = self.tracker.track(self.frame)

            #send to ledApp to know statuses of leds
            self.ledModifier.starter(vis, self.tracker.all_circles_new, self.tracker.all_cNames, self.tracker.all_cRadiuses)
            # use the lists created in ledapp to senf to interpreter task!

            for tr in tracked:
                #print(tr.quad)
                cv2.polylines(vis, [np.int32(tr.quad)], True, (255, 255, 255), 2)
                # for (x, y) in np.int32(tr.p1):
                #     cv2.circle(vis, (x, y), 2, (255, 255, 255))
                #print(tr.circles)
            
            for i in range(len(self.tracker.all_circles_new)):
                #print(new_center)
                [x, y]= np.int32(self.tracker.all_circles_new[i][0])
                tempR= self.tracker.all_cRadiuses[i]
                cv2.circle(vis, (x, y), tempR, (255,0,0), 2)
                cv2.putText(vis, self.tracker.all_cNames[i], (x-15, y-tempR-5),  cv2.FONT_HERSHEY_PLAIN, 1.0, (25,0,225), 2)

            
            cv2.imshow('plane', vis)
            ch = cv2.waitKey(1)
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == 27:
                break


if __name__ == '__main__':
    print __doc__
    import sys
    
    try:
        file_name= sys.argv[1]
    except:
        file_name= "outputFile.p"
        #print("ERROR: Need to provide path to object file. See the usage below:\nUsage:\n\tImageProcessingTask.py <saved/object/file> [<video_source>]")
    
    try: 
        video_src = sys.argv[2]
    except: 
        video_src = 0
    
    ImageProcessionApp(file_name, video_src).run()
