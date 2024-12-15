"""
Created on Wed Jul  3 15:05:11 2024

@author: anash
"""

from ultralytics import YOLO
import numpy as np
import time
import matplotlib.pyplot as plt
import ast
from collections import namedtuple

Point = namedtuple('Point', 'x y')

class CNNInference():

     def __init__(self, model_name):
        
         self.model = YOLO(model_name)
        
         self.lower_arc_length = 0
         self.upper_arc_length = 0
         self.cathode_pos = [490,260]
         self.contact_pos = [560,290]
         self.contact_width = 0
         self.starter_pos = [520,240]
         self.normalise = np.sqrt((self.cathode_pos[0]-self.starter_pos[0])**2+(self.cathode_pos[1]-self.starter_pos[1])**2)
         
         self.flame_detected = 0
         self.contact_detected = 0
         self.cathode_detected = 0
         self.starter_detected = 0
         
         self.la_len_list = []
         self.ua_len_list = []
         self.dis_x_list = []
         self.dis_y_list = []
   
        
     def lower_arc(self, flame):
         cond = False
         low_arc = []
         for point in flame:
             if point[1] >= self.cathode_pos[1]:
                 cond = True
             if cond:
                 low_arc.append(point)
                 if point[0] >= self.contact_pos[0]-self.contact_width/2:
                     break
         return low_arc   
        
     def higher_arc(self, flame):
          cond = False
          cond2 = False
          i = 0
          high_arc = []
          for point in flame:
              while cond:
                  high_arc.append(flame[i])
                  if flame[i][0] >= self.contact_pos[0]+self.contact_width/2:
                      cond2 = True
                  if cond2:
                      if flame[i][0] <= self.contact_pos[0]+self.contact_width/2:
                          return high_arc
                  i -= 1
              if point.y >= self.cathode_pos[1]:
                  cond = True
              if not cond:
                  i += 1  
        
     def arc_length(self, arc):
         length = 0
         for i in range(len(arc)-1):
             length += np.sqrt((arc[i+1].x-arc[i].x)**2+(arc[i+1].y-arc[i].y)**2)
         return length
     
     def detect(self, source, show=False, conf=0.25, save=False):
        results = self.model.predict(source, conf=conf, show=show, save=save)
         
        img_list = [ast.literal_eval(img.tojson()) for img in results]
        
        self.la_point_list = []
        self.ua_point_list = []
        self.la_len_list = []
        self.ua_len_list = []
        self.dis_x_list = []
        self.dis_y_list = []
        
        for img_eval in img_list:
            
            eval_list = {}
            for class_eval in img_eval:
                name = class_eval['name']
                confidence = class_eval['confidence']
                if name not in eval_list or confidence > eval_list[name]['confidence']:
                    eval_list[name] = class_eval
                    
            eval_list = list(eval_list.values())
            class_list = [d["name"] for d in eval_list]
            
            try:
                flame_index = class_list.index("flame")
                self.flame_detected = 1
            except:
                self.flame_detected = 0
            try:
                contact_index = class_list.index("contact")
                self.contact_detected = 1
            except:
                self.contact_detected = 0
            try:
                cathode_index = class_list.index("cathode")
                self.cathode_detected = 1
            except:
                self.cathode_detected = 0
            try:
                starter_index = class_list.index("starter")
                self.starter_detected = 1
            except:
                self.starter_detected = 0
                
            if self.contact_detected:
                self.contact_pos[0] = (eval_list[contact_index]["box"]["x1"]+eval_list[contact_index]["box"]["x2"])/2
                self.contact_pos[1] = (eval_list[contact_index]["box"]["y1"]+eval_list[contact_index]["box"]["y2"])/2
                self.contact_width = eval_list[contact_index]["box"]["x2"]-eval_list[contact_index]["box"]["x1"]
                
            if self.cathode_detected:
                self.cathode_pos[0] = eval_list[cathode_index]["box"]["x2"]
                self.cathode_pos[1] = (eval_list[cathode_index]["box"]["y1"]+eval_list[cathode_index]["box"]["y2"])/2
            
            if self.starter_detected:
                self.starter_pos[0] = eval_list[starter_index]["box"]["x2"]
                self.starter_pos[1] = eval_list[starter_index]["box"]["y2"]
            
            if self.flame_detected: 
                flame_points = [Point(eval_list[flame_index]["segments"]["x"][i], eval_list[flame_index]["segments"]["y"][i]) 
                                    for i in range(len(eval_list[flame_index]["segments"]["x"]))]
                try:
                    self.lower_arc_points = self.lower_arc(flame_points)
                    self.higher_arc_points = self.higher_arc(flame_points)
                    self.lower_arc_length = self.arc_length(self.lower_arc_points)
                    self.upper_arc_length = self.arc_length(self.higher_arc_points)
                    self.la_point_list.append(self.lower_arc_points)
                    self.ua_point_list.append(self.higher_arc_points)
                    self.la_len_list.append(self.lower_arc_length)
                    self.ua_len_list.append(self.upper_arc_length)
                except:
                    self.la_point_list.append([Point(0,0),Point(0,0)])
                    self.ua_point_list.append([Point(0,0),Point(0,0)])
                    self.la_len_list.append(self.lower_arc_length)
                    self.ua_len_list.append(self.upper_arc_length)
            else:
                self.la_point_list.append([Point(0,0),Point(0,0)])
                self.ua_point_list.append([Point(0,0),Point(0,0)])
                self.la_len_list.append(self.lower_arc_length)
                self.ua_len_list.append(self.upper_arc_length)
                    
            if self.cathode_detected and self.starter_detected:    
                self.normalise = np.sqrt((self.cathode_pos[0]-self.starter_pos[0])**2+(self.cathode_pos[1]-self.starter_pos[1])**2)
            
            self.dis_x_list.append(np.abs(self.contact_pos[0]-self.cathode_pos[0]))
            self.dis_y_list.append(np.abs(self.contact_pos[1]-self.cathode_pos[1]))
            
        self.la_len_list[:] = [val / self.normalise for val in self.la_len_list]
        self.ua_len_list[:] = [val / self.normalise for val in self.ua_len_list]
        self.dis_x_list[:] = [val / self.normalise for val in self.dis_x_list]
        self.dis_y_list[:] = [val / self.normalise for val in self.dis_y_list]
                    
if __name__ == '__main__': # for testing
    clip_no = 213
    lis = list()
    for i in range(1):
        model_id = r".\runs\segment\train23\weights\best.pt"
        cnnmodel = CNNInference(model_id)      
        st = time.time()
        # run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
        cnnmodel.detect(rf".\dataset_clips\clip363.mp4", show=True, conf=0.1, save=False)
        lis.append([clip_no, cnnmodel.cathode_detected, cnnmodel.starter_detected, cnnmodel.contact_detected, cnnmodel.flame_detected])
        clip_no += 10
        et = time.time()
        print(cnnmodel.dis_x_list)    
        xpoints = [d.x for d in cnnmodel.lower_arc_points]
        ypoints = [d.y for d in cnnmodel.lower_arc_points]
        xpoints1 = [d.x for d in cnnmodel.higher_arc_points]
        ypoints1 = [d.y for d in cnnmodel.higher_arc_points]
        plt.figure(dpi=1200)
        plt.plot(xpoints,ypoints, label="Lower flame contour, Length: " + str(round(cnnmodel.lower_arc_length,3)))
        plt.plot(xpoints1,ypoints1, label="Upper flame contour, Length: " + str(round(cnnmodel.upper_arc_length,3)))
        plt.plot(cnnmodel.cathode_pos[0], cnnmodel.cathode_pos[1], 'go', label="Cathode tip")
        plt.plot(cnnmodel.contact_pos[0], cnnmodel.contact_pos[1], 'ro', label="Contact point")
        plt.gca().invert_yaxis()
        plt.legend(loc="upper left", prop={'size': 6})
        plt.xlabel("x-Coordinate")
        plt.ylabel("y-Coordinate")
        plt.title("Extracted features from the image segementation")
        plt.show()
        print(et-st)
