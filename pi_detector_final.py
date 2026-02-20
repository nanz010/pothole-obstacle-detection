#!/usr/bin/env python3
"""
Road Hazard Detection for Raspberry Pi 4
Detects: Potholes, People, Moving Obstacles
Uses: OpenCV only (no YOLO)
"""

import cv2
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Detector")

class DangerLevel:
    SAFE, LOW, MEDIUM, HIGH = 0, 1, 2, 3
    
    @staticmethod
    def get_color(level):
        return [(0,255,0), (0,255,255), (0,165,255), (0,0,255)][level]
    
    @staticmethod
    def get_label(level):
        return ["SAFE", "LOW", "MED", "HIGH"][level]

class Detector:
    def __init__(self):
        self.min_area, self.max_area = 500, 50000
        self.prev_frame = None
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(500, 16, False)
        self.potholes, self.people, self.obstacles = [], [], []
        self.total_p, self.total_pe, self.total_o = 0, 0, 0
        self.cooldown = 3.0
    
    def detect_potholes(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi_mask = np.zeros(gray.shape, np.uint8)
        roi_start = int(h * 0.5)
        roi_mask[roi_start:, :] = 255
        
        if self.prev_frame is not None:
            diff = cv2.absdiff(gray, self.prev_frame)
            motion_mask = (diff < 25).astype(np.uint8) * 255
        else:
            motion_mask = np.ones(gray.shape, np.uint8) * 255
        self.prev_frame = gray.copy()
        
        blurred = cv2.GaussianBlur(gray, (7,7), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
        thresh = cv2.bitwise_and(thresh, roi_mask)
        thresh = cv2.bitwise_and(thresh, motion_mask)
        
        kernel = np.ones((5,5), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                (x,y), r = cv2.minEnclosingCircle(cnt)
                c = (int(x), int(y))
                r = int(r)
                
                if c[1] < roi_start + 30 or c[0] < 50 or c[0] > w-50:
                    continue
                
                mask = np.zeros(gray.shape, np.uint8)
                cv2.circle(mask, c, r, 255, -1)
                mean_val = cv2.mean(gray, mask=mask)[0]
                
                if 20 < mean_val < 120:
                    is_dup = any(np.sqrt((c[0]-d['c'][0])**2+(c[1]-d['c'][1])**2)<80 and time.time()-d['t']<self.cooldown for d in self.potholes)
                    if not is_dup:
                        danger = DangerLevel.HIGH if area>15000 else DangerLevel.MEDIUM if area>8000 else DangerLevel.LOW
                        self.total_p += 1
                        self.potholes.append({'c':c, 'r':r, 'd':danger, 't':time.time(), 'id':self.total_p})
                        logger.info(f"POTHOLE #{self.total_p} - {DangerLevel.get_label(danger)}")
        
        self.potholes = [d for d in self.potholes if time.time()-d['t']<4.0]
    
    def detect_people(self, frame):
        small = cv2.resize(frame, (320,240))
        try:
            boxes, _ = self.hog.detectMultiScale(small, winStride=(4,4), padding=(8,8), scale=1.05, hitThreshold=0.5)
            for (x,y,w,h) in boxes:
                x = int(x * frame.shape[1] / 320)
                y = int(y * frame.shape[0] / 240)
                w = int(w * frame.shape[1] / 320)
                h = int(h * frame.shape[0] / 240)
                c = (x+w//2, y+h//2)
                
                is_dup = any(np.sqrt((c[0]-d['c'][0])**2+(c[1]-d['c'][1])**2)<100 and time.time()-d['t']<self.cooldown for d in self.people)
                if not is_dup:
                    self.total_pe += 1
                    self.people.append({'b':(x,y,x+w,y+h), 'c':c, 't':time.time(), 'id':self.total_pe})
                    logger.info(f"PERSON #{self.total_pe}")
            self.people = [d for d in self.people if time.time()-d['t']<3.0]
        except:
            pass
    
    def detect_obstacles(self, frame):
        fg = self.bg_sub.apply(frame)
        
        # Stronger morphology to reduce noise and get tighter contours
        kernel_small = np.ones((3,3), np.uint8)
        kernel_med = np.ones((5,5), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel_small, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_med, iterations=1)
        
        # Erode slightly to get tighter fit
        fg = cv2.erode(fg, kernel_small, iterations=1)
        
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = frame.shape[:2]
        edge_margin = int(w * 0.2)  # Exclude 20% from edges (trees)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Tighter size range for actual obstacles (people, animals, vehicles)
            if 800 < area < 20000:
                # Get tight bounding box
                x,y,bw,bh = cv2.boundingRect(cnt)
                
                # Additional padding reduction for tighter fit
                padding = 5
                x = max(0, x + padding)
                y = max(0, y + padding)
                bw = max(10, bw - 2*padding)
                bh = max(10, bh - 2*padding)
                
                # Skip if at edges (likely trees/background)
                if x < edge_margin or x+bw > w-edge_margin:
                    continue
                
                # Skip if in bottom 30% (road/potholes area)
                if y > h*0.7:
                    continue
                
                # Skip if in top 10% (sky/distant trees)
                if y < h*0.1:
                    continue
                
                # Skip if too tall (likely trees/poles)
                if bh > h*0.5:
                    continue
                
                # Skip if too wide (likely background/trees)
                if bw > w*0.4:
                    continue
                
                # Skip if aspect ratio suggests tree (very tall/thin)
                aspect = bh / (bw + 1)
                if aspect > 2.5:
                    continue
                
                # Skip if too flat (likely shadows)
                if aspect < 0.3:
                    continue
                
                # Calculate density (how much of bounding box is filled)
                mask_roi = fg[y:y+bh, x:x+bw]
                if mask_roi.size > 0:
                    density = np.count_nonzero(mask_roi) / mask_roi.size
                    # Skip if too sparse (likely noise/trees)
                    if density < 0.15:
                        continue
                
                c = (x+bw//2, y+bh//2)
                is_dup = any(np.sqrt((c[0]-d['c'][0])**2+(c[1]-d['c'][1])**2)<80 and time.time()-d['t']<self.cooldown for d in self.obstacles)
                if not is_dup:
                    self.total_o += 1
                    self.obstacles.append({'b':(x,y,x+bw,y+bh), 'c':c, 't':time.time(), 'id':self.total_o})
                    logger.info(f"OBSTACLE #{self.total_o}")
        
        self.obstacles = [d for d in self.obstacles if time.time()-d['t']<2.5]
    
    def draw(self, frame):
        t = time.time()
        for d in self.potholes:
            alpha = max(0.3, 1.0-(t-d['t'])/4.0)
            col = tuple(int(c*alpha) for c in DangerLevel.get_color(d['d']))
            cv2.circle(frame, d['c'], d['r'], col, 3)
            cv2.putText(frame, f"P#{d['id']}", (d['c'][0]-20, d['c'][1]-d['r']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        
        for d in self.people:
            alpha = max(0.3, 1.0-(t-d['t'])/3.0)
            col = tuple(int(c*alpha) for c in DangerLevel.get_color(DangerLevel.HIGH))
            x1,y1,x2,y2 = d['b']
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, 3)
            cv2.putText(frame, f"PERSON #{d['id']}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        
        for d in self.obstacles:
            alpha = max(0.3, 1.0-(t-d['t'])/2.5)
            col = tuple(int(c*alpha) for c in DangerLevel.get_color(DangerLevel.MEDIUM))
            x1,y1,x2,y2 = d['b']
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
            cv2.putText(frame, f"OBS #{d['id']}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        
        cv2.rectangle(frame, (10,10), (350,110), (0,0,0), -1)
        y = 25
        for txt in [f"Raspberry Pi | OpenCV Only", 
                    f"Potholes: {len(self.potholes)} | Total: {self.total_p}",
                    f"People: {len(self.people)} | Total: {self.total_pe}",
                    f"Obstacles: {len(self.obstacles)} | Total: {self.total_o}",
                    "Press 'q' to quit"]:
            cv2.putText(frame, txt, (15,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            y += 20
        return frame

def main():
    logger.info("="*60)
    logger.info("Road Hazard Detection - Raspberry Pi")
    logger.info("="*60)
    
    det = Detector()
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        logger.error("Camera not found!")
        return 1
    
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    logger.info("Starting detection...")
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    
    fc, fps_start, fps = 0, time.time(), 0
    
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue
            
            det.detect_potholes(frame)
            if fc % 3 == 0:
                det.detect_people(frame)
            det.detect_obstacles(frame)
            
            frame = det.draw(frame)
            
            fc += 1
            if fc % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                fps_start = time.time()
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (550,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imshow('Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        logger.info("Stopped")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        logger.info(f"Potholes: {det.total_p} | People: {det.total_pe} | Obstacles: {det.total_o}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
