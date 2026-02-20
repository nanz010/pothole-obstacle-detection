"""
Portable Road Hazard Detection System
Works on: Windows, Linux, Raspberry Pi 4
Transfer via: WhatsApp, USB, Email, etc.

Single file - no external dependencies needed (auto-installs)
Just run: python road_detector_portable.py

Features:
- Accurate pothole detection with false-positive filtering
- Obstacle detection (people, animals, vehicles)
- Danger-level color coding (GREEN/YELLOW/ORANGE/RED)
- Proper counting with duplicate prevention
- Auto-optimizes for platform (Pi vs Laptop)
"""

import sys
import subprocess
import os

# Auto-install dependencies
def check_and_install_dependencies():
    """Check and install required packages."""
    required = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
    }
    
    optional = {
        'ultralytics': 'ultralytics'
    }
    
    print("Checking dependencies...")
    
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    # Try to install YOLO (optional)
    try:
        __import__('ultralytics')
        print("✓ ultralytics (YOLO)")
    except ImportError:
        print("Installing ultralytics (this may take a moment)...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics'])
        except:
            print("⚠ Could not install YOLO - will run without obstacle detection")

# Run dependency check
if __name__ == "__main__":
    if '--skip-install' not in sys.argv:
        check_and_install_dependencies()

import cv2
import numpy as np
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
import json
import platform

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


def detect_platform():
    """Detect if running on Raspberry Pi or regular computer."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            if 'Raspberry Pi' in f.read():
                return 'raspberry_pi'
    except:
        pass
    return 'computer'


PLATFORM = detect_platform()
IS_PI = PLATFORM == 'raspberry_pi'

# Platform-specific optimizations
if IS_PI:
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480
    DEFAULT_FPS = 20
    YOLO_CONF = 0.4
    PROCESS_EVERY_N_FRAMES = 1
else:
    DEFAULT_WIDTH = 1280
    DEFAULT_HEIGHT = 720
    DEFAULT_FPS = 30
    YOLO_CONF = 0.35
    PROCESS_EVERY_N_FRAMES = 1

print(f"Platform detected: {PLATFORM.upper()}")
print(f"Optimized settings: {DEFAULT_WIDTH}x{DEFAULT_HEIGHT} @ {DEFAULT_FPS}fps")


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


class DangerLevel:
    """Danger level classification with color coding."""
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    
    @staticmethod
    def get_color(level):
        """Get BGR color for danger level."""
        return {
            0: (0, 255, 0),      # Green
            1: (0, 255, 255),    # Yellow
            2: (0, 165, 255),    # Orange
            3: (0, 0, 255)       # Red
        }.get(level, (255, 255, 255))
    
    @staticmethod
    def get_label(level):
        """Get text label for danger level."""
        return {0: "SAFE", 1: "LOW", 2: "MED", 3: "HIGH"}.get(level, "?")


class Detection:
    """Base detection class with unique ID."""
    _next_id = 1
    
    def __init__(self, detection_type, position, confidence, danger_level):
        self.id = Detection._next_id
        Detection._next_id += 1
        self.type = detection_type
        self.position = position
        self.confidence = confidence
        self.danger_level = danger_level
        self.timestamp = time.time()
    
    def age(self):
        return time.time() - self.timestamp
    
    def is_duplicate(self, other, threshold=80):
        if self.type != other.type:
            return False
        dist = np.sqrt((self.position[0] - other.position[0])**2 + 
                      (self.position[1] - other.position[1])**2)
        return dist < threshold


class PotholeDetection(Detection):
    def __init__(self, center, radius, area, confidence, danger_level):
        super().__init__('pothole', center, confidence, danger_level)
        self.center = center
        self.radius = radius
        self.area = area


class ObstacleDetection(Detection):
    def __init__(self, bbox, center, class_name, confidence, danger_level):
        super().__init__('obstacle', center, confidence, danger_level)
        self.bbox = bbox
        self.class_name = class_name


class RoadHazardDetector:
    """Main detection system."""
    
    def __init__(self, display=True):
        self.display = display
        
        # Detection parameters
        self.min_pothole_area = 500
        self.max_pothole_area = 50000
        
        # Active detections
        self.active_potholes = []
        self.active_obstacles = []
        self.pothole_persistence = 4.0
        self.obstacle_persistence = 2.5
        
        # Unique tracking
        self.unique_potholes = []
        self.unique_obstacles = []
        self.cooldown_time = 3.0
        
        # Counters
        self.total_pothole_count = 0
        self.total_obstacle_count = 0
        
        # Motion tracking
        self.prev_frame = None
        self.motion_threshold = 25
        
        # Load YOLO
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                model_path = 'yolov8n.pt'
                if not os.path.exists(model_path):
                    print("Downloading YOLOv8n model (first time only)...")
                self.yolo_model = YOLO(model_path)
                print(f"✓ YOLO loaded (optimized for {PLATFORM})")
            except Exception as e:
                print(f"⚠ YOLO not available: {e}")
        
        # Obstacle configuration
        self.obstacle_config = {
            0: ('person', DangerLevel.HIGH),
            15: ('cat', DangerLevel.HIGH),
            16: ('dog', DangerLevel.HIGH),
            17: ('horse', DangerLevel.HIGH),
            18: ('sheep', DangerLevel.MEDIUM),
            19: ('cow', DangerLevel.HIGH),
            20: ('elephant', DangerLevel.HIGH),
            21: ('bear', DangerLevel.HIGH),
            22: ('zebra', DangerLevel.HIGH),
            1: ('bicycle', DangerLevel.MEDIUM),
            2: ('car', DangerLevel.MEDIUM),
            3: ('motorcycle', DangerLevel.MEDIUM),
            5: ('bus', DangerLevel.MEDIUM),
            7: ('truck', DangerLevel.MEDIUM),
            14: ('bird', DangerLevel.LOW),
        }
        
        self.logger = logging.getLogger("Detector")
    
    def assess_pothole_danger(self, area):
        """Assess danger level based on pothole size."""
        if area > 15000:
            return DangerLevel.HIGH
        elif area > 8000:
            return DangerLevel.MEDIUM
        elif area > 3000:
            return DangerLevel.LOW
        return DangerLevel.SAFE
    
    def detect_potholes(self, frame):
        """Detect potholes with accurate filtering."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # ROI: bottom 50% (road area)
        roi_mask = np.zeros(gray.shape, dtype=np.uint8)
        roi_start_y = int(height * 0.5)
        roi_mask[roi_start_y:, :] = 255
        
        # Motion compensation
        if self.prev_frame is not None:
            frame_diff = cv2.absdiff(gray, self.prev_frame)
            motion_mask = frame_diff < self.motion_threshold
        else:
            motion_mask = np.ones(gray.shape, dtype=bool)
        
        self.prev_frame = gray.copy()
        
        # Detect dark regions
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 3
        )
        
        # Apply masks
        thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)
        thresh = cv2.bitwise_and(thresh, thresh, mask=motion_mask.astype(np.uint8) * 255)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        new_detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_pothole_area < area < self.max_pothole_area:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                # Filters
                if center[1] < roi_start_y + 30:
                    continue
                
                edge_margin = 50
                if (center[0] < edge_margin or center[0] > width - edge_margin or
                    center[1] > height - edge_margin):
                    continue
                
                # Intensity check
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, center, radius, 255, -1)
                mean_intensity = cv2.mean(gray, mask=mask)[0]
                
                if mean_intensity > 120 or mean_intensity < 20:
                    continue
                
                # Texture variance
                std_intensity = np.std(gray[mask > 0])
                if std_intensity < 5:
                    continue
                
                # Circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if not (0.15 < circularity < 0.95):
                        continue
                
                danger_level = self.assess_pothole_danger(area)
                detection = PotholeDetection(center, radius, area, circularity, danger_level)
                
                # Check duplicates
                is_duplicate = any(detection.is_duplicate(d) for d in self.active_potholes)
                
                if not is_duplicate:
                    is_new_unique = True
                    for unique_det in self.unique_potholes:
                        if detection.is_duplicate(unique_det) and unique_det.age() < self.cooldown_time:
                            is_new_unique = False
                            break
                    
                    if is_new_unique:
                        self.total_pothole_count += 1
                        self.unique_potholes.append(detection)
                        self.logger.info(f"🕳️  POTHOLE #{self.total_pothole_count} - "
                                       f"{DangerLevel.get_label(danger_level)} danger")
                    
                    new_detections.append(detection)
                    self.active_potholes.append(detection)
        
        # Cleanup
        current_time = time.time()
        self.active_potholes = [d for d in self.active_potholes 
                               if current_time - d.timestamp < self.pothole_persistence]
        self.unique_potholes = [d for d in self.unique_potholes 
                               if d.age() < self.cooldown_time * 2]
        
        return new_detections
    
    def detect_obstacles(self, frame):
        """Detect obstacles using YOLO."""
        if not self.yolo_model:
            return []
        
        new_detections = []
        
        try:
            results = self.yolo_model(frame, verbose=False, conf=YOLO_CONF, iou=0.45)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    
                    if cls in self.obstacle_config:
                        class_name, danger_level = self.obstacle_config[cls]
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox = (int(x1), int(y1), int(x2), int(y2))
                        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        conf = float(box.conf[0])
                        
                        detection = ObstacleDetection(bbox, center, class_name, conf, danger_level)
                        
                        is_duplicate = any(detection.is_duplicate(d, threshold=100) 
                                         for d in self.active_obstacles)
                        
                        if not is_duplicate:
                            is_new_unique = True
                            for unique_det in self.unique_obstacles:
                                if (detection.is_duplicate(unique_det, threshold=100) and 
                                    unique_det.age() < self.cooldown_time):
                                    is_new_unique = False
                                    break
                            
                            if is_new_unique:
                                self.total_obstacle_count += 1
                                self.unique_obstacles.append(detection)
                                self.logger.info(f"⚠️  OBSTACLE #{self.total_obstacle_count} - "
                                               f"{class_name.upper()} ({DangerLevel.get_label(danger_level)})")
                            
                            new_detections.append(detection)
                            self.active_obstacles.append(detection)
        
        except Exception as e:
            pass
        
        # Cleanup
        current_time = time.time()
        self.active_obstacles = [d for d in self.active_obstacles 
                                if current_time - d.timestamp < self.obstacle_persistence]
        self.unique_obstacles = [d for d in self.unique_obstacles 
                                if d.age() < self.cooldown_time * 2]
        
        return new_detections
    
    def draw_detections(self, frame):
        """Draw all active detections."""
        current_time = time.time()
        
        # Draw potholes
        for detection in self.active_potholes:
            age = current_time - detection.timestamp
            alpha = 1.0 - (age / self.pothole_persistence)
            alpha = max(0.3, min(1.0, alpha))
            
            color = DangerLevel.get_color(detection.danger_level)
            color_faded = tuple(int(c * alpha) for c in color)
            
            cv2.circle(frame, detection.center, detection.radius, color_faded, 3)
            cv2.circle(frame, detection.center, 5, color_faded, -1)
            
            label = f"P#{detection.id} {DangerLevel.get_label(detection.danger_level)}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_x = detection.center[0] - label_size[0] // 2
            label_y = detection.center[1] - detection.radius - 10
            
            cv2.rectangle(frame, (label_x - 3, label_y - label_size[1] - 3),
                         (label_x + label_size[0] + 3, label_y + 3), color_faded, -1)
            cv2.putText(frame, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw obstacles
        for detection in self.active_obstacles:
            age = current_time - detection.timestamp
            alpha = 1.0 - (age / self.obstacle_persistence)
            alpha = max(0.3, min(1.0, alpha))
            
            color = DangerLevel.get_color(detection.danger_level)
            color_faded = tuple(int(c * alpha) for c in color)
            
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_faded, 3)
            
            label = f"O#{detection.id} {detection.class_name.upper()} {DangerLevel.get_label(detection.danger_level)}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0] + 10, y1), color_faded, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def draw_stats(self, frame, fps):
        """Draw statistics overlay."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        stats = [
            f"Platform: {PLATFORM.upper()} | FPS: {fps:.1f}",
            f"",
            f"POTHOLES: Active={len(self.active_potholes)} | Total={self.total_pothole_count}",
            f"OBSTACLES: Active={len(self.active_obstacles)} | Total={self.total_obstacle_count}",
            f"",
            f"Colors: GREEN=Safe YELLOW=Low ORANGE=Med RED=High"
        ]
        
        y_offset = 30
        for text in stats:
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            y_offset += 20
        
        return frame


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Portable Road Hazard Detection')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--no-display', action='store_true', help='Headless mode')
    parser.add_argument('--skip-install', action='store_true', help='Skip dependency check')
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger("Main")
    
    logger.info("="*60)
    logger.info(f"Road Hazard Detection - {PLATFORM.upper()}")
    logger.info("="*60)
    logger.info(f"Camera: {args.camera}")
    logger.info(f"Display: {not args.no_display}")
    logger.info("="*60)
    
    detector = RoadHazardDetector(display=not args.no_display)
    
    # Initialize camera
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        logger.error(f"Failed to open camera {args.camera}")
        return 1
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, DEFAULT_FPS)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    logger.info(f"Camera: {int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    if not args.no_display:
        cv2.namedWindow('Road Hazard Detection', cv2.WINDOW_NORMAL)
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                logger.warning("Frame read failed, reconnecting...")
                camera.release()
                time.sleep(1)
                camera = cv2.VideoCapture(args.camera)
                if not camera.isOpened():
                    break
                continue
            
            # Detect hazards
            potholes = detector.detect_potholes(frame)
            obstacles = detector.detect_obstacles(frame)
            
            # Draw
            frame = detector.draw_detections(frame)
            
            # FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            
            frame = detector.draw_stats(frame, fps)
            
            if not args.no_display:
                cv2.imshow('Road Hazard Detection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    detector.total_pothole_count = 0
                    detector.total_obstacle_count = 0
                    detector.unique_potholes.clear()
                    detector.unique_obstacles.clear()
                    logger.info("Counts reset")
    
    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    finally:
        camera.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        
        logger.info("\n" + "="*60)
        logger.info("SESSION SUMMARY")
        logger.info("="*60)
        logger.info(f"Frames: {frame_count}")
        logger.info(f"Potholes: {detector.total_pothole_count}")
        logger.info(f"Obstacles: {detector.total_obstacle_count}")
        logger.info(f"Avg FPS: {fps:.2f}")
        logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
