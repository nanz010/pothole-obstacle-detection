"""Enhanced road hazard detection: Potholes + Obstacles (people, animals, vehicles).

This system combines:
- Computer Vision for pothole detection (circles around potholes)
- YOLOv8 for obstacle detection (boxes around people, animals, vehicles, etc.)

Usage:
    python car_detection_enhanced.py [--camera CAMERA_ID] [--save-detections]
    
Press 'q' to quit, 's' to save current frame, 'c' to clear detection history
"""

import cv2
import numpy as np
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
import json
import threading

try:
    import winsound
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLOv8 not available - install with: pip install ultralytics")


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


class RoadHazardDetector:
    """Detects both potholes and obstacles on the road."""
    
    def __init__(self, save_detections=False, output_dir="output/road_hazards", enable_audio=True):
        self.save_detections = save_detections
        self.output_dir = Path(output_dir)
        if save_detections:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detection parameters for potholes
        self.min_pothole_area = 800
        self.max_pothole_area = 100000
        self.detection_cooldown = 2.0
        
        # Persistent tracking - keep detections visible until camera passes
        self.active_potholes = []  # Potholes currently visible
        self.active_obstacles = []  # Obstacles currently visible
        self.pothole_persistence_time = 5.0  # Keep visible for 5 seconds
        self.obstacle_persistence_time = 3.0  # Keep visible for 3 seconds
        
        # Detection history
        self.recent_potholes = []
        self.recent_obstacles = []
        self.pothole_log = []
        self.obstacle_log = []
        
        # Counters
        self.total_potholes_detected = 0
        self.total_obstacles_detected = 0
        
        # Motion compensation
        self.prev_frame = None
        self.motion_threshold = 30
        
        # Alert system
        self.enable_audio = enable_audio and AUDIO_AVAILABLE
        self.last_alert_time = 0
        self.alert_cooldown = 1.5
        
        # Load YOLO model for obstacle detection
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n.pt')  # Lightweight model
                print("✓ YOLO model loaded for obstacle detection")
            except Exception as e:
                print(f"Could not load YOLO model: {e}")
        
        # COCO classes that are obstacles on roads
        self.obstacle_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
            5: 'bus', 7: 'truck', 14: 'bird', 15: 'cat', 16: 'dog',
            17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
            21: 'bear', 22: 'zebra', 23: 'giraffe'
        }
    
    def detect_potholes(self, frame):
        """Detect potholes using computer vision (dark holes in road).
        
        Returns:
            List of pothole detections with circular contours
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Create ROI mask - focus on ROAD AREA ONLY (bottom 50% of frame)
        # This excludes sky, trees, and upper background
        roi_mask = np.zeros(gray.shape, dtype=np.uint8)
        roi_start_y = int(height * 0.5)  # Start from 50% down (skip top 50%)
        roi_mask[roi_start_y:, :] = 255  # Only bottom 50% is active
        
        # Motion compensation
        if self.prev_frame is not None:
            frame_diff = cv2.absdiff(gray, self.prev_frame)
            motion_mask = frame_diff < self.motion_threshold
        else:
            motion_mask = np.ones(gray.shape, dtype=bool)
        
        self.prev_frame = gray.copy()
        
        # Detect dark regions (potholes) - only in ROI
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Use adaptive thresholding to detect dark regions
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 3  # Balanced parameters
        )
        
        # Apply ROI mask first (most important - excludes sky/trees)
        thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)
        
        # Apply motion mask
        thresh = cv2.bitwise_and(thresh, thresh, mask=motion_mask.astype(np.uint8) * 255)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        potholes = []
        current_time = time.time()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_pothole_area < area < self.max_pothole_area:
                # Get minimum enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                # Additional filtering: check if region is actually on road surface
                # Skip if too close to top of ROI (likely trees/sky bleeding in)
                if center[1] < roi_start_y + 30:
                    continue
                
                # Skip if too close to edges (likely frame artifacts)
                edge_margin = 50
                if (center[0] < edge_margin or center[0] > width - edge_margin or
                    center[1] > height - edge_margin):
                    continue
                
                # Check average intensity in the region (potholes should be dark)
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, center, radius, 255, -1)
                mean_intensity = cv2.mean(gray, mask=mask)[0]
                
                # Potholes are dark (shadows/holes) but not completely black
                # Skip if too bright (not a pothole) or too dark (likely sky/tree shadow)
                if mean_intensity > 120 or mean_intensity < 20:
                    continue
                
                # Check intensity variance - real potholes have some texture
                std_intensity = np.std(gray[mask > 0])
                if std_intensity < 5:  # Too uniform, likely not a pothole
                    continue
                
                # Filter by circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if 0.15 < circularity < 0.95:
                        # Check for duplicates
                        is_duplicate = False
                        for prev in self.recent_potholes:
                            if current_time - prev['time'] < self.detection_cooldown:
                                dist = np.sqrt((center[0] - prev['center'][0])**2 + 
                                             (center[1] - prev['center'][1])**2)
                                if dist < 100:
                                    is_duplicate = True
                                    break
                        
                        if not is_duplicate:
                            detection = {
                                'type': 'pothole',
                                'center': center,
                                'radius': radius,
                                'contour': contour,
                                'confidence': min(circularity * 1.3, 0.99),
                                'area': area,
                                'time': current_time,
                                'timestamp': datetime.now().isoformat(),
                                'id': self.total_potholes_detected + 1
                            }
                            potholes.append(detection)
                            self.recent_potholes.append(detection)
                            self.active_potholes.append(detection)
                            self.total_potholes_detected += 1
        
        # Clean old detections from recent history
        self.recent_potholes = [
            p for p in self.recent_potholes 
            if current_time - p['time'] < self.detection_cooldown * 2
        ]
        
        # Clean old detections from active display (persistent tracking)
        self.active_potholes = [
            p for p in self.active_potholes 
            if current_time - p['time'] < self.pothole_persistence_time
        ]
        
        return potholes
    
    def detect_obstacles(self, frame):
        """Detect obstacles using YOLO (people, animals, vehicles).
        
        Returns:
            List of obstacle detections with bounding boxes
        """
        if not self.yolo_model:
            return []
        
        obstacles = []
        current_time = time.time()
        
        try:
            # Run YOLO inference
            results = self.yolo_model(frame, verbose=False, conf=0.4)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    
                    # Only detect obstacle classes
                    if cls in self.obstacle_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        
                        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        
                        # Check for duplicates
                        is_duplicate = False
                        for prev in self.recent_obstacles:
                            if current_time - prev['time'] < self.detection_cooldown:
                                dist = np.sqrt((center[0] - prev['center'][0])**2 + 
                                             (center[1] - prev['center'][1])**2)
                                if dist < 100:
                                    is_duplicate = True
                                    break
                        
                        if not is_duplicate:
                            detection = {
                                'type': 'obstacle',
                                'class': self.obstacle_classes[cls],
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'center': center,
                                'confidence': conf,
                                'time': current_time,
                                'timestamp': datetime.now().isoformat(),
                                'id': self.total_obstacles_detected + 1
                            }
                            obstacles.append(detection)
                            self.recent_obstacles.append(detection)
                            self.active_obstacles.append(detection)
                            self.total_obstacles_detected += 1
        
        except Exception as e:
            pass  # Silently handle YOLO errors
        
        # Clean old detections from recent history
        self.recent_obstacles = [
            o for o in self.recent_obstacles 
            if current_time - o['time'] < self.detection_cooldown * 2
        ]
        
        # Clean old detections from active display (persistent tracking)
        self.active_obstacles = [
            o for o in self.active_obstacles 
            if current_time - o['time'] < self.obstacle_persistence_time
        ]
        
        return obstacles
    
    def save_detection(self, frame, detection, detection_id, det_type):
        """Save detected hazard image and metadata."""
        if not self.save_detections:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{det_type}_{timestamp}_{detection_id}"
        
        # Save image
        img_path = self.output_dir / f"{filename}.jpg"
        cv2.imwrite(str(img_path), frame)
        
        # Save metadata
        metadata = {
            'type': det_type,
            'timestamp': detection['timestamp'],
            'confidence': detection['confidence'],
            'gps': {'latitude': 0.0, 'longitude': 0.0, 'note': 'Simulated GPS'}
        }
        
        if det_type == 'pothole':
            metadata['center'] = detection['center']
            metadata['radius'] = detection['radius']
            metadata['area'] = detection['area']
        else:  # obstacle
            metadata['class'] = detection['class']
            metadata['bbox'] = detection['bbox']
        
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if det_type == 'pothole':
            self.pothole_log.append(metadata)
        else:
            self.obstacle_log.append(metadata)
        
        return img_path
    
    def play_alert_sound(self, alert_type='hazard'):
        """Play audio alert for driver."""
        if not self.enable_audio:
            return
        
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        self.last_alert_time = current_time
        
        def play_sound():
            try:
                if AUDIO_AVAILABLE:
                    if alert_type == 'pothole':
                        winsound.Beep(800, 200)  # Lower tone for pothole
                        time.sleep(0.1)
                        winsound.Beep(800, 200)
                    else:  # obstacle
                        winsound.Beep(1200, 200)  # Higher tone for obstacle
                        time.sleep(0.1)
                        winsound.Beep(1200, 200)
            except:
                pass
        
        thread = threading.Thread(target=play_sound, daemon=True)
        thread.start()
    
    def save_session_logs(self):
        """Save complete session logs."""
        if not self.save_detections:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.pothole_log:
            log_path = self.output_dir / f"potholes_log_{timestamp}.json"
            with open(log_path, 'w') as f:
                json.dump(self.pothole_log, f, indent=2)
        
        if self.obstacle_log:
            log_path = self.output_dir / f"obstacles_log_{timestamp}.json"
            with open(log_path, 'w') as f:
                json.dump(self.obstacle_log, f, indent=2)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Road hazard detection system')
    parser.add_argument('--camera', type=int, default=1, help='Camera device ID')
    parser.add_argument('--save-detections', action='store_true', help='Save detections')
    parser.add_argument('--no-audio', action='store_true', help='Disable audio alerts')
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger("road_hazards")
    
    logger.info("="*60)
    logger.info("Road Hazard Detection System")
    logger.info("="*60)
    logger.info(f"\nCamera ID: {args.camera}")
    logger.info(f"Save detections: {args.save_detections}")
    logger.info(f"Audio alerts: {not args.no_audio and AUDIO_AVAILABLE}")
    logger.info("\nDetection Types:")
    logger.info("  - POTHOLES: Circles (red) - CV-based detection")
    logger.info("  - OBSTACLES: Boxes (orange) - YOLO detection")
    logger.info("    (people, animals, vehicles, etc.)")
    logger.info("\nControls: 'q'=Quit, 's'=Save frame, 'c'=Clear history")
    logger.info("="*60)
    
    # Initialize detector
    detector = RoadHazardDetector(
        save_detections=args.save_detections,
        enable_audio=not args.no_audio
    )
    
    # Initialize camera
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        logger.error(f"Failed to open camera {args.camera}!")
        return 1
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    logger.info(f"\nCamera initialized: {camera.get(cv2.CAP_PROP_FRAME_WIDTH)}x{camera.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    
    # Create window
    cv2.namedWindow('Road Hazard Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Road Hazard Detection', 1280, 720)
    
    # Stats
    frame_count = 0
    pothole_count = 0
    obstacle_count = 0
    fps_start_time = time.time()
    fps = 0
    alert_flash = False
    alert_flash_time = 0
    alert_type = None
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                logger.warning("Failed to read frame, attempting to reconnect...")
                camera.release()
                time.sleep(1)
                camera = cv2.VideoCapture(args.camera)
                if not camera.isOpened():
                    logger.error("Failed to reconnect to camera!")
                    break
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                camera.set(cv2.CAP_PROP_FPS, 30)
                continue
            
            start_time = time.time()
            
            # Detect potholes and obstacles
            potholes = detector.detect_potholes(frame)
            obstacles = detector.detect_obstacles(frame)
            
            current_time = time.time()
            
            # Handle alerts for NEW detections only
            if len(potholes) > 0 or len(obstacles) > 0:
                alert_flash = True
                alert_flash_time = current_time
                if len(potholes) > 0:
                    detector.play_alert_sound('pothole')
                    alert_type = 'pothole'
                if len(obstacles) > 0:
                    detector.play_alert_sound('obstacle')
                    if alert_type == 'pothole' and len(obstacles) > 0:
                        alert_type = 'both'
                    else:
                        alert_type = 'obstacle'
            
            show_alert = alert_flash and (current_time - alert_flash_time < 1.0)
            if current_time - alert_flash_time >= 1.0:
                alert_flash = False
            
            # Save NEW detections
            for pothole in potholes:
                if args.save_detections:
                    img_path = detector.save_detection(frame, pothole, pothole['id'], 'pothole')
                    logger.info(f"🕳️  POTHOLE #{pothole['id']} DETECTED! Saved to: {img_path}")
            
            for obstacle in obstacles:
                if args.save_detections:
                    img_path = detector.save_detection(frame, obstacle, obstacle['id'], 'obstacle')
                    logger.info(f"⚠️  OBSTACLE #{obstacle['id']} ({obstacle['class'].upper()}) DETECTED! Saved to: {img_path}")
            
            # Draw ALL ACTIVE POTHOLES (persistent tracking)
            for pothole in detector.active_potholes:
                center = pothole['center']
                radius = pothole['radius']
                
                # Calculate fade based on age
                age = current_time - pothole['time']
                alpha = 1.0 - (age / detector.pothole_persistence_time)
                alpha = max(0.3, min(1.0, alpha))  # Keep at least 30% visible
                
                # Draw circle (red) with fade
                color_intensity = int(255 * alpha)
                cv2.circle(frame, center, radius, (0, 0, color_intensity), 3)
                cv2.circle(frame, center, 5, (0, 0, color_intensity), -1)
                
                # Draw label with ID
                label = f"P#{pothole['id']}: {pothole['confidence']:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_x = center[0] - label_size[0] // 2
                label_y = center[1] - radius - 15
                
                cv2.rectangle(frame, (label_x - 5, label_y - label_size[1] - 5),
                            (label_x + label_size[0] + 5, label_y + 5), 
                            (0, 0, color_intensity), -1)
                cv2.putText(frame, label, (label_x, label_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw ALL ACTIVE OBSTACLES (persistent tracking)
            for obstacle in detector.active_obstacles:
                x1, y1, x2, y2 = obstacle['bbox']
                
                # Calculate fade based on age
                age = current_time - obstacle['time']
                alpha = 1.0 - (age / detector.obstacle_persistence_time)
                alpha = max(0.3, min(1.0, alpha))
                
                # Draw box (orange) with fade
                color_intensity = int(255 * alpha)
                cv2.rectangle(frame, (x1, y1), (x2, y2), 
                            (0, int(165 * alpha), color_intensity), 3)
                
                # Draw label with ID
                label = f"O#{obstacle['id']} {obstacle['class'].upper()}: {obstacle['confidence']:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0] + 10, y1), 
                            (0, int(165 * alpha), color_intensity), -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw alert banner
            if show_alert:
                cv2.circle(frame, (60, 60), 40, (0, 0, 255), -1)
                cv2.putText(frame, "!", (45, 75),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
                
                if alert_type == 'both':
                    warning_text = "⚠️  POTHOLE & OBSTACLE AHEAD  ⚠️"
                elif alert_type == 'pothole':
                    warning_text = "🕳️  POTHOLE AHEAD  🕳️"
                else:
                    warning_text = "⚠️  OBSTACLE AHEAD  ⚠️"
                
                text_size, _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                text_x = (frame.shape[1] - text_size[0]) // 2
                
                cv2.rectangle(frame, (text_x - 20, 10),
                            (text_x + text_size[0] + 20, 55), (0, 0, 255), -1)
                cv2.putText(frame, warning_text, (text_x, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            
            processing_time = (time.time() - start_time) * 1000
            
            # Draw info overlay with SEPARATE COUNTERS
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (500, 180), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            info_text = [
                f"FPS: {fps:.1f} | Processing: {processing_time:.1f}ms",
                f"",
                f"POTHOLES:",
                f"  Active: {len(detector.active_potholes)} | Total: {detector.total_potholes_detected}",
                f"OBSTACLES:",
                f"  Active: {len(detector.active_obstacles)} | Total: {detector.total_obstacles_detected}",
                f"Frame: {frame_count}"
            ]
            
            y_offset = 35
            for i, text in enumerate(info_text):
                if i == 2:  # POTHOLES header
                    color = (0, 0, 255)  # Red
                elif i == 4:  # OBSTACLES header
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 0)  # Green
                
                cv2.putText(frame, text, (20, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 22
            
            # Status
            cv2.putText(frame, "PERSISTENT TRACKING: Circles=Potholes | Boxes=Obstacles",
                       (20, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Road Hazard Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = Path("output/manual_saves")
                save_path.mkdir(parents=True, exist_ok=True)
                filename = save_path / f"manual_{timestamp}.jpg"
                cv2.imwrite(str(filename), frame)
                logger.info(f"Frame saved: {filename}")
            elif key == ord('c'):
                detector.recent_potholes.clear()
                detector.recent_obstacles.clear()
                logger.info("Detection history cleared")
    
    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        
        if args.save_detections:
            detector.save_session_logs()
        
        logger.info("\n" + "="*60)
        logger.info("SESSION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total frames: {frame_count}")
        logger.info(f"Potholes detected: {detector.total_potholes_detected}")
        logger.info(f"Obstacles detected: {detector.total_obstacles_detected}")
        logger.info(f"Average FPS: {fps:.2f}")
        logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
