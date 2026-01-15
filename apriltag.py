import cv2
from pyapriltags import Detector
import numpy as np
from picamera2 import Picamera2
import time

class AprilTagDetector:
    def __init__(self, tag_family='tag36h11', camera_params=None):
        """
        Initialize AprilTag detector for Raspberry Pi with IMX296 camera
        
        Args:
            tag_family: AprilTag family (default: 'tag36h11')
            camera_params: Camera calibration parameters [fx, fy, cx, cy]
        """
        # Initialize the detector
        self.detector = Detector(
            families=tag_family,
            nthreads=4,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        
        # Default camera parameters for IMX296 (adjust based on your calibration)
        if camera_params is None:
            # These are approximate values - calibrate for better accuracy
            self.fx = 500  # Focal length x
            self.fy = 500  # Focal length y
            self.cx = 740  # Principal point x (half of 1480)
            self.cy = 555  # Principal point y (half of 1110)
        else:
            self.fx, self.fy, self.cx, self.cy = camera_params
        
        # Initialize Picamera2
        self.picam2 = Picamera2()
        
        # Configure camera for IMX296
        config = self.picam2.create_preview_configuration(
            main={"size": (1480, 1110), "format": "RGB888"},
            controls={"FrameRate": 30}
        )
        self.picam2.configure(config)
        self.picam2.start()
        
        # Allow camera to warm up
        time.sleep(2)
        
    def detect_tags(self, image):
        """
        Detect AprilTags in the image
        
        Args:
            image: BGR image from camera
            
        Returns:
            List of detected tags
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect tags
        tags = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[self.fx, self.fy, self.cx, self.cy],
            tag_size=0.1  # Tag size in meters (adjust to your actual tag size)
        )
        
        return tags
    
    def draw_detection(self, image, tag):
        """
        Draw detection results on the image
        
        Args:
            image: Image to draw on
            tag: Detected tag object
        """
        # Draw corners
        corners = tag.corners.astype(int)
        for i in range(4):
            cv2.line(image, tuple(corners[i]), tuple(corners[(i+1)%4]), (0, 255, 0), 2)
        
        # Draw center
        center = tuple(tag.center.astype(int))
        cv2.circle(image, center, 5, (0, 0, 255), -1)
        
        # Draw tag ID
        cv2.putText(image, f"ID: {tag.tag_id}", 
                    (center[0] - 20, center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw pose information if available
        if tag.pose_t is not None:
            distance = np.linalg.norm(tag.pose_t)
            cv2.putText(image, f"Dist: {distance:.2f}m", 
                        (center[0] - 20, center[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    def run(self, display=True, save_video=False, output_file='output.avi'):
        """
        Run the detection loop
        
        Args:
            display: Show detection window (requires display/VNC)
            save_video: Save output to video file
            output_file: Output video filename
        """
        fps_counter = 0
        fps_start_time = time.time()
        fps = 0
        
        # Video writer setup
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(output_file, fourcc, 20.0, (1480, 1110))
        
        print("Starting AprilTag detection... Press 'q' to quit")
        
        try:
            while True:
                # Capture frame
                frame = self.picam2.capture_array()
                
                # Detect tags
                tags = self.detect_tags(frame)
                
                # Draw detections
                for tag in tags:
                    self.draw_detection(frame, tag)
                    
                    # Print detection info
                    print(f"Detected tag ID {tag.tag_id} at center {tag.center}")
                    if tag.pose_t is not None:
                        distance = np.linalg.norm(tag.pose_t)
                        print(f"  Distance: {distance:.3f}m")
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter >= 30:
                    fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Draw FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Save frame
                if save_video and video_writer is not None:
                    video_writer.write(frame)
                
                # Display frame
                if display:
                    cv2.imshow('AprilTag Detection', frame)
                    
                    # Check for quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        except KeyboardInterrupt:
            print("\nStopping detection...")
        
        finally:
            self.cleanup(video_writer)
    
    def cleanup(self, video_writer=None):
        """Clean up resources"""
        if video_writer is not None:
            video_writer.release()
        self.picam2.stop()
        cv2.destroyAllWindows()
        print("Cleanup complete")


if __name__ == "__main__":
    # Create detector instance
    # You can specify custom camera parameters if you've calibrated your camera
    detector = AprilTagDetector(
        tag_family='tag36h11',  # Options: 'tag36h11', 'tag25h9', 'tag16h5', etc.
        camera_params=None  # Or provide [fx, fy, cx, cy]
    )
    
    # Run detection
    # Set display=False if running headless (no display/VNC)
    detector.run(display=True, save_video=False)