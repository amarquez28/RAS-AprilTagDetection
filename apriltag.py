import cv2
from pyapriltags import Detector
import numpy as np
from picamera2 import Picamera2
import time
from ntcore import NetworkTableInstance

class AprilTagDetector:
    def __init__(self, tag_family='tag36h11', camera_params=None, roborio_ip='10.TE.AM.2'):
        """
        Initialize AprilTag detector for Raspberry Pi with IMX296 camera
        
        Args:
            tag_family: AprilTag family (default: 'tag36h11')
            camera_params: Camera calibration parameters [fx, fy, cx, cy]
            roborio_ip: IP address of the roboRIO (default: '10.TE.AM.2', replace TE.AM with your team number)
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
            #TODO:
            # These are approximate values - calibrate for better accuracy
            self.fx = 500  # Focal length x
            self.fy = 500  # Focal length y
            self.cx = 740  # Principal point x (half of 1480)
            self.cy = 555  # Principal point y (half of 1110)
        else:
            self.fx, self.fy, self.cx, self.cy = camera_params
        
        # Initialize NetworkTables
        self.setup_networktables(roborio_ip)
        
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
        
    def setup_networktables(self, roborio_ip):
        """
        Setup NetworkTables connection to roboRIO
        
        Args:
            roborio_ip: IP address of the roboRIO
        """
        # Get NetworkTables instance
        self.nt_inst = NetworkTableInstance.getDefault()
        
        # Set up as a client
        self.nt_inst.startClient4("apriltag_detector")
        self.nt_inst.setServerTeam(int(roborio_ip.split('.')[1]))  # Extract team number
        # Or use: self.nt_inst.setServer(roborio_ip)
        
        # Get the vision table
        self.vision_table = self.nt_inst.getTable("Vision")
        
        # Create entries for publishing data
        self.tag_detected_entry = self.vision_table.getBooleanTopic("tag_detected").publish()
        self.tag_id_entry = self.vision_table.getIntegerTopic("tag_id").publish()
        self.tag_x_entry = self.vision_table.getDoubleTopic("tag_x").publish()
        self.tag_y_entry = self.vision_table.getDoubleTopic("tag_y").publish()
        self.tag_distance_entry = self.vision_table.getDoubleTopic("tag_distance").publish()
        self.tag_count_entry = self.vision_table.getIntegerTopic("tag_count").publish()
        
        # For multiple tags, create array entries
        self.tags_ids_entry = self.vision_table.getIntegerArrayTopic("tags_ids").publish()
        self.tags_x_entry = self.vision_table.getDoubleArrayTopic("tags_x").publish()
        self.tags_y_entry = self.vision_table.getDoubleArrayTopic("tags_y").publish()
        self.tags_distances_entry = self.vision_table.getDoubleArrayTopic("tags_distances").publish()
        
        # Heartbeat for connection monitoring
        self.heartbeat_entry = self.vision_table.getIntegerTopic("heartbeat").publish()
        self.heartbeat_counter = 0
        
        print(f"NetworkTables initialized, connecting to roboRIO at {roborio_ip}")
        
    def publish_detections(self, tags):
        """
        Publish detection results to NetworkTables
        
        Args:
            tags: List of detected tags
        """
        # Update heartbeat
        self.heartbeat_counter += 1
        self.heartbeat_entry.set(self.heartbeat_counter)
        
        # Publish number of tags detected
        num_tags = len(tags)
        self.tag_count_entry.set(num_tags)
        
        if num_tags > 0:
            # Publish first/primary tag data (for simple use cases)
            primary_tag = tags[0]
            self.tag_detected_entry.set(True)
            self.tag_id_entry.set(int(primary_tag.tag_id))
            self.tag_x_entry.set(float(primary_tag.center[0]))
            self.tag_y_entry.set(float(primary_tag.center[1]))
            
            if primary_tag.pose_t is not None:
                distance = float(np.linalg.norm(primary_tag.pose_t))
                self.tag_distance_entry.set(distance)
            else:
                self.tag_distance_entry.set(-1.0)
            
            # Publish all tags as arrays
            ids = [int(tag.tag_id) for tag in tags]
            x_coords = [float(tag.center[0]) for tag in tags]
            y_coords = [float(tag.center[1]) for tag in tags]
            distances = []
            
            for tag in tags:
                if tag.pose_t is not None:
                    distances.append(float(np.linalg.norm(tag.pose_t)))
                else:
                    distances.append(-1.0)
            
            self.tags_ids_entry.set(ids)
            self.tags_x_entry.set(x_coords)
            self.tags_y_entry.set(y_coords)
            self.tags_distances_entry.set(distances)
            
        else:
            # No tags detected
            self.tag_detected_entry.set(False)
            self.tag_id_entry.set(-1)
            self.tag_x_entry.set(0.0)
            self.tag_y_entry.set(0.0)
            self.tag_distance_entry.set(-1.0)
            
            # Clear arrays
            self.tags_ids_entry.set([])
            self.tags_x_entry.set([])
            self.tags_y_entry.set([])
            self.tags_distances_entry.set([])
    
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
        
        # Draw coordinates
        cv2.putText(image, f"X: {tag.center[0]:.1f}, Y: {tag.center[1]:.1f}", 
                    (center[0] - 20, center[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw pose information if available
        if tag.pose_t is not None:
            distance = np.linalg.norm(tag.pose_t)
            cv2.putText(image, f"Dist: {distance:.2f}m", 
                        (center[0] - 20, center[1] + 60),
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
        print(f"NetworkTables connected: {self.nt_inst.isConnected()}")
        
        try:
            frame = self.picam2.capture_array()
            while !detect_start_light(frame):
                print("Looking for start_light")

            while True:
                # Capture frame
                frame = self.picam2.capture_array()
                # Detect tags
                tags = self.detect_tags(frame)
                
                # Publish to NetworkTables
                self.publish_detections(tags)
                
                # Draw detections
                for tag in tags:
                    self.draw_detection(frame, tag)
                    
                    # Print detection info
                    print(f"Detected tag ID {tag.tag_id} at X:{tag.center[0]:.1f}, Y:{tag.center[1]:.1f}")
                    if tag.pose_t is not None:
                        distance = np.linalg.norm(tag.pose_t)
                        print(f"  Distance: {distance:.3f}m")
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter >= 30:
                    fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Draw FPS and NetworkTables status
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                nt_status = "NT: Connected" if self.nt_inst.isConnected() else "NT: Disconnected"
                nt_color = (0, 255, 0) if self.nt_inst.isConnected() else (0, 0, 255)
                cv2.putText(frame, nt_status, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, nt_color, 2)
                
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
        
        # Stop NetworkTables
        self.nt_inst.stopClient()
        print("Cleanup complete")

def detect_start_light(frame):
    roi = frame[50:150, 250:390]
    
    gray = cv2.cvtColor(roi cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshhold(gray, 230, 255, cv2.THRESH_BINARY)

    bright_pixels = cv2.countNonZero(thresh)

    cv2.imshow("Bright Mask", thresh)

    return bright_pixels > 3000


if __name__ == "__main__":
    # Create detector instance
    # Option 1: Use team number directly (recommended)
    # detector = AprilTagDetector(
    #     tag_family='tag36h11',
    #     camera_params=None,
    #     roborio_ip=2518  # Replace with your actual team number
    # )
    
    # Option 2: Use full IP address
    detector = AprilTagDetector(
        tag_family='tag36h11',  # Options: 'tag36h11', 'tag25h9', 'tag16h5', etc.
        camera_params=None,  # Or provide [fx, fy, cx, cy]
        roborio_ip='10.25.18.2'  # Replace with your roboRIO IP address
    )
    # Run detection
    # Set display=False if running headless (no display/VNC)
    detector.run(display=True, save_video=False)