import cv2
from pyapriltags import Detector
import numpy as np
from picamera2 import Picamera2
import time
from ntcore import NetworkTableInstance
import socket
import struct


def listen():
    """
    Debug/diagnostic tool only - run this manually during testing.
    Listens on port 1150 for roboRIO reply packets to confirm
    packets are reaching the roboRIO.
    """
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind(("0.0.0.0", 1150))
    recv_sock.settimeout(1.0)
    while True:
        try:
            data, addr = recv_sock.recvfrom(1024)
            print(f" << roboRIO reply: {data.hex()}")
        except socket.timeout:
            pass


class DSPacketSender:
    """
    FRC Driver Station UDP packet sender.
    Sends enable/disable commands to the roboRIO.
    """

    def __init__(self, roborio_ip):
        self.roborio_ip = roborio_ip
        self.roborio_port = 1110
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sequence = 0

        self.AUTONOMOUS_DISABLED = 0x02
        self.AUTONOMOUS_ENABLED  = 0x06
        self.ALLIANCE_STATION    = 0x00  # Red 1

        print("DS Packet Sender initialized")

    def create_ds_packet(self, enabled):
        control = self.AUTONOMOUS_ENABLED if enabled else self.AUTONOMOUS_DISABLED
        packet = bytearray(6)
        struct.pack_into('>H', packet, 0, self.sequence & 0xFFFF)
        packet[2] = 0x01
        packet[3] = control
        packet[4] = 0x00
        packet[5] = self.ALLIANCE_STATION
        return packet

    def enable_robot(self, autonomous=True):
        packet = self.create_ds_packet(enabled=True)
        self.sock.sendto(packet, (self.roborio_ip, self.roborio_port))
        self.sequence = (self.sequence + 1) & 0xFFFF
        print(f"Sent ENABLE command (autonomous={autonomous})")

    def disable_robot(self):
        packet = self.create_ds_packet(enabled=False)
        self.sock.sendto(packet, (self.roborio_ip, self.roborio_port))
        self.sequence = (self.sequence + 1) & 0xFFFF
        print("Sent DISABLE command")

    def send_keepalive(self, enabled=False):
        """Send keepalive packet - call at ~50Hz from main loop."""
        packet = self.create_ds_packet(enabled=enabled)
        self.sock.sendto(packet, (self.roborio_ip, self.roborio_port))
        self.sequence = (self.sequence + 1) & 0xFFFF

    def close(self):
        self.sock.close()


class AprilTagDetector:
    def __init__(self, tag_family='tag36h11', camera_params=None, roborio_ip='10.0.67.2', display=False):
        """
        Initialize AprilTag detector for Raspberry Pi with IMX296 camera.

        Args:
            tag_family:    AprilTag family (default: 'tag36h11')
            camera_params: Camera calibration parameters [fx, fy, cx, cy]
            roborio_ip:    IP address of the roboRIO
            display:       True to configure for display mode, False for headless max-fps
        """
        self.task_done_sub = None
        self.display = display

        self.detector = Detector(
            families=tag_family,
            nthreads=4,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

        # Camera parameters for IMX296 after 90° CW rotation.
        #
        # The physical sensor captures 1480 (W) × 1110 (H).
        # After cv2.ROTATE_90_CLOCKWISE the frame becomes 1110 (W) × 1480 (H).
        # The principal point axes swap accordingly:
        #   cx (horizontal centre) = 1110 / 2 = 555
        #   cy (vertical centre)   = 1480 / 2 = 740
        # fx and fy are swapped to match the new axis assignment.
        # These are uncalibrated estimates — proper camera calibration will
        # improve pose accuracy significantly.
        if camera_params is None:
            self.fx = 500  # Focal length along new X axis (was Y on raw sensor)
            self.fy = 500  # Focal length along new Y axis (was X on raw sensor)
            self.cx = 555  # Principal point x: half of rotated frame width  (1110/2)
            self.cy = 740  # Principal point y: half of rotated frame height (1480/2)
        else:
            self.fx, self.fy, self.cx, self.cy = camera_params

        self.setup_networktables(roborio_ip)

        self.ds_sender = DSPacketSender(roborio_ip)
        self.robot_enabled = False
        self.last_keepalive_time = time.time()

        self.picam2 = Picamera2()

        if display:
            # Preview config - includes ISP pipeline for display quality
            config = self.picam2.create_preview_configuration(
                main={"size": (1480, 1110), "format": "RGB888"},
                controls={"FrameRate": 30}
            )
        else:
            # Still config - lower ISP overhead, better throughput for headless
            config = self.picam2.create_still_configuration(
                main={"size": (1480, 1110), "format": "RGB888"}
            )

        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)  # Camera warm-up

    def setup_networktables(self, roborio_ip):
        self.nt_inst = NetworkTableInstance.getDefault()
        self.nt_inst.startClient4("apriltag_detector")

        if isinstance(roborio_ip, int):
            self.nt_inst.setServerTeam(roborio_ip)
        else:
            self.nt_inst.setServer(roborio_ip)

        self.vision_table = self.nt_inst.getTable("Vision")
        self.task_done_sub = self.vision_table.getBooleanTopic("task_done").subscribe(False)

        # Single tag (primary)
        self.tag_detected_entry  = self.vision_table.getBooleanTopic("tag_detected").publish()
        self.tag_id_entry        = self.vision_table.getIntegerTopic("tag_id").publish()
        self.tag_x_entry         = self.vision_table.getDoubleTopic("tag_x").publish()
        self.tag_y_entry         = self.vision_table.getDoubleTopic("tag_y").publish()
        self.tag_distance_entry  = self.vision_table.getDoubleTopic("tag_distance").publish()
        self.tag_count_entry     = self.vision_table.getIntegerTopic("tag_count").publish()

        # All tags as arrays
        self.tags_ids_entry       = self.vision_table.getIntegerArrayTopic("tags_ids").publish()
        self.tags_x_entry         = self.vision_table.getDoubleArrayTopic("tags_x").publish()
        self.tags_y_entry         = self.vision_table.getDoubleArrayTopic("tags_y").publish()
        self.tags_distances_entry = self.vision_table.getDoubleArrayTopic("tags_distances").publish()

        self.heartbeat_entry   = self.vision_table.getIntegerTopic("heartbeat").publish()
        self.heartbeat_counter = 0
        self.start_light_entry = self.vision_table.getBooleanTopic("start_light_detected").publish()

        print(f"NetworkTables initialized, connecting to roboRIO at {roborio_ip}")

    def publish_detections(self, tags):
        self.heartbeat_counter += 1
        self.heartbeat_entry.set(self.heartbeat_counter)
        self.tag_count_entry.set(len(tags))

        if tags:
            primary = tags[0]
            self.tag_detected_entry.set(True)
            self.tag_id_entry.set(int(primary.tag_id))
            self.tag_x_entry.set(float(primary.center[0]))
            self.tag_y_entry.set(float(primary.center[1]))
            dist = float(np.linalg.norm(primary.pose_t)) if primary.pose_t is not None else -1.0
            self.tag_distance_entry.set(dist)

            # ── CALIBRATION OUTPUT ────────────────────────────────────────
            # Aim the robot straight at a tag, read tag_x over SSH, and
            # average 5-10 readings. That value is your kCameraCenter_px.
            # Comment this block out once calibration is done.
            # dist_str = f"{dist:.3f}m" if dist >= 0 else "N/A"
            # print(f"[CAL] tag_x={float(primary.center[0]):.1f}  "
            #       f"tag_y={float(primary.center[1]):.1f}  "
            #       f"dist={dist_str}")

            self.tags_ids_entry.set([int(t.tag_id) for t in tags])
            self.tags_x_entry.set([float(t.center[0]) for t in tags])
            self.tags_y_entry.set([float(t.center[1]) for t in tags])
            self.tags_distances_entry.set([
                float(np.linalg.norm(t.pose_t)) if t.pose_t is not None else -1.0
                for t in tags
            ])
        else:
            self.tag_detected_entry.set(False)
            self.tag_id_entry.set(-1)
            self.tag_x_entry.set(0.0)
            self.tag_y_entry.set(0.0)
            self.tag_distance_entry.set(-1.0)
            self.tags_ids_entry.set([])
            self.tags_x_entry.set([])
            self.tags_y_entry.set([])
            self.tags_distances_entry.set([])

    def send_ds_keepalive(self):
        current_time = time.time()
        if current_time - self.last_keepalive_time >= 0.02:  # 50Hz
            self.ds_sender.send_keepalive(enabled=self.robot_enabled)
            self.last_keepalive_time = current_time

    def detect_tags(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[self.fx, self.fy, self.cx, self.cy],
            tag_size=0.1  # Tag size in meters - adjust to your actual tag size
        )

    def draw_detection(self, image, tag):
        """Draw tag overlay - only called in display mode."""
        corners = tag.corners.astype(int)
        for i in range(4):
            cv2.line(image, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)

        center = tuple(tag.center.astype(int))
        cv2.circle(image, center, 5, (0, 0, 255), -1)
        cv2.putText(image, f"ID: {tag.tag_id}",
                    (center[0] - 20, center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(image, f"X: {tag.center[0]:.1f}, Y: {tag.center[1]:.1f}",
                    (center[0] - 20, center[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        if tag.pose_t is not None:
            distance = np.linalg.norm(tag.pose_t)
            cv2.putText(image, f"Dist: {distance:.2f}m",
                        (center[0] - 20, center[1] + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    def run(self, save_video=False, output_file='output.avi'):
        """
        Run the detection loop.

        Args:
            save_video:   Save annotated output to video file (display mode only)
            output_file:  Output video filename
        """
        # Video writer - display mode only
        video_writer = None
        if self.display and save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(output_file, fourcc, 20.0, (1480, 1110))

        # FPS tracking - display mode only
        fps = 0
        fps_counter = 0
        fps_start_time = time.time()

        print("Starting AprilTag detection...")
        print(f"NetworkTables connected: {self.nt_inst.isConnected()}")
        print(f"Mode: {'display' if self.display else 'headless'}")

        try:
            time.sleep(2)
            self.ds_sender.enable_robot(autonomous=True)
            self.robot_enabled = True
            print("Start light detected - Robot enabled - Beginning AprilTag detection")

            # ── Main detection loop ───────────────────────────────────────
            while True:
                if self.task_done_sub.get():
                    print("RIO signaled sweep done - stopping DS keep alive")
                    self.ds_sender.disable_robot()
                    self.robot_enabled = False
                    break

                frame = self.picam2.capture_array()
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                self.send_ds_keepalive()

                tags = self.detect_tags(frame)
                self.publish_detections(tags)

                if self.display:
                    for tag in tags:
                        self.draw_detection(frame, tag)
                        print(f"Tag ID {tag.tag_id} | X:{tag.center[0]:.1f} Y:{tag.center[1]:.1f}"
                              + (f" Dist:{np.linalg.norm(tag.pose_t):.3f}m" if tag.pose_t is not None else ""))

                    # FPS overlay
                    fps_counter += 1
                    if fps_counter >= 30:
                        fps = fps_counter / (time.time() - fps_start_time)
                        fps_counter = 0
                        fps_start_time = time.time()

                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    nt_status = "NT: Connected" if self.nt_inst.isConnected() else "NT: Disconnected"
                    nt_color  = (0, 255, 0) if self.nt_inst.isConnected() else (0, 0, 255)
                    cv2.putText(frame, nt_status, (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, nt_color, 2)

                    robot_status = "ROBOT: ENABLED" if self.robot_enabled else "ROBOT: DISABLED"
                    robot_color  = (0, 255, 0) if self.robot_enabled else (0, 0, 255)
                    cv2.putText(frame, robot_status, (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, robot_color, 2)

                    if save_video and video_writer is not None:
                        video_writer.write(frame)

                    cv2.imshow('AprilTag Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            print("\nStopping detection...")

        finally:
            if self.robot_enabled:
                print("Disabling robot...")
                self.ds_sender.disable_robot()
                self.robot_enabled = False
            self.cleanup(video_writer)

    def cleanup(self, video_writer=None):
        if video_writer is not None:
            video_writer.release()
        self.picam2.stop()
        cv2.destroyAllWindows()
        self.nt_inst.stopClient()
        self.ds_sender.close()
        print("Cleanup complete")


def detect_start_light(frame):
    """
    Detect the start light in the camera frame.

    Args:
        frame: Camera frame (RGB888 from Picamera2)

    Returns:
        bool: True if start light is detected
    """
    roi = frame[50:150, 250:390]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(thresh) > 3000


if __name__ == "__main__":
    detector = AprilTagDetector(
        tag_family='tag36h11',  # Options: 'tag36h11', 'tag25h9', 'tag16h5'
        camera_params=None,     # Or provide [fx, fy, cx, cy] from calibration
        display=False           # Set True to enable camera feed and debug overlays
    )

    detector.run(save_video=False)
