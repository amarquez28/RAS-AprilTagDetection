import os
import cv2
from pyapriltags import Detector
import numpy as np
import time
from ntcore import NetworkTableInstance
import socket
import struct

try:
    from picamera2 import Picamera2
except ImportError:
    # mock for testing on my mac or any non pi systems
    class Picamera2:
        def configure(self, config):
            pass
        def start(self):
            print("[Mock] Camera started")
        def capture_array(self):
            import numpy as np
            return np.zeros((480, 640, 3), dtype=np.uint8)  # blank frame

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration_params.npz')
display = True


def calibrate(picam2, board_size=(9, 6), square_size=1.0, min_frames=15, capture_interval=2.0):
    """
    Calibrate camera from live Picamera2 feed with a printed chessboard.

    Hold the chessboard in front of the camera at different angles/distances.
    The function captures a frame every `capture_interval` seconds, checks for
    the chessboard, and collects until `min_frames` good detections are gathered.
    Press 'q' to stop early (if at least 5 frames collected).

    Args:
        picam2:           Already-started Picamera2 instance.
        board_size:       Inner corners of the chessboard (cols, rows).
        square_size:      Size of one square in your chosen unit (e.g. mm or cm).
        min_frames:       Number of good chessboard frames to collect.
        capture_interval: Seconds between capture attempts (gives you time to move the board).

    Returns:
        (camMatrix, distCoeff) on success, or None on failure.
    """
    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    world_pts = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    world_pts[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    world_pts *= square_size

    world_pts_list = []
    img_pts_list = []
    used_count = 0

    print(f"[calibrate] Hold chessboard ({board_size[0]}x{board_size[1]}) in front of camera.")
    print(f"[calibrate] Capturing every {capture_interval}s. Need {min_frames} good frames. Press 'q' to finish early.")

    last_capture = 0.0

    while True:
        frame = picam2.capture_array()
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        now = time.time()
        status_text = f"Collected: {used_count}/{min_frames}"

        if now - last_capture >= capture_interval:
            last_capture = now
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(frame_gray, board_size, None)

            if found:
                corners_refined = cv2.cornerSubPix(frame_gray, corners, (11, 11), (-1, -1), term_criteria)
                world_pts_list.append(world_pts)
                img_pts_list.append(corners_refined)
                used_count += 1
                print(f"  chessboard found ({used_count}/{min_frames})")
                cv2.drawChessboardCorners(frame, board_size, corners_refined, found)
                status_text = f"CAPTURED {used_count}/{min_frames}"
            else:
                status_text = f"No board detected ({used_count}/{min_frames})"

            if used_count >= min_frames:
                break

        cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow("Calibration")

    if used_count < 5:
        print(f"[calibrate] ERROR: only collected {used_count} frames (need at least 5)")
        return None

    print(f"[calibrate] Running calibration with {used_count} frames...")
    rep_error, cam_matrix, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(
        world_pts_list, img_pts_list, frame_gray.shape[::-1], None, None
    )

    print(f"  Camera matrix:\n{cam_matrix}")
    print(f"  Reprojection error: {rep_error:.4f} pixels")

    np.savez(CALIBRATION_FILE,
             repError=rep_error, camMatrix=cam_matrix, distCoeff=dist_coeff,
             rvecs=rvecs, tvecs=tvecs)
    print(f"  Saved to {CALIBRATION_FILE}")

    return cam_matrix, dist_coeff


def load_calibration():
    """Load calibration from file. Returns (camMatrix, distCoeff) or None."""
    if not os.path.exists(CALIBRATION_FILE):
        return None
    data = np.load(CALIBRATION_FILE)
    print(f"[calibration] Loaded from {CALIBRATION_FILE} (reprojection error: {data['repError']:.4f}px)")
    return data['camMatrix'], data['distCoeff']


def undistort_frame(frame, cam_matrix, dist_coeff):
    """Remove lens distortion from a frame."""
    h, w = frame.shape[:2]
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coeff, (w, h), 1, (w, h))
    return cv2.undistort(frame, cam_matrix, dist_coeff, None, new_matrix)
    
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
    def __init__(self, tag_family='tag36h11', camera_params=None, roborio_ip='10.0.67.2'):
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

        # Camera parameters for IMX296 after 90° CCW rotation.
        #
        # The physical sensor captures 1480 (W) × 1110 (H).
        # After cv2.ROTATE_90_COUNTERCLOCKWISE the frame becomes 1110 (W) × 1480 (H).
        # The principal point axes swap accordingly:
        #   cx (horizontal centre) = 1110 / 2 = 555
        #   cy (vertical centre)   = 1480 / 2 = 740
        # fx and fy are swapped to match the new axis assignment.
        # These are uncalibrated estimates — proper camera calibration will
        # improve pose accuracy significantly.
        #
        # pose_t from pyapriltags is in the rotated camera's coordinate frame:
        #   pose_t[0] (tx) = horizontal offset in rotated image (+ = right in image)
        #   pose_t[1] (ty) = vertical offset in rotated image (+ = down in image)
        #   pose_t[2] (tz) = depth along optical axis (+ = away from camera)
        # The camera is rear-mounted, so "away from camera" = toward the tag.
        # The RIO must combine these with the robot's current heading (theta)
        # to convert into field-relative coordinates.
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

        # Primary tag pose (camera-relative, metres)
        self.tag_pose_tx_entry       = self.vision_table.getDoubleTopic("tag_pose_tx").publish()
        self.tag_pose_ty_entry       = self.vision_table.getDoubleTopic("tag_pose_ty").publish()
        self.tag_pose_tz_entry       = self.vision_table.getDoubleTopic("tag_pose_tz").publish()
        self.tag_pose_err_entry      = self.vision_table.getDoubleTopic("tag_pose_err").publish()
        self.tag_decision_margin_entry = self.vision_table.getDoubleTopic("tag_decision_margin").publish()

        # All tags as arrays
        self.tags_ids_entry       = self.vision_table.getIntegerArrayTopic("tags_ids").publish()
        self.tags_x_entry         = self.vision_table.getDoubleArrayTopic("tags_x").publish()
        self.tags_y_entry         = self.vision_table.getDoubleArrayTopic("tags_y").publish()
        self.tags_distances_entry = self.vision_table.getDoubleArrayTopic("tags_distances").publish()
        self.tags_pose_tx_entry   = self.vision_table.getDoubleArrayTopic("tags_pose_tx").publish()
        self.tags_pose_ty_entry   = self.vision_table.getDoubleArrayTopic("tags_pose_ty").publish()
        self.tags_pose_tz_entry   = self.vision_table.getDoubleArrayTopic("tags_pose_tz").publish()

        # Capture timestamp in RIO FPGA time (microseconds), converted via NT4 clock sync
        self.tag_timestamp_entry = self.vision_table.getIntegerTopic("tag_timestamp_us").publish()

        self.heartbeat_entry   = self.vision_table.getIntegerTopic("heartbeat").publish()
        self.heartbeat_counter = 0
        self.start_light_entry = self.vision_table.getBooleanTopic("start_light_detected").publish()

        print(f"NetworkTables initialized, connecting to roboRIO at {roborio_ip}")

    def capture_rio_timestamp(self):
        """
        Record a timestamp at frame capture and convert to RIO FPGA time
        using NT4's built-in clock synchronization.

        Returns RIO-relative timestamp in microseconds, or -1 if not synced yet.
        """
        local_time_us = time.monotonic_ns() // 1000
        offset = self.nt_inst.getServerTimeOffset()
        if offset is None:
            return -1
        return local_time_us + offset

    def publish_detections(self, tags, capture_timestamp_us):
        self.heartbeat_counter += 1
        self.heartbeat_entry.set(self.heartbeat_counter)
        self.tag_count_entry.set(len(tags))
        self.tag_timestamp_entry.set(capture_timestamp_us)

        if tags:
            primary = tags[0]
            self.tag_detected_entry.set(True)
            self.tag_id_entry.set(int(primary.tag_id))
            self.tag_x_entry.set(float(primary.center[0]))
            self.tag_y_entry.set(float(primary.center[1]))
            dist = float(np.linalg.norm(primary.pose_t)) if primary.pose_t is not None else -1.0
            self.tag_distance_entry.set(dist)
            self.tag_decision_margin_entry.set(float(primary.decision_margin))

            if primary.pose_t is not None:
                self.tag_pose_tx_entry.set(float(primary.pose_t[0]))
                self.tag_pose_ty_entry.set(float(primary.pose_t[1]))
                self.tag_pose_tz_entry.set(float(primary.pose_t[2]))
                self.tag_pose_err_entry.set(float(primary.pose_err))
            else:
                self.tag_pose_tx_entry.set(0.0)
                self.tag_pose_ty_entry.set(0.0)
                self.tag_pose_tz_entry.set(0.0)
                self.tag_pose_err_entry.set(-1.0)

            self.tags_ids_entry.set([int(t.tag_id) for t in tags])
            self.tags_x_entry.set([float(t.center[0]) for t in tags])
            self.tags_y_entry.set([float(t.center[1]) for t in tags])
            self.tags_distances_entry.set([
                float(np.linalg.norm(t.pose_t)) if t.pose_t is not None else -1.0
                for t in tags
            ])
            self.tags_pose_tx_entry.set([float(t.pose_t[0]) if t.pose_t is not None else 0.0 for t in tags])
            self.tags_pose_ty_entry.set([float(t.pose_t[1]) if t.pose_t is not None else 0.0 for t in tags])
            self.tags_pose_tz_entry.set([float(t.pose_t[2]) if t.pose_t is not None else 0.0 for t in tags])
        else:
            self.tag_detected_entry.set(False)
            self.tag_id_entry.set(-1)
            self.tag_x_entry.set(0.0)
            self.tag_y_entry.set(0.0)
            self.tag_distance_entry.set(-1.0)
            self.tag_pose_tx_entry.set(0.0)
            self.tag_pose_ty_entry.set(0.0)
            self.tag_pose_tz_entry.set(0.0)
            self.tag_pose_err_entry.set(-1.0)
            self.tag_decision_margin_entry.set(0.0)
            self.tags_ids_entry.set([])
            self.tags_x_entry.set([])
            self.tags_y_entry.set([])
            self.tags_distances_entry.set([])
            self.tags_pose_tx_entry.set([])
            self.tags_pose_ty_entry.set([])
            self.tags_pose_tz_entry.set([])

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
        """Draw a 3D cube on the tag to visualize its pose. Only called in display mode."""
        if tag.pose_R is None or tag.pose_t is None:
            return

        half = 0.1 / 2  # half of tag_size (0.1m)
        cube_height = 0.1  # cube extends 10cm out from the tag face

        # 8 corners of a cube: bottom face sits on the tag, top face extends toward camera
        cube_pts = np.float32([
            [-half, -half, 0],  [half, -half, 0],
            [half,  half, 0],   [-half,  half, 0],
            [-half, -half, -cube_height], [half, -half, -cube_height],
            [half,  half, -cube_height],  [-half,  half, -cube_height],
        ])

        cam_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

        rvec, _ = cv2.Rodrigues(tag.pose_R)
        tvec = tag.pose_t.reshape(3, 1)

        img_pts, _ = cv2.projectPoints(cube_pts, rvec, tvec, cam_matrix, None)
        pts = img_pts.reshape(-1, 2).astype(int)

        # Draw bottom face (green, on the tag)
        cv2.drawContours(image, [pts[:4]], -1, (0, 255, 0), 2)
        # Draw top face (red, floating above)
        cv2.drawContours(image, [pts[4:]], -1, (0, 0, 255), 2)
        # Draw vertical pillars (blue)
        for i in range(4):
            cv2.line(image, tuple(pts[i]), tuple(pts[i + 4]), (255, 0, 0), 2)

        # Tag ID and distance text
        center = tuple(tag.center.astype(int))
        cv2.putText(image, f"ID: {tag.tag_id}",
                    (center[0] - 20, center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        distance = np.linalg.norm(tag.pose_t)
        cv2.putText(image, f"Dist: {distance:.2f}m",
                    (center[0] - 20, center[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    def run(self, cam_matrix=None, dist_coeff=None):
        """
        Run the detection loop.

        Args:
            cam_matrix: Camera matrix from calibration (None to skip undistortion).
            dist_coeff: Distortion coefficients from calibration (None to skip undistortion).
        """
        # FPS tracking - display mode only
        fps = 0
        fps_counter = 0
        fps_start_time = time.time()

        use_undistort = cam_matrix is not None and dist_coeff is not None
        if use_undistort:
            print("Lens distortion correction: ENABLED")
        else:
            print("Lens distortion correction: DISABLED (no calibration data)")

        print("Starting AprilTag detection...")
        print(f"NetworkTables connected: {self.nt_inst.isConnected()}")
        print(f"Mode: {'display' if self.display else 'headless'}")

        try:
            time.sleep(2)
            self.ds_sender.enable_robot(autonomous=True)
            self.robot_enabled = True

            # ── Main detection loop ───────────────────────────────────────
            while True:
                if self.task_done_sub.get():
                    print("RIO signaled task done - stopping DS keepalive")
                    self.ds_sender.disable_robot()
                    self.robot_enabled = False
                    break

                frame = self.picam2.capture_array()
                capture_timestamp_us = self.capture_rio_timestamp()
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                if use_undistort:
                    frame = undistort_frame(frame, cam_matrix, dist_coeff)

                self.send_ds_keepalive()

                tags = self.detect_tags(frame)
                self.publish_detections(tags, capture_timestamp_us)

                if tags:
                    for tag in tags:
                        if tag.pose_t is not None:
                            tx, ty, tz = float(tag.pose_t[0]), float(tag.pose_t[1]), float(tag.pose_t[2])
                            dist = np.linalg.norm(tag.pose_t)
                            print(f"[NT] id={tag.tag_id}  "
                                  f"px=({tag.center[0]:.1f}, {tag.center[1]:.1f})  "
                                  f"pose=({tx:.3f}, {ty:.3f}, {tz:.3f})m  "
                                  f"dist={dist:.3f}m  "
                                  f"err={tag.pose_err:.4f}  "
                                  f"margin={tag.decision_margin:.1f}  "
                                  f"ts={capture_timestamp_us}")
                        else:
                            print(f"[NT] id={tag.tag_id}  "
                                  f"px=({tag.center[0]:.1f}, {tag.center[1]:.1f})  "
                                  f"pose=N/A  ts={capture_timestamp_us}")
                elif self.heartbeat_counter % 50 == 0:
                    print(f"[NT] no tags | heartbeat={self.heartbeat_counter} | ts={capture_timestamp_us}")

                if self.display:
                    for tag in tags:
                        self.draw_detection(frame, tag)

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

    def cleanup(self):
        self.picam2.stop()
        cv2.destroyAllWindows()
        self.nt_inst.stopClient()
        self.ds_sender.close()
        print("Cleanup complete")


if __name__ == "__main__":
    import sys

    # --- Calibration mode: python apriltag.py --calibrate [cols rows] ---
    if len(sys.argv) >= 2 and sys.argv[1] == '--calibrate':
        board = (9, 6)  # default chessboard inner corners
        if len(sys.argv) >= 4:
            board = (int(sys.argv[2]), int(sys.argv[3]))

        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (1480, 1110), "format": "RGB888"},
            controls={"FrameRate": 30}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        result = calibrate(picam2, board_size=board)
        picam2.stop()
        if result is None:
            print("Calibration failed.")
            sys.exit(1)
        print("Calibration complete.")
        sys.exit(0)

    # --- Detection mode ---
    calib = load_calibration()
    cam_matrix = None
    dist_coeff = None
    camera_params = None

    if calib is not None:
        cam_matrix, dist_coeff = calib
        # Extract fx, fy, cx, cy from calibrated camera matrix for AprilTag pose estimation
        camera_params = [cam_matrix[0, 0], cam_matrix[1, 1], cam_matrix[0, 2], cam_matrix[1, 2]]
    else:
        print("WARNING: No calibration file found. Running with estimated camera params and no undistortion.")

    detector = AprilTagDetector(
        tag_family='tag36h11',
        camera_params=camera_params,
    )

    try:
        detector.run(cam_matrix, dist_coeff)
    finally:
        detector.cleanup()

