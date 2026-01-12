import cv2
from pyapriltags import Detector

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Set resolution explicitly; required for many Pi cameras to start streaming
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")
else:
    print("camera is working")

at_detector = Detector(searchpath=['apriltags'],
                        families='tag36h11',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #detect tags
        results = at_detector.detect(gray)

        #draw detections
        for r in results:
            #r.corners is 4x2 float array: [top-left, top-right, bottom-right, bottom-left]
            corners = r.corners.astype(int)
            (ptA, ptB, ptC, ptD) = corners

            #outline
            cv2.line(frame, tuple(ptA), tuple(ptB), (0,255,0), 2)
            cv2.line(frame, tuple(ptB), tuple(ptC), (0,255,0), 2)
            cv2.line(frame, tuple(ptC), tuple(ptD), (0,255,0), 2)
            cv2.line(frame, tuple(ptD), tuple(ptA), (0,255,0), 2)
            

            #center
            cX, cY = r.center.astype(int)
            cv2.circle(frame, (cX, cY), 4, (0,0,255), -1)
            
            #tag ID
            tag_id = r.tag_id
            cv2.putText(frame, f"ID: {tag_id}", (ptA[0],ptA[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        cv2.imshow("AprilTag", frame)

        #exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()