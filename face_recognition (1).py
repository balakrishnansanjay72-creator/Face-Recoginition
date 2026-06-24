import cv2
import time

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

DROIDCAM_IP = "192.168.1.5"
DROIDCAM_PORT = "4747"

url = f"http://{DROIDCAM_IP}:{DROIDCAM_PORT}/video"

print("Connecting to DroidCam...")
video = cv2.VideoCapture(url)

# Give stream time to stabilize
time.sleep(2)

if not video.isOpened():
    print("Could not connect to DroidCam.")
    exit()

print("Connected! Opening camera window...")
print("Press 'q' to quit.")

while True:
    ret, frame = video.read()

    if not ret or frame is None:
        print("Failed to grab frame, retrying...")
        time.sleep(0.5)
        continue

    # Resize for better performance
    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improved detection parameters
    # scaleFactor=1.1  (more sensitive than 1.3)
    # minNeighbors=5   (lower = detects more faces but more false positives)
    # minSize=(30,30)  (minimum face size to detect)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) > 0:
        print(f"Faces detected: {len(faces)}")

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Face Detected", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show face count on screen
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face Detection - DroidCam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
