
import cv2

# Load pretrained Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_eye.xml"
)

# Check that the classifiers loaded correctly
if face_cascade.empty() or eye_cascade.empty():
    raise RuntimeError("Could not load Haar Cascade classifiers.")

# Open the default webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Cannot access webcam. Check camera permissions."
    )

print("Face and Eye Detection started.")
print("Press Q to quit.")

while True:
    # Read a frame from the webcam
    success, frame = camera.read()

    if not success:
        print("Could not read webcam frame.")
        break

    # Mirror the image for a natural webcam preview
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Detect eyes within each detected face
    for (x, y, w, h) in faces:
        # Draw a green rectangle around the face
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Search for eyes only inside the face region
        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(15, 15)
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                face_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (255, 0, 0),
                2
            )

        # Display a face label
        cv2.putText(
            frame,
            "Face Detected",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Show the number of faces detected
    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Display the live video
    cv2.imshow("Enhanced Face and Eye Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close the window
camera.release()
cv2.destroyAllWindows()
