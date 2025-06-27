import cv2

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open video
cap = cv2.VideoCapture("input.mp4")

# Output settings
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width, frame_height = 224, 224  # target crop size
out = cv2.VideoWriter("cropped_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        # Choose the largest face (closest one)
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]

        # Add padding and crop
        pad = 20
        x1, y1 = max(x-pad, 0), max(y-pad, 0)
        x2, y2 = min(x+w+pad, frame.shape[1]), min(y+h+pad, frame.shape[0])
        face_crop = frame[y1:y2, x1:x2]

        # Resize to target and write
        face_crop = cv2.resize(face_crop, (frame_width, frame_height))
        out.write(face_crop)

cap.release()
out.release()
