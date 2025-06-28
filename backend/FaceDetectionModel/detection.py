import cv2

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load video
cap = cv2.VideoCapture("backend/FaceDetectionModel/input.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

print("🎥 Video opened successfully.")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"📈 Total frames in video: {frame_count}")
print(f"⚙️  FPS: {fps}")

# Each face will be resized to 224x224 → total frame = 448x224
face_size = (224, 224)
out = cv2.VideoWriter("podcast_faces.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (448, 224))

frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("✅ Reached end of video or can't read next frame.")
        break

    frame_index += 1
    print(f"\n🧠 Processing frame: {frame_index}/{frame_count}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    print(f"👥 Faces detected: {len(faces)}")

    # Sort by area and pick top 2
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[:2]

    crops = []
    for i, (x, y, w, h) in enumerate(faces):
        pad = 20
        x1, y1 = max(x - pad, 0), max(y - pad, 0)
        x2, y2 = min(x + w + pad, frame.shape[1]), min(y + h + pad, frame.shape[0])
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, face_size)
        crops.append(crop)
        print(f"   ➤ Cropped face {i+1}: [{x1},{y1}] to [{x2},{y2}]")

    # Fill missing with black frame if only one or none found
    while len(crops) < 2:
        print("   ➤ Filling missing face with black frame.")
        crops.append(cv2.resize(frame, face_size))

    # Combine and write
    combined = cv2.hconcat(crops)
    out.write(combined)
    print("💾 Frame written to output.")

cap.release()
out.release()
print("✅ Video processing complete. Output saved as 'podcast_faces.mp4'")
