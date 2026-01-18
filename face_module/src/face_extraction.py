import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extract_face_from_document(image_path, output_size=(160, 160)):
   

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Unable to read image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        raise ValueError("No face detected in document")

    
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    face_crop = image[y:y+h, x:x+w]

    # Resize face
    face_resized = cv2.resize(face_crop, output_size)

    return face_resized