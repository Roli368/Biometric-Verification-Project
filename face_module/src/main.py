import cv2
from processor import is_blurry, bad_lighting, draw_warning
from face_utils import detect_faces

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not working")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blurry, blur_val = is_blurry(frame)
    bad_light, light_val = bad_lighting(frame)

    status = "FRAME OK"

    if blurry:
        status = f"BLURRY FRAME (Score: {int(blur_val)})"
        draw_warning(frame, status)

    elif bad_light:
        status = f"BAD LIGHTING (Score: {int(light_val)})"
        draw_warning(frame, status)

    else:
        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Input Quality Check", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()