import cv2
from face_extraction import extract_face_from_document

image_path = "sample_document.jpg"

face = extract_face_from_document(image_path)

cv2.imshow("Extracted Face", face)
cv2.waitKey(0)
cv2.destroyAllWindows()