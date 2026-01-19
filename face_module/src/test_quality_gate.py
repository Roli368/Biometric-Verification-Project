import cv2
import time
from frame_quality import FrameQualityGate

# Initialize quality gate
quality_gate = FrameQualityGate()

# Open laptop camera
cap = cv2.VideoCapture(0)

bad_start_time = None

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Track continuous bad duration
    current_time = time.time()

    if bad_start_time is None:
        bad_duration = 0
    else:
        bad_duration = current_time - bad_start_time

    # Run quality gate
    result = quality_gate.evaluate(frame, bad_duration)

    # Handle bad quality timing
    if result["paused"]:
        if bad_start_time is None:
            bad_start_time = current_time
    else:
        bad_start_time = None

    # Display result on frame
    status = "GOOD" if result["ok"] else "PAUSED"
    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if result["ok"] else (0, 0, 255),
        2
    )

    if result["message"]:
        cv2.putText(
            frame,
            result["message"],
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.imshow("ML-3 Quality Gate Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()