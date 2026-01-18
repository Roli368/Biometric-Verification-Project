import cv2
import time
import numpy as np
from processor import run_guardrails, align_face

cap = cv2.VideoCapture(0)

print("--- Unified System Active. Press 'q' to exit. ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    start_time = time.time()
    
    passed, message = run_guardrails(frame)
    
    aligned_frame = align_face(frame)
    
    if not passed or aligned_frame is None:
        
        display_right = np.zeros_like(frame)
        cv2.putText(display_right, "REJECTED: " + message, (20, frame.shape[0]//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        display_right = aligned_frame
        cv2.putText(display_right, "ALIGNED & STABILIZED", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    combined_view = np.hstack((frame, display_right))
    
    proc_time = (time.time() - start_time) * 1000
    cv2.putText(combined_view, f"Latency: {proc_time:.1f}ms", (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Technical Workflow: Original (Left) vs. Fortress (Right)", combined_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()