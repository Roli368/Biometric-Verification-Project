import cv2
import numpy as np
import time


class FrameQualityGate:
    """
    ML-3 Quality Gate
    Decides whether a frame is good enough to be consumed
    by downstream liveness and face match models.
    """

    def evaluate(self, frame, bad_duration_sec):
        """
        Args:
            frame (np.ndarray): BGR image frame
            bad_duration_sec (float): continuous seconds of bad quality

        Returns:
            dict: {
                "ok": bool,
                "paused": bool,
                "message": str or None
            }
        """

        # 1️⃣ Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2️⃣ Lighting check
        brightness = gray.mean()
        lighting_ok = 60 <= brightness <= 200

        # 3️⃣ Blur check
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_ok = blur_score >= 100

        # 4️⃣ Final decision
        ok = lighting_ok and blur_ok

        if ok:
            return {
                "ok": True,
                "paused": False,
                "message": None
            }

        # 5️⃣ User-friendly message escalation
        if bad_duration_sec < 30:
            message = "Please improve lighting"
        elif bad_duration_sec < 90:
            message = "Still having trouble — try brighter area"
        else:
            message = "Conditions difficult — we are waiting patiently"

        return {
            "ok": False,
            "paused": True,
            "message": message
        }

