import time
import numpy as np
import cv2
from typing import Dict, Any, Optional
import mediapipe as mp

# Constants for challenge steps
CHALLENGE_ORDER = ["TURN_LEFT", "TURN_RIGHT", "BLINK", "SMILE"]
STEP_HOLD_MS = 500  # How long condition must remain true (debounce)
STEP_TIMEOUT_MS = 12000  # Max time allowed per step
CALIBRATION_FRAMES = 10  # Number of frames to calibrate neutral position

# Thresholds
YAW_THRESHOLD_LEFT = -15.0  # degrees
YAW_THRESHOLD_RIGHT = 15.0  # degrees
EAR_BLINK_THRESHOLD = 0.2  # Eye Aspect Ratio threshold for blink
EAR_BLINK_FRAMES = 3  # Consecutive frames below threshold
SMILE_THRESHOLD = 0.6  # Mouth aspect ratio threshold for smile


def calculate_ear(eye_landmarks):
    """Calculate Eye Aspect Ratio (EAR)"""
    # Vertical distances
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    # Horizontal distance
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (v1 + v2) / (2.0 * h)
    return ear


def calculate_yaw(face_landmarks, img_w, img_h):
    """Estimate yaw angle from face landmarks"""
    # Using nose tip and face contour points
    nose_tip = face_landmarks[1]  # Nose tip
    left_face = face_landmarks[234]  # Left face contour
    right_face = face_landmarks[454]  # Right face contour
    
    # Calculate ratios
    left_dist = abs(nose_tip[0] - left_face[0])
    right_dist = abs(nose_tip[0] - right_face[0])
    
    # Normalize to [-1, 1] range, then convert to degrees
    if left_dist + right_dist == 0:
        return 0.0
    ratio = (right_dist - left_dist) / (right_dist + left_dist)
    yaw = ratio * 45.0  # Scale to approximate degrees
    return yaw


def calculate_smile(mouth_landmarks):
    """Calculate smile metric (mouth aspect ratio)"""
    # Mouth corners
    left_corner = mouth_landmarks[0]
    right_corner = mouth_landmarks[1]
    # Mouth top and bottom
    top = mouth_landmarks[2]
    bottom = mouth_landmarks[3]
    
    # Width and height
    width = np.linalg.norm(right_corner - left_corner)
    height = np.linalg.norm(top - bottom)
    
    if height == 0:
        return 0.0
    smile_ratio = width / height
    return smile_ratio


class UltimateLiveness10:
    """Deterministic liveness detection with fixed challenge sequence"""
    
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Challenge state
        self.challenge_index = 0
        self.challenge_order = CHALLENGE_ORDER.copy()
        
        # Step tracking
        self.step_started_at = time.time() * 1000  # milliseconds
        self.condition_met_since = None  # When condition first became true
        self.step_passed = False
        
        # Calibration
        self.calibrating = True
        self.calibration_frames_count = 0
        self.neutral_yaw_samples = []
        self.neutral_yaw = 0.0
        
        # Blink tracking
        self.blink_frame_count = 0
        self.blink_started = False
        
        # Metrics history for stability
        self.last_yaw = 0.0
        self.last_ear = 0.3
        self.last_smile = 0.0
        
    def verify(self, frame) -> Dict[str, Any]:
        """Process a single frame and return liveness status"""
        current_time_ms = time.time() * 1000
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        # Detect face landmarks
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return {
                "status": "PROCESSING",
                "trust": 0.0,
                "metrics": {
                    "command": self.get_current_command(),
                    "step": self.challenge_index + 1,
                    "total_steps": len(self.challenge_order),
                    "yaw": self.last_yaw,
                    "ear": self.last_ear,
                    "smile": self.last_smile,
                    "message": "No face detected"
                },
                "message": "Center your face in the frame"
            }
        
        # Extract landmarks
        face_landmarks = results.multi_face_landmarks[0]
        landmarks_array = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])
        
        # Calculate metrics
        # Left eye: 33, 160, 158, 133, 153, 144
        # Right eye: 362, 385, 387, 263, 373, 380
        left_eye = landmarks_array[[33, 160, 158, 133, 153, 144]]
        right_eye = landmarks_array[[362, 385, 387, 263, 373, 380]]
        ear_left = calculate_ear(left_eye)
        ear_right = calculate_ear(right_eye)
        ear = (ear_left + ear_right) / 2.0
        
        # Yaw calculation
        yaw = calculate_yaw(landmarks_array, w, h)
        
        # Smile calculation (mouth landmarks: 61, 291, 13, 14)
        mouth_landmarks = landmarks_array[[61, 291, 13, 14]]
        smile = calculate_smile(mouth_landmarks)
        
        # Update last known metrics
        self.last_yaw = yaw
        self.last_ear = ear
        self.last_smile = smile
        
        # CALIBRATION PHASE
        if self.calibrating:
            self.neutral_yaw_samples.append(yaw)
            self.calibration_frames_count += 1
            
            if self.calibration_frames_count >= CALIBRATION_FRAMES:
                self.neutral_yaw = np.mean(self.neutral_yaw_samples)
                self.calibrating = False
                self.step_started_at = current_time_ms
                
            return {
                "status": "CALIBRATING",
                "trust": 0.0,
                "metrics": {
                    "command": "LOOK_STRAIGHT",
                    "step": 0,
                    "total_steps": len(self.challenge_order),
                    "yaw": yaw,
                    "ear": ear,
                    "smile": smile,
                    "calibration_progress": self.calibration_frames_count / CALIBRATION_FRAMES
                },
                "message": f"Calibrating... ({self.calibration_frames_count}/{CALIBRATION_FRAMES})"
            }
        
        # CHALLENGE SEQUENCE
        if self.challenge_index >= len(self.challenge_order):
            # All challenges completed
            return {
                "status": "LIVE HUMAN",
                "trust": 1.0,
                "metrics": {
                    "command": "COMPLETE",
                    "step": len(self.challenge_order),
                    "total_steps": len(self.challenge_order),
                    "yaw": yaw,
                    "ear": ear,
                    "smile": smile
                },
                "message": "Liveness verification complete"
            }
        
        current_command = self.challenge_order[self.challenge_index]
        condition_met = False
        feedback_message = ""
        
        # Check if current step condition is met (with debounce)
        if current_command == "TURN_LEFT":
            adjusted_yaw = yaw - self.neutral_yaw
            condition_met = adjusted_yaw < YAW_THRESHOLD_LEFT
            feedback_message = f"Turn left (yaw: {adjusted_yaw:.1f}°)"
            
        elif current_command == "TURN_RIGHT":
            adjusted_yaw = yaw - self.neutral_yaw
            condition_met = adjusted_yaw > YAW_THRESHOLD_RIGHT
            feedback_message = f"Turn right (yaw: {adjusted_yaw:.1f}°)"
            
        elif current_command == "BLINK":
            # Blink detection: EAR must drop below threshold and then rise
            if ear < EAR_BLINK_THRESHOLD:
                if not self.blink_started:
                    self.blink_started = True
                    self.blink_frame_count = 1
                else:
                    self.blink_frame_count += 1
            else:
                # EAR above threshold
                if self.blink_started and self.blink_frame_count >= EAR_BLINK_FRAMES:
                    # Blink completed
                    condition_met = True
                    self.blink_started = False
                    self.blink_frame_count = 0
                else:
                    self.blink_started = False
                    self.blink_frame_count = 0
            feedback_message = f"Blink your eyes (EAR: {ear:.2f})"
            
        elif current_command == "SMILE":
            condition_met = smile > SMILE_THRESHOLD
            feedback_message = f"Smile (ratio: {smile:.2f})"
        
        # DEBOUNCE LOGIC: Condition must be stable for HOLD_MS
        if condition_met:
            if self.condition_met_since is None:
                # Condition just became true
                self.condition_met_since = current_time_ms
            else:
                # Check if held long enough
                hold_duration = current_time_ms - self.condition_met_since
                if hold_duration >= STEP_HOLD_MS:
                    # Step passed! Move to next challenge
                    self.challenge_index += 1
                    self.step_started_at = current_time_ms
                    self.condition_met_since = None
                    self.step_passed = True
                    
                    # Return immediate feedback
                    return {
                        "status": "PROCESSING",
                        "trust": self.challenge_index / len(self.challenge_order),
                        "metrics": {
                            "command": self.get_current_command(),
                            "step": self.challenge_index + 1,
                            "total_steps": len(self.challenge_order),
                            "yaw": yaw,
                            "ear": ear,
                            "smile": smile
                        },
                        "message": f"{current_command} passed! Next: {self.get_current_command()}"
                    }
                else:
                    feedback_message += f" - Hold for {int((STEP_HOLD_MS - hold_duration) / 1000) + 1}s"
        else:
            # Condition not met, reset debounce timer
            self.condition_met_since = None
        
        # Check timeout
        step_elapsed = current_time_ms - self.step_started_at
        if step_elapsed > STEP_TIMEOUT_MS:
            feedback_message = "HOLD LONGER - Keep position stable"
            # Reset step timer but don't change step
            self.step_started_at = current_time_ms
        
        # Return processing status
        return {
            "status": "PROCESSING",
            "trust": self.challenge_index / len(self.challenge_order),
            "metrics": {
                "command": current_command,
                "step": self.challenge_index + 1,
                "total_steps": len(self.challenge_order),
                "yaw": yaw,
                "ear": ear,
                "smile": smile
            },
            "message": feedback_message
        }
    
    def get_current_command(self) -> str:
        """Get the current challenge command"""
        if self.calibrating:
            return "LOOK_STRAIGHT"
        if self.challenge_index >= len(self.challenge_order):
            return "COMPLETE"
        return self.challenge_order[self.challenge_index]
    
    def reset(self):
        """Reset the liveness detector state"""
        self.challenge_index = 0
        self.step_started_at = time.time() * 1000
        self.condition_met_since = None
        self.step_passed = False
        self.calibrating = True
        self.calibration_frames_count = 0
        self.neutral_yaw_samples = []
        self.neutral_yaw = 0.0
        self.blink_frame_count = 0
        self.blink_started = False
