import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from typing import List, Tuple, Dict

# MoveNet keypoint indices
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Skeleton connections for drawing
EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (0, 5), (0, 6), (5, 6),  # Shoulders to head
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16)  # Right leg
]


class PoseVisualizer:
    """
    Extracts pose keypoints using MoveNet and overlays skeleton on video frames.
    """
    
    def __init__(self, model_type='thunder', confidence_threshold=0.2):
        """
        Initialize MoveNet model.
        
        Args:
            model_type: 'thunder' (more accurate, 256x256) or 'lightning' (faster, 192x192)
            confidence_threshold: Minimum confidence for displaying keypoints
        """
        self.confidence_threshold = confidence_threshold
        self.model_type = model_type
        
        # Load MoveNet model
        model_url = f'https://tfhub.dev/google/movenet/singlepose/{model_type}/4'
        print(f"Loading MoveNet {model_type} model...")
        self.model = hub.load(model_url)
        self.movenet = self.model.signatures['serving_default']
        print(f"MoveNet {model_type} loaded successfully!")
        print(f"Input size: {'192x192' if model_type == 'lightning' else '256x256'}")
        
    def extract_pose(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract pose keypoints from a single frame.
        
        Args:
            frame: RGB image (H, W, 3)
            
        Returns:
            keypoints: Array of shape (17, 3) containing [y, x, confidence]
        """
        # MoveNet Lightning requires 192x192, Thunder requires 256x256
        input_size = 192 if self.model_type == 'lightning' else 256
        
        # Resize and pad to required size
        img = tf.expand_dims(frame, axis=0)
        img = tf.image.resize_with_pad(img, input_size, input_size)
        img = tf.cast(img, dtype=tf.int32)
        
        # Run inference using the 'input' keyword argument
        outputs = self.movenet(input=img)
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]
        
        return keypoints
    
    def draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, 
                      draw_connections=True, draw_keypoints=True) -> np.ndarray:
        """
        Draw skeleton overlay on frame.
        
        Args:
            frame: Image to draw on (will be copied)
            keypoints: Array of shape (17, 3) containing [y, x, confidence]
            draw_connections: Whether to draw lines between keypoints
            draw_keypoints: Whether to draw circles at keypoint locations
            
        Returns:
            Annotated frame
        """
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        keypoints_px = keypoints.copy()
        keypoints_px[:, 0] *= height  # y
        keypoints_px[:, 1] *= width   # x
        
        # Draw connections
        if draw_connections:
            for edge in EDGES:
                p1, p2 = edge
                y1, x1, c1 = keypoints_px[p1]
                y2, x2, c2 = keypoints_px[p2]
                
                # Only draw if both keypoints are confident
                if c1 > self.confidence_threshold and c2 > self.confidence_threshold:
                    cv2.line(overlay, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)),
                            (0, 255, 0), 2, cv2.LINE_AA)
        
        # Draw keypoints
        if draw_keypoints:
            for i, (y, x, conf) in enumerate(keypoints_px):
                if conf > self.confidence_threshold:
                    # Color code: high confidence = green, low = yellow
                    color = (0, 255, 0) if conf > 0.5 else (0, 255, 255)
                    cv2.circle(overlay, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
                    cv2.circle(overlay, (int(x), int(y)), 5, (0, 0, 0), 1, cv2.LINE_AA)
        
        return overlay
    
    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate angle between three points (in degrees).
        
        Args:
            p1, p2, p3: Points as [y, x, confidence]
            
        Returns:
            Angle in degrees at point p2
        """
        # Vector from p2 to p1
        v1 = p1[:2] - p2[:2]
        # Vector from p2 to p3
        v2 = p3[:2] - p2[:2]
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def get_golf_metrics(self, keypoints: np.ndarray) -> Dict[str, float]:
        """
        Calculate golf-specific metrics from pose keypoints.
        
        Args:
            keypoints: Array of shape (17, 3) containing [y, x, confidence]
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Get key points
        left_shoulder = keypoints[KEYPOINT_DICT['left_shoulder']]
        right_shoulder = keypoints[KEYPOINT_DICT['right_shoulder']]
        left_hip = keypoints[KEYPOINT_DICT['left_hip']]
        right_hip = keypoints[KEYPOINT_DICT['right_hip']]
        left_elbow = keypoints[KEYPOINT_DICT['left_elbow']]
        right_elbow = keypoints[KEYPOINT_DICT['right_elbow']]
        left_wrist = keypoints[KEYPOINT_DICT['left_wrist']]
        right_wrist = keypoints[KEYPOINT_DICT['right_wrist']]
        
        # Check confidence
        min_confidence = self.confidence_threshold
        
        # Shoulder-Hip Separation (X-Factor)
        if all(kp[2] > min_confidence for kp in [left_shoulder, right_shoulder, left_hip, right_hip]):
            # Calculate shoulder line angle
            shoulder_vector = right_shoulder[:2] - left_shoulder[:2]
            shoulder_angle = np.degrees(np.arctan2(shoulder_vector[0], shoulder_vector[1]))
            
            # Calculate hip line angle
            hip_vector = right_hip[:2] - left_hip[:2]
            hip_angle = np.degrees(np.arctan2(hip_vector[0], hip_vector[1]))
            
            # X-Factor is the difference
            metrics['hip_shoulder_separation'] = abs(shoulder_angle - hip_angle)
        
        # Left arm angle (shoulder-elbow-wrist)
        if all(kp[2] > min_confidence for kp in [left_shoulder, left_elbow, left_wrist]):
            metrics['left_arm_angle'] = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # Right arm angle
        if all(kp[2] > min_confidence for kp in [right_shoulder, right_elbow, right_wrist]):
            metrics['right_arm_angle'] = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Spine angle (shoulder center to hip center relative to vertical)
        if all(kp[2] > min_confidence for kp in [left_shoulder, right_shoulder, left_hip, right_hip]):
            shoulder_center = (left_shoulder[:2] + right_shoulder[:2]) / 2
            hip_center = (left_hip[:2] + right_hip[:2]) / 2
            
            # Angle from vertical
            spine_vector = hip_center - shoulder_center
            spine_angle = np.degrees(np.arctan2(spine_vector[1], spine_vector[0]))
            metrics['spine_angle'] = abs(90 - abs(spine_angle))  # Deviation from vertical
        
        # Lead arm to ground angle (useful for backswing analysis)
        # Assuming right-handed golfer - use left arm
        if all(kp[2] > min_confidence for kp in [left_shoulder, left_wrist]):
            arm_vector = left_wrist[:2] - left_shoulder[:2]
            # Angle from horizontal (ground)
            arm_to_ground = np.degrees(np.arctan2(arm_vector[0], arm_vector[1]))
            metrics['lead_arm_to_ground'] = abs(arm_to_ground)
        
        return metrics
    
    def annotate_frame_with_metrics(self, frame: np.ndarray, keypoints: np.ndarray, 
                                    event_name: str = "", confidence: float = 0.0) -> np.ndarray:
        """
        Draw skeleton and display golf metrics on frame.
        
        Args:
            frame: Image to annotate
            keypoints: Pose keypoints
            event_name: Name of the golf swing event
            confidence: Model confidence for this event
            
        Returns:
            Fully annotated frame
        """
        # Draw skeleton
        annotated = self.draw_skeleton(frame, keypoints)
        
        # Calculate metrics
        metrics = self.get_golf_metrics(keypoints)
        
        # Add text overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        y_offset = 30
        
        # Event name
        if event_name:
            cv2.putText(annotated, event_name, (10, y_offset), 
                       font, font_scale * 1.2, (0, 255, 0), thickness)
            y_offset += 35
        
        # Model confidence
        if confidence > 0:
            cv2.putText(annotated, f'Confidence: {confidence:.3f}', (10, y_offset),
                       font, font_scale, (0, 255, 255), thickness)
            y_offset += 30
        
        # Metrics
        y_offset += 10
        cv2.putText(annotated, 'Golf Metrics:', (10, y_offset),
                   font, font_scale * 0.8, (255, 255, 255), thickness - 1)
        y_offset += 25
        
        for metric_name, value in metrics.items():
            display_name = metric_name.replace('_', ' ').title()
            text = f'{display_name}: {value:.1f}°'
            cv2.putText(annotated, text, (10, y_offset),
                       font, font_scale * 0.7, (200, 200, 255), thickness - 1)
            y_offset += 25
        
        return annotated


def process_video_with_pose(video_path: str, output_path: str = None, 
                            event_frames: List[int] = None,
                            event_names: List[str] = None,
                            confidence_scores: List[float] = None):
    """
    Process video and overlay pose estimation on all frames or specific event frames.
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video (optional)
        event_frames: List of frame indices for key events
        event_names: Names of events corresponding to event_frames
        confidence_scores: Model confidence scores for each event
    """
    visualizer = PoseVisualizer(model_type='lightning')  # Use lightning for speed
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer if output path provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {total_frames} frames")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for MoveNet
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract pose
        keypoints = visualizer.extract_pose(frame_rgb)
        
        # Check if this is an event frame
        event_name = ""
        confidence = 0.0
        if event_frames and frame_idx in event_frames:
            idx = event_frames.index(frame_idx)
            if event_names and idx < len(event_names):
                event_name = event_names[idx]
            if confidence_scores and idx < len(confidence_scores):
                confidence = confidence_scores[idx]
        
        # Annotate frame
        annotated = visualizer.annotate_frame_with_metrics(
            frame, keypoints, event_name, confidence
        )
        
        # Write or display
        if writer:
            writer.write(annotated)
        
        # Display progress
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
        
        frame_idx += 1
    
    cap.release()
    if writer:
        writer.release()
        print(f"Saved output video to: {output_path}")
    
    print("Processing complete!")


if __name__ == '__main__':
    # Example usage
    video_path = 'test_video.mp4'
    output_path = 'output_with_pose.mp4'
    
    # Example with specific event frames
    event_frames = [10, 25, 40, 55, 70, 85, 100, 115]  # Replace with actual frames
    event_names = ['Address', 'Toe-up', 'Mid-backswing', 'Top', 
                   'Mid-downswing', 'Impact', 'Mid-follow-through', 'Finish']
    confidence_scores = [0.95, 0.87, 0.92, 0.88, 0.91, 0.96, 0.89, 0.93]
    
    process_video_with_pose(
        video_path=video_path,
        output_path=output_path,
        event_frames=event_frames,
        event_names=event_names,
        confidence_scores=confidence_scores
    )