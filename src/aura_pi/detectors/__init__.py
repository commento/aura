from .base import Detection, Detector
from .hailo_person import HailoPersonDetector
from .motion_people import MotionPeopleDetector

__all__ = ["Detection", "Detector", "HailoPersonDetector", "MotionPeopleDetector"]
