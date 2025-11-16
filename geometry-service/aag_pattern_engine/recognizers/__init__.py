"""Feature recognizers package"""

from .hole_recognizer import HoleRecognizer, HoleFeature, HoleType
from .pocket_recognizer import PocketRecognizer, PocketFeature, PocketType
from .slot_recognizer import SlotRecognizer, SlotFeature, SlotType
from .boss_step_island_recognizer import (
    BossStepIslandRecognizer,
    BossStepIslandFeature,
    BossType,
    StepType,
    IslandType
)
from .fillet_chamfer_recognizer import (
    FilletRecognizer,
    ChamferRecognizer,
    FilletFeature,
    ChamferFeature,
    FilletType,
    ChamferType
)
from .turning_recognizer import (
    TurningRecognizer,
    TurningFeature,
    TurningFeatureType
)

__all__ = [
    "HoleRecognizer",
    "HoleFeature",
    "HoleType",
    "PocketRecognizer",
    "PocketFeature",
    "PocketType",
    "SlotRecognizer",
    "SlotFeature",
    "SlotType",
    "BossStepIslandRecognizer",
    "BossStepIslandFeature",
    "BossType",
    "StepType",
    "IslandType",
    "FilletRecognizer",
    "ChamferRecognizer",
    "FilletFeature",
    "ChamferFeature",
    "FilletType",
    "ChamferType",
    "TurningRecognizer",
    "TurningFeature",
    "TurningFeatureType",
]
