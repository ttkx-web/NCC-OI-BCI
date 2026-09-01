"""Compatibility imports for the three-mental-state application."""
from bci_dayloop.applications.three_mental_states.contract import HeadCheckpointInfo, HeadPrediction, TASK_OUTPUT_DIMS, ThreeMentalStateDiagnostics, ThreeMentalStatePrediction
from bci_dayloop.applications.three_mental_states.predictor import MultiHeadPredictor, ThreeMentalStatePredictor, _LoadedHead
MultiHeadPrediction = ThreeMentalStatePrediction
MultiHeadInferenceDiagnostics = ThreeMentalStateDiagnostics
