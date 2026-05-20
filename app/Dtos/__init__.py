"""
Burner worker DTOs.

  TaskDto/
    BurnTask    (+ Bbox, BurnFrame, BurnFacePrediction)   orchestrator → burner
    BurnResult                                            burner → orchestrator
"""
from app.Dtos.TaskDto import BurnTask, BurnFrame, BurnFacePrediction, BurnResult

__all__ = ["BurnTask", "BurnFrame", "BurnFacePrediction", "BurnResult"]