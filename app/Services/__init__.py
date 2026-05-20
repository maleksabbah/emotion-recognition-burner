"""
Burner services.

  Overlay      per-frame drawing (text, boxes)        ← existing
  Stitcher     ffmpeg-style frame-by-frame stitch     ← existing
  BurnService  fetch → stitch → save → result
"""
from app.Services.BurnService import BurnService

__all__ = ["BurnService"]