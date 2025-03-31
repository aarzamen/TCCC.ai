"""
TCCC.ai Display Visualization Components
--------------------------------------
Collection of visualization components for the TCCC.ai display module:
- vital_signs: Real-time visualization of patient vital signs
- timeline: Timeline visualization of significant events
- injury_map: Body map visualization for injuries
- charts: General medical charting components
"""

from .vital_signs import VitalSignsMonitor

__all__ = ['VitalSignsMonitor']