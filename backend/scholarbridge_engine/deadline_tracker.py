"""
ScholarBridge Deadline & Application Timeline Tracker
=====================================================
Processes application dates, window statuses, and timeline countdowns.
"""

from datetime import datetime, date
from typing import Dict, Optional, Any
from .models import Scholarship


class DeadlineTracker:
    """Computes timeline details and urgency for scholarships."""

    @staticmethod
    def get_timeline_info(scholarship: Scholarship, ref_date: Optional[date] = None) -> Dict[str, Any]:
        """Calculates application timeline details for a scholarship."""
        today = ref_date or date.today()
        start_str = scholarship.application_start_date
        end_str = scholarship.application_end_date
        status = scholarship.deadline_status
        
        info = {
            "scholarship_id": scholarship.scholarship_id,
            "short_name": scholarship.short_name,
            "application_cycle": scholarship.application_cycle,
            "start_date": start_str,
            "end_date": end_str,
            "status": status,
            "days_remaining": None,
            "is_open": False,
            "timeline_display": "Open for current cycle"
        }
        
        if scholarship.status == "Suspended":
            info["status"] = "SUSPENDED"
            info["timeline_display"] = "Scheme currently suspended nationwide"
            return info
            
        if status == "YEAR_ROUND":
            info["is_open"] = True
            info["timeline_display"] = "Applications accepted year-round"
            return info
            
        try:
            if start_str and end_str and start_str not in ["None", "NOT_APPLICABLE", "NOT_ANNOUNCED"]:
                s_date = datetime.strptime(start_str, "%Y-%m-%d").date()
                e_date = datetime.strptime(end_str, "%Y-%m-%d").date()
                
                if today < s_date:
                    days_to_open = (s_date - today).days
                    info["timeline_display"] = f"Opens in {days_to_open} days ({start_str})"
                elif s_date <= today <= e_date:
                    days_left = (e_date - today).days
                    info["is_open"] = True
                    info["days_remaining"] = days_left
                    info["timeline_display"] = f"Open now — {days_left} days left (Closes {end_str})"
                else:
                    info["timeline_display"] = f"Closed on {end_str} for current cycle"
        except Exception:
            info["timeline_display"] = f"Schedule: {scholarship.application_cycle}"
            
        return info
