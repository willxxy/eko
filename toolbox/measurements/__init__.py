"""ECG measurement tools available to the agent."""

from .ecg import clear_cache
from .ecg_analysis import ecg_analysis
from .heart_rate import heart_rate
from .pr_interval import pr_interval
from .qrs_duration import qrs_duration
from .qtc_interval import qtc_interval

TOOLS = {
    "ecg_analysis": ecg_analysis,
    "heart_rate": heart_rate,
    "pr_interval": pr_interval,
    "qrs_duration": qrs_duration,
    "qtc_interval": qtc_interval,
}
