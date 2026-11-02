from .heart_rate import heart_rate
from .pr_interval import pr_interval
from .qrs_duration import qrs_duration
from .qtc_interval import qtc_interval

def ecg_analysis(ecg_path: str) -> dict[str, str | None]:
    """Use ECG-Agent's display precision: HR/QTc 2 decimals, PR/QRS whole ms."""
    measurements = {
        "heart_rate": (heart_rate(ecg_path), 2),
        "pr_interval": (pr_interval(ecg_path), 0),
        "qrs_duration": (qrs_duration(ecg_path), 0),
        "qtc_interval": (qtc_interval(ecg_path), 2),
    }
    return {name: f"{value:.{digits}f}" if value is not None else None
            for name, (value, digits) in measurements.items()}
