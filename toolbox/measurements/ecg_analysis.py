from .heart_rate import heart_rate
from .pr_interval import pr_interval
from .qrs_duration import qrs_duration
from .qtc_interval import qtc_interval

def ecg_analysis(ecg_path: str) -> dict[str, str | None]:
    """Use ECG-Agent's display precision: HR/QTc 2 decimals, PR/QRS whole ms."""
    measurements = {
        "heart_rate": (heart_rate, 2),
        "pr_interval": (pr_interval, 0),
        "qrs_duration": (qrs_duration, 0),
        "qtc_interval": (qtc_interval, 2),
    }
    try:
        values = {name: (measure(ecg_path), digits)
                  for name, (measure, digits) in measurements.items()}
    except Exception:
        # ECG-Agent's CSV interface represents failed analyses as missing values.
        return dict.fromkeys(measurements)
    return {name: f"{value:.{digits}f}" if value is not None else None
            for name, (value, digits) in values.items()}
