from dataclasses import dataclass
from pathlib import Path
from joblib import Memory
import neurokit2 as nk
import numpy as np

cache_dir = Path.cwd() / ".eko-cache"
cache_dir.mkdir(mode=0o700, exist_ok=True)
cache = Memory(cache_dir / "measurements", verbose=0)
# Peak minimum, interval limits, and inclusive sample counts follow the supplied extractor.
MIN_PEAKS = 7

@dataclass
class ECG:
    cleaned: np.ndarray
    sampling_rate: int
    r_peaks: np.ndarray
    rr_ms: np.ndarray
    median_rr_ms: float | None

    @property
    @cache.cache
    def waves(self) -> dict:
        """Compute and cache wave landmarks only when needed."""
        return nk.ecg_delineate(
            self.cleaned, self.r_peaks,
            sampling_rate=self.sampling_rate, method="dwt")[1]

def median(values: np.ndarray, minimum: float, maximum: float) -> float | None:
    """Return the median within the supplied extractor's limits, or None."""
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values) & (values >= minimum) & (values <= maximum)]
    return float(np.median(valid)) if valid.size else None

def load_ecg(ecg_path: Path) -> tuple[np.ndarray, int | None]:
    """Return (signal, sampling_rate_hz), with signal shaped (leads, samples).

    Rows must start with leads I and II; measurements use lead II.
    Return None for an unknown sampling rate; measurements then use 500 Hz.
    """
    from scipy.io import loadmat

    data = loadmat(ecg_path)
    signal = np.asarray(data["feats"], dtype=float)
    rate = data.get("curr_sample_rate")
    if rate is None:
        return signal, None
    rate = np.asarray(rate)
    if rate.size != 1:
        raise ValueError("curr_sample_rate must be a scalar")
    rate = float(rate.item())
    if not np.isfinite(rate) or rate <= 0 or not rate.is_integer():
        raise ValueError("curr_sample_rate must be a positive integer")
    return signal, int(rate)

@cache.cache
def prepare(ecg_path: str, file_version: tuple, neurokit_version: str) -> ECG:
    # File metadata and NeuroKit version invalidate cached preprocessing.
    signal, sampling_rate = load_ecg(Path(ecg_path))
    if sampling_rate is None:
        sampling_rate = 500
    signal = np.asarray(signal, dtype=float)
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive and finite")
    if signal.ndim != 2 or signal.shape[0] < 2:
        raise ValueError("Expected (leads, samples), with leads I and II first")

    lead = signal[1]
    if np.all(lead == 0) or np.all(np.isnan(lead)):
        raise ValueError("Lead II contains only zeros or NaNs")

    cleaned = nk.ecg_clean(lead, sampling_rate=sampling_rate, method="neurokit")
    info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)[1]
    r_peaks = np.asarray(info["ECG_R_Peaks"], dtype=float)
    if r_peaks.size < MIN_PEAKS:
        raise ValueError(
            f"At least {MIN_PEAKS} R-peaks are required; found {r_peaks.size}")

    rr_ms = np.diff(r_peaks) / sampling_rate * 1000
    median_rr = median(rr_ms[rr_ms < 2000], 300, 2000)
    return ECG(cleaned, sampling_rate, r_peaks, rr_ms, median_rr)

def get_ecg(ecg_path: str) -> ECG:
    """Reuse preprocessing for an unchanged file in this working directory."""
    path = Path(ecg_path).expanduser().resolve(strict=True)
    stat = path.stat()
    file_version = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
    return prepare(str(path), file_version, nk.__version__)

def intervals(waves: dict, start: str, end: str, sampling_rate: int,
              *, require_increasing: bool = False) -> np.ndarray:
    """Drop missing landmark pairs; only QRS requires end > start in the original."""
    starts = np.asarray(waves.get(start, []), dtype=float)
    ends = np.asarray(waves.get(end, []), dtype=float)
    count = min(starts.size, ends.size)
    starts, ends = starts[:count], ends[:count]
    valid = ~np.isnan(starts) & ~np.isnan(ends)
    if require_increasing:
        valid &= ends > starts
    return (ends[valid] - starts[valid] + 1) / sampling_rate * 1000

def clear_cache() -> None:
    """Clear cached ECG data after changing the loader or processing helpers."""
    cache.clear(warn=False)
