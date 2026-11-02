from dataclasses import dataclass
from pathlib import Path
from joblib import Memory
import neurokit2 as nk
import numpy as np

cache_dir = Path.cwd() / ".eko-cache"
cache_dir.mkdir(mode=0o700, exist_ok=True)
cache = Memory(cache_dir / "measurements", verbose=0)
# ECG-Agent measurement limits.
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
        return nk.ecg_delineate(
            self.cleaned, self.r_peaks,
            sampling_rate=self.sampling_rate, method="dwt")[1]

def median(values: np.ndarray, minimum: float, maximum: float) -> float | None:
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values) & (values >= minimum) & (values <= maximum)]
    return float(np.median(valid)) if valid.size else None

def load_ecg(ecg_path: Path) -> tuple[np.ndarray, int, int]:
    """Load signal, sample rate, and measurement lead; defaults are 500 Hz and II."""
    from scipy.io import loadmat

    data = loadmat(ecg_path)
    signal = np.asarray(data["feats"], dtype=float)
    rate = int(np.asarray(data.get("curr_sample_rate", 500)).item())
    lead = int(np.asarray(data.get("measurement_lead", 1)).item())
    return signal, rate, lead

@cache.cache
def prepare(ecg_path: str, file_version: tuple, neurokit_version: str) -> ECG:
    signal, sampling_rate, measurement_lead = load_ecg(Path(ecg_path))
    lead = signal[measurement_lead]
    if np.all(lead == 0) or np.all(np.isnan(lead)):
        raise ValueError(f"Measurement lead {measurement_lead} contains only zeros or NaNs")

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
    cache.clear(warn=False)
