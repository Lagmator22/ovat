# ovat/agentic/hardware.py
"""Live CPU/GPU/NPU/RAM utilization sampling for interactive status bars.

Reuses telemetry sources from ``ovat/telemetry/sources.py``, omitting
unavailable metrics instead of reporting false zeros.
"""


class HardwareMonitor:
    def __init__(self, sources=None):
        if sources is None:
            from ovat.telemetry.sources import (GPUSource, NPUSource,
                                           SystemSource)
            sources = [SystemSource(), GPUSource(), NPUSource()]
        self._sources = [src for src in sources
                         if getattr(src, "unavailable", None) is None]
        self.last: dict = {}

    def sample(self) -> dict:
        """{"cpu": %, "ram": %, "gpu": %, "npu": %}, each only when known. Can take a
        few milliseconds (a Windows counter), so call it off the UI thread."""
        merged: dict = {}
        for src in self._sources:
            try:
                merged.update({f"{src.name}.{k}": v
                               for k, v in src.sample().items()})
            except Exception:
                continue
        out = {}
        for key, name in (("system.cpu_pct", "cpu"),
                          ("system.ram_used_pct", "ram"),
                          ("gpu.utilization", "gpu"),
                          ("npu.utilization", "npu")):
            if merged.get(key) is not None:
                out[name] = float(merged[key])
        self.last = out
        return out

    def close(self) -> None:
        for src in self._sources:
            try:
                src.stop()
            except Exception:
                pass


def _percent(value: float) -> str:
    # A busy machine rounded down to "0%" reads as a dead gauge.
    return "<1%" if 0 < value < 0.5 else f"{value:.0f}%"


def format_reading(reading: dict) -> str:
    """"CPU 23% · GPU 12% · NPU 41% · RAM 61%" (only the known ones)."""
    parts = [f"{label} {_percent(reading[key])}"
             for key, label in (("cpu", "CPU"), ("gpu", "GPU"), ("npu", "NPU"),
                        ("ram", "RAM"))
             if key in reading]
    return "  ·  ".join(parts)
