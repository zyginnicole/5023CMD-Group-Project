"""Wall-clock timing record: merges fields into <run_dir>/timing.json (measured, never estimated)."""
import json
import os
import platform


def update_timing(run_dir, **fields):
    path = os.path.join(run_dir, "timing.json")
    d = json.load(open(path)) if os.path.exists(path) else {}
    d.update(fields)
    d["python"] = platform.python_version()
    d["machine"] = f"{platform.system()} {platform.machine()}, {os.cpu_count()} logical CPUs"
    try:
        import torch
        d["torch"] = torch.__version__
        d["cuda"] = bool(torch.cuda.is_available())
    except ImportError:
        pass
    json.dump(d, open(path, "w"), indent=2)
    return d
