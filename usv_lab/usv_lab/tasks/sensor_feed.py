"""Live camera and lidar for the dashboard, switched on and off while a run is going.

The dashboard's switch creates or removes the file ~/usv_feed/enabled. While it exists, a running
environment saves the first few boats' latest camera images and lidar scans there, twice a second of
simulated time. While it does not, the only cost is checking that the file exists at that same rate.
Files are replaced atomically, so the dashboard never reads a half-written one.
"""

from __future__ import annotations

import json
import math
import os
import struct
import time
import zlib
from pathlib import Path

import numpy as np

FEED_DIR = Path(os.environ.get("USV_FEED_DIR", "~/usv_feed")).expanduser()
ENABLED = FEED_DIR / "enabled"


def png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an H x W x 3 uint8 image as PNG."""
    height, width, _ = rgb.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))


def replace_file(path: Path, data: bytes) -> None:
    # one temporary name per process: two runs writing the feed at once must not move each other's files
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


class SensorFeed:
    def __init__(self, env, every_steps: int, boats: int):
        self.env, self.every, self.boats = env, max(1, every_steps), boats
        self.step = 0

    def maybe_publish(self) -> None:
        self.step += 1
        if self.step % self.every == 0 and ENABLED.exists():
            try:
                self.publish()
            except OSError as err:   # a viewing aid must never stop a simulation
                print(f"[sensor feed] skipped an update: {err}", flush=True)

    def publish(self) -> None:
        env = self.env
        count = min(self.boats, env.num_envs)
        meta = {"time": time.time(), "sim_time": env.sim_time, "task": type(env).__name__, "boats": count}
        if env.camera is not None:
            images = env.camera.data.output["rgb"].torch[:count, ..., :3].clamp(0, 255).byte().cpu().numpy()
            for i in range(count):
                replace_file(FEED_DIR / f"camera_{i}.png", png_bytes(images[i]))
            meta["camera"] = True
        if env.lidar is not None:
            directions = env.lidar.directions.cpu().numpy()
            ranges = env.lidar_ranges[:count].cpu().numpy()
            meta["lidar"] = {
                "max_range": env.cfg.lidar.max_range,
                "azimuth": [round(float(a), 4) for a in np.arctan2(directions[:, 1], directions[:, 0])],
                "elevation": [round(math.asin(float(z)), 4) for z in directions[:, 2]],
                "ranges": [[None if not math.isfinite(v) else round(float(v), 1) for v in row] for row in ranges],
            }
        replace_file(FEED_DIR / "meta.json", json.dumps(meta).encode())
