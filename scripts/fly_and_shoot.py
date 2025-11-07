import os
import io
import csv
import sys
import time
import math
import asyncio
import argparse
from pathlib import Path
from typing import Optional, Tuple

import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import cv2
from PIL import Image

from mavsdk import System
from mavsdk.offboard import (OffboardError, VelocityNedYaw)
from mavsdk.geofence import Point
from mavsdk import telemetry

from dedupe import ShotDB, ShotMeta, decide_duplicate, append_csv
from nbv_planner import sample_views, pick_next_best
from gimbal_control import Gimbal
from rangefinder_reader import Rangefinder

# ---------- Helpers ----------
def ensure_dirs(photo_dir: str, log_csv: str):
    Path(photo_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(log_csv)).mkdir(parents=True, exist_ok=True)

def euler_deg_from_quat(q):
    """
    q: telemetry.Quaternion
    """
    # Converts NED quaternion to yaw/pitch/roll (deg).
    # PX4 uses FRD (North-East-Down); MAVSDK returns in NED.
    # We'll compute yaw (psi), pitch (theta), roll (phi).
    import numpy as np
    w, x, y, z = q.w, q.x, q.y, q.z
    # yaw
    t0 = +2.0 * (w * z + x * y)
    t1 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.degrees(math.atan2(t0, t1))
    # pitch
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.degrees(math.asin(t2))
    # roll
    t3 = +2.0 * (w * x + y * z)
    t4 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.degrees(math.atan2(t3, t4))
    return yaw, pitch, roll

def ned_to_xyz(north_m, east_m, down_m):
    # Convert NED to a simple XYZ (x=east, y=north, z=up)
    return east_m, north_m, -down_m

def now_ts():
    return time.time()

def save_frame(frame, out_path: str):
    cv2.imwrite(out_path, frame)

def capture_shots_via_webcam(n=12, device=0, out_dir="shots", interval_s=1.0):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(device)  # try 0, 1, 2...
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {device}. Try a different index or grant Camera permission.")

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Camera read failed; retrying...")
            time.sleep(0.2)
            continue
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(out_dir, f"shot_{ts}_{i:03d}.jpg")
        cv2.imwrite(path, frame)
        print(f"[SHOT] Saved {path}")
        time.sleep(interval_s)
    cap.release()

# ---------- Photo capture ----------
class PhotoManager:
    def __init__(self, photo_dir: str, use_webcam: bool, webcam_index: int, save_full_res: bool):
        self.photo_dir = photo_dir
        self.use_webcam = use_webcam
        self.webcam_index = webcam_index
        self.save_full_res = save_full_res
        self.cap = None
        if self.use_webcam:
            self.cap = cv2.VideoCapture(self.webcam_index)
            if not self.cap.isOpened():
                print("[WARN] Webcam not available, photos will be blank frames.")
                self.cap = None

    def snap(self) -> str:
        ts = int(now_ts() * 1000)
        path = os.path.join(self.photo_dir, f"IMG_{ts}.jpg")
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                save_frame(frame, path)
                return path
        # Fallback: generate a blank with timestamp (still exercises dedupe)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, f"TS {ts}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        save_frame(img, path)
        return path

    def close(self):
        if self.cap is not None:
            self.cap.release()

# ---------- Main mission ----------
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["nbv", "orbit"], default="nbv")
    parser.add_argument("--shots", type=int, default=20)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    connection_url = cfg.get("connection_url", "udpin://:14540")
    takeoff_alt = float(cfg.get("takeoff_altitude_m", 5.0))
    max_minutes = float(cfg.get("max_flight_minutes", 2.0))

    photo_dir = cfg.get("photo_dir", "./data/shots")
    log_csv = cfg.get("log_csv", "./logs/shots.csv")
    ensure_dirs(photo_dir, log_csv)

    # Imaging
    pm = PhotoManager(photo_dir, cfg.get("use_webcam", True), int(cfg.get("webcam_index", 0)),
                      bool(cfg.get("save_full_res", True)))
    gimbal = Gimbal()
    rng = Rangefinder(simulate=True)

    # Dedupe params
    phash_size = int(cfg.get("phash_size", 16))
    phash_thresh = int(cfg.get("phash_threshold", 12))
    pos_radius_m = float(cfg.get("pos_radius_m", 1.0))
    angle_threshold_deg = float(cfg.get("angle_threshold_deg", 7.0))
    sharpness_replace_delta = float(cfg.get("sharpness_replace_delta", 10.0))

    # NBV params
    nbv_cfg = cfg.get("nbv", {})
    radii = nbv_cfg.get("radii_m", [3, 6, 10])
    heights = nbv_cfg.get("heights_m", [0.0, 1.5, 3.0])
    az_steps = int(nbv_cfg.get("azimuth_steps", 12))
    min_move_m = float(nbv_cfg.get("min_move_m", 2.0))

    # Connect to drone
    print(f"[INFO] Connecting to {connection_url} ...")
    drone = System()
    await drone.connect(system_address=connection_url)

    # Wait for connection/health
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[INFO] PX4 connected.")
            break

    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("[INFO] EKF ok, GPS ok, home ok.")
            break
        else:
            print("[INFO] Waiting for GPS/EKF...")
        await asyncio.sleep(1)

    # Get home position & start offboard-ish flow (we'll use simple goto in SITL)
    home_lat = None
    home_lon = None
    async for hp in drone.telemetry.home():
        home_lat = hp.latitude_deg
        home_lon = hp.longitude_deg
        print(f"[INFO] Home: lat={home_lat:.7f}, lon={home_lon:.7f}")
        break

    # Arm + takeoff
    print("[INFO] Arming...")
    await drone.action.arm()
    print(f"[INFO] Takeoff to {takeoff_alt} m AGL...")
    await drone.action.takeoff()
    await asyncio.sleep(5)

    # Climb to target altitude
    async for pos in drone.telemetry.position():
        current_alt = pos.relative_altitude_m
        if current_alt >= takeoff_alt - 0.5:
            break
        await asyncio.sleep(0.2)

    # Initialize DB
    db = ShotDB()

    # Define a simple local frame origin (XYZ) as (0,0,0) at first shot; we’ll accumulate NED deltas
    kept_positions = []

    # POI XYZ = origin for sampling (you can refine later)
    poi_xyz = (0.0, 0.0, takeoff_alt)

    start_t = time.time()
    shot_count = 0

    async def get_pose_xyz_ypr():
        # Use telemetry to approximate XYZ from NED velocities integrated (simplify: we will use relative estimates)
        # For now, we fake XYZ = (0,0,alt) since SITL example focuses on imaging loop and dedupe.
        # Extension: subscribe to odometry and integrate. Here we read attitude & relative altitude.
        att = await anext(drone.telemetry.attitude_quaternion())
        yaw, pitch, roll = euler_deg_from_quat(att)
        pos = await anext(drone.telemetry.position())
        xyz = (0.0, 0.0, pos.relative_altitude_m)
        return xyz, (yaw, pitch, roll)

    async def goto_relative_north_east_alt(dn: float, de: float, alt: float):
        pos = await anext(drone.telemetry.position())
        # Convert small N/E offsets to lat/lon deltas
        dlat = dn / 111320.0
        dlon = de / (111320.0 * math.cos(math.radians(pos.latitude_deg)))
        target_lat = pos.latitude_deg + dlat
        target_lon = pos.longitude_deg + dlon
        await drone.action.goto_location(target_lat, target_lon, pos.absolute_altitude_m - pos.relative_altitude_m + alt, 0.0)

    async def capture_and_process():
        nonlocal shot_count, kept_positions
        xyz, (yaw, pitch, roll) = await get_pose_xyz_ypr()
        img_path = pm.snap()

        is_dup, reason, phash, sharp = decide_duplicate(
            img_path,
            xyz,
            yaw,
            db,
            phash_thresh=phash_thresh,
            pos_radius_m=pos_radius_m,
            angle_threshold_deg=angle_threshold_deg,
            sharpness_replace_delta=sharpness_replace_delta,
            phash_size=phash_size,
        )
        kept = not is_dup or reason == "replace_better_sharpness"
        phash_hex = phash.__str__()  # hex
        shot = ShotMeta(
            timestamp=time.time(),
            img_path=img_path,
            x=xyz[0], y=xyz[1], z=xyz[2],
            yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
            phash_hex=phash_hex,
            sharpness=sharp,
            kept=kept,
            reason=reason
        )
        db.add(shot)
        append_csv(log_csv, shot)
        if kept:
            kept_positions.append([xyz[0], xyz[1], xyz[2]])
        shot_count += 1
        print(f"[SHOT] kept={kept} reason={reason} sharp={sharp:.1f} file={os.path.basename(img_path)}")

    if args.mode == "orbit":
        # Simple ring around current position (N/E plane), taking a photo each leg
        print("[MODE] ORBIT")
        R = 6.0
        legs = 12
        for i in range(legs):
            if (time.time() - start_t) > max_minutes * 60:
                break
            az = 2.0 * math.pi * i / legs
            dn = R * math.cos(az)
            de = R * math.sin(az)
            await goto_relative_north_east_alt(dn, de, takeoff_alt)
            await asyncio.sleep(3)
            await gimbal.set_angles(pitch_deg=-10.0)  # slight down tilt (placeholder)
            await capture_and_process()
            if shot_count >= args.shots:
                break

    else:
        print("[MODE] NBV")
        # Sample candidate viewpoints in a ring/sphere around POI in our toy XYZ
        candidates = sample_views(poi_xyz, radii=radii, heights=heights, azimuth_steps=az_steps)
        kept_np = None

        for _ in range(args.shots):
            if (time.time() - start_t) > max_minutes * 60:
                break
            if kept_positions:
                kept_np = np.array(kept_positions)
            current_xyz, _ypr = await get_pose_xyz_ypr()
            best = pick_next_best(candidates, kept_np, current_xyz, min_move_m=min_move_m)
            if best is None:
                print("[NBV] No valid candidate; breaking.")
                break
            # Move relative: map XYZ delta -> N/E + altitude
            target_xyz = best["pos"]
            dn = target_xyz[1] - current_xyz[1]  # y -> north
            de = target_xyz[0] - current_xyz[0]  # x -> east
            alt = target_xyz[2]                 # z -> altitude (up)
            await goto_relative_north_east_alt(dn, de, alt)
            await asyncio.sleep(3)
            await gimbal.set_angles(pitch_deg=best["pitch"], yaw_deg=best["yaw"])
            await capture_and_process()

    print("[INFO] Mission complete. Landing...")
    try:
        await drone.action.land()
    except Exception:
        pass
    await asyncio.sleep(5)
    pm.close()
    print(f"[INFO] Kept {sum(1 for s in db.shots if s.kept)}/{len(db.shots)} shots. Logs at {log_csv}")

if __name__ == "__main__":
    asyncio.run(main())
