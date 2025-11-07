# scripts/fly_and_shoot.py
"""
Minimal non-agent demo: connect → takeoff → capture N frames → land.

Usage:
  python scripts/fly_and_shoot.py --shots 5 --config configs/config.yaml
"""
import os
import sys
import yaml
import argparse
import asyncio

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from ai_photographer.io_utils import (
    Webcam, placeholder_image, save_image_bgr, timestamp_name
)

try:
    from mavsdk import System
except Exception:
    System = None


async def connect_and_takeoff(connection_url: str, alt_m: float):
    if System is None:
        print(f"[DRY] MAVSDK not installed; simulating takeoff to {alt_m} m.")
        return None
    drone = System()
    print(f"Connecting to PX4 at {connection_url} ...")
    await drone.connect(system_address=connection_url)

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("PX4 connected.")
            break

    await drone.action.arm()
    await asyncio.sleep(0.3)
    await drone.action.takeoff()
    await asyncio.sleep(4.0)
    return drone


async def land_and_close(drone):
    if drone is None:
        print("[DRY] Land (simulated).")
        return
    try:
        await drone.action.land()
        await asyncio.sleep(2.0)
    except Exception as e:
        print(f"Landing error: {e}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=5, help="number of photos to capture")
    ap.add_argument("--config", type=str, default=os.path.join(ROOT, "configs", "config.yaml"))
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["photo_dir"], exist_ok=True)

    # PX4 connect + takeoff
    drone = await connect_and_takeoff(cfg["connection_url"], cfg["takeoff_altitude_m"])

    # Camera (webcam or placeholder)
    cam = None
    if bool(cfg.get("use_webcam", True)):
        try:
            cam = Webcam(index=int(cfg.get("webcam_index", 0)))
            print("Webcam opened.")
        except Exception as e:
            print(f"Webcam unavailable ({e}); using placeholder frames.")
            cam = None

    try:
        for i in range(args.shots):
            frame = cam.read() if cam else placeholder_image()
            out_path = os.path.join(cfg["photo_dir"], timestamp_name())
            save_image_bgr(out_path, frame)
            print(f"[{i+1}/{args.shots}] saved {out_path}")
            await asyncio.sleep(1.0)
    finally:
        if cam:
            cam.release()
        await land_and_close(drone)
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
