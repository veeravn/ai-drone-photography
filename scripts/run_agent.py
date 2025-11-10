# scripts/run_agent.py
import argparse
import os
import sys
import time
import yaml
import asyncio
import logging
from typing import Optional, Tuple

# repo root on path for ai_photographer/*
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from ai_photographer.io_utils import (
    Webcam, placeholder_image, save_image_bgr, timestamp_name,
)
from ai_photographer.nbv_planner import sample_views, pick_next_best
from ai_photographer.agent import NBVAgent, AgentParams
from ai_photographer.gimbal_control import Gimbal
from ai_photographer.rangefinder_reader import Rangefinder
from ai_photographer.dedupe import ShotDB, ShotMeta, decide_duplicate

try:
    from mavsdk import System
except Exception:
    System = None


def setup_logging(log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(os.path.dirname(log_path), "session.log")),
        ],
    )


async def connect_px4(connection_url: str) -> Optional["System"]:
    if System is None:
        logging.warning("MAVSDK not available, running in camera-only mode.")
        return None
    drone = System()
    logging.info(f"Connecting to PX4 at {connection_url} ...")
    await drone.connect(system_address=connection_url)
    async for state in drone.core.connection_state():
        if state.is_connected:
            logging.info("PX4 connected.")
            break
    return drone


async def safe_takeoff(drone: Optional["System"], alt_m: float) -> None:
    if drone is None:
        logging.info(f"[DRY] Takeoff to {alt_m} m (simulated).")
        return
    await drone.action.arm()
    await asyncio.sleep(0.5)
    await drone.action.takeoff()
    await asyncio.sleep(5.0)


async def safe_land(drone: Optional["System"]) -> None:
    try:
        if drone is None:
            logging.info("[DRY] Land (simulated).")
            return
        await drone.action.land()
        await asyncio.sleep(3.0)
    except Exception as e:
        logging.error(f"Landing error: {e}")


def current_pose_stub() -> Tuple[float, float, float]:
    # Until telemetry is wired, keep pose at origin (NBV yaw provides heading context)
    return (0.0, 0.0, 0.0)


async def run(config_path: str = os.path.join(ROOT, "config.yaml")) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    setup_logging(cfg["log_csv"])

    os.makedirs(cfg["photo_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["log_csv"]), exist_ok=True)

    # PX4 connect + takeoff
    drone = await connect_px4(cfg["connection_url"])
    await safe_takeoff(drone, cfg["takeoff_altitude_m"])

    # Camera
    cam = None
    if bool(cfg.get("use_webcam", True)):
        try:
            cam = Webcam(index=int(cfg.get("webcam_index", 0)))
            logging.info("Webcam opened.")
        except Exception as e:
            logging.warning(f"Webcam unavailable ({e}); using placeholder frames.")
            cam = None

    # Gimbal + Rangefinder (simulation-friendly)
    gimbal = Gimbal()
    rng = Rangefinder(simulate=True)  # set simulate=False on real hardware if sensor is wired

    # Agent & Shot DB
    agent = NBVAgent(AgentParams(**cfg["agent"])) if "agent" in cfg else None
    db = ShotDB()

    # Timing
    t_start = time.time()
    max_minutes = float(cfg["max_flight_minutes"])

    try:
        while (time.time() - t_start) / 60.0 < max_minutes:
            # --- 1) Plan NBV candidates (and pick a nominal yaw) ---
            cands = sample_views(cfg["nbv"]) if "nbv" in cfg else []
            best = pick_next_best(cands) if cands else None
            best_yaw = 0.0 if best is None else float(best.get("yaw", 0.0))

            # --- 2) Capture frame ---
            if cam:
                frame_bgr = cam.read()
            else:
                frame_bgr = placeholder_image()

            # Save image first (your dedupe works on image path)
            name = timestamp_name(prefix="shot", ext=".jpg")
            out_path = os.path.join(cfg["photo_dir"], name)
            save_image_bgr(out_path, frame_bgr)

            # --- 3) Pose & rangefinder (stubs for now) ---
            x, y, z = current_pose_stub()
            try:
                rng_dist = await rng.read_distance_m()
            except TypeError:
                rng_dist = rng.read_meters()

            # --- 4) Dedupe decision using your advanced logic ---
            is_dup, reason, phash, sharp = decide_duplicate(
                candidate_img=out_path,
                candidate_pose=(x, y, z),
                candidate_yaw_deg=best_yaw,
                db=db,
                phash_thresh=int(cfg.get("phash_threshold", 12)),
                pos_radius_m=float(cfg.get("pos_radius_m", 1.0)),
                angle_threshold_deg=float(cfg.get("angle_threshold_deg", 7.0)),
                sharpness_replace_delta=float(cfg.get("sharpness_replace_delta", 10.0)),
                phash_size=int(cfg.get("phash_size", 16)),
            )

            kept = not is_dup  # if duplicate, we skip keeping by default

            # --- 5) Log to CSV via ShotMeta + add to in-memory DB ---
            shot = ShotMeta(
                timestamp=time.time(),
                img_path=out_path,
                x=x, y=y, z=z,
                yaw_deg=best_yaw,
                pitch_deg=gimbal.last_pitch_deg or 0.0,
                roll_deg=0.0,
                phash_hex=str(phash),
                sharpness=float(sharp),
                kept=bool(kept),
                reason=reason,
            )
            # Append to CSV (kept + not kept) so we have full audit
            # (If you prefer to log only kept, gate this call.)
            from ai_photographer.dedupe import append_shotmeta_csv as append_csv_shot
            append_csv_shot(cfg["log_csv"], shot)

            # Only index kept shots for spatial dedupe
            if kept:
                db.add(shot)

            logging.info(
                f"kept={kept} reason={reason} sharp={sharp:.1f} "
                f"rng={rng_dist} yaw={best_yaw:.1f} gimbal=({gimbal.last_pitch_deg},{gimbal.last_yaw_deg})"
            )

            # --- 6) Agentic response (pitch retake; movement remains dry-run) ---
            if agent:
                agent.update_after_shot(kept=kept, sharpness=float(sharp), xyz=(x, y, z))
                action = agent.decide()
                _, new_alt, pitch_off = agent.apply_action_to_candidates(
                    action, cands or [], cfg["takeoff_altitude_m"]
                )

                if pitch_off is not None:
                    await gimbal.set_angles(pitch_off, best_yaw)

                logging.info(f"agent_action={action} pitch_offset={pitch_off}")

            await asyncio.sleep(1.5)

    finally:
        if cam:
            cam.release()
        await safe_land(drone)
        logging.info("Session complete.")


if __name__ == "__main__":
    # default to the repo root config.yaml (scripts/ is one level below repo root)
    import asyncio
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Run NBV Agent")
    parser.add_argument("--config", "-c", default=str(ROOT / "config.yaml"), help="path to config yaml")
    args = parser.parse_args()
    asyncio.run(run(args.config))
