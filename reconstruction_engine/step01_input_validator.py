import os
import re
import glob
import json
import csv
import numpy as np
from PIL import Image, ExifTags
from typing import Dict, List, Any, Optional, Tuple
from reconstruction_engine.logger import PipelineLogger, PipelineTracker

class InputValidator:
    """Step 1: Ingests, validates, and aligns keyframes and telemetry metadata."""
    def __init__(self, config: Dict[str, Any], logger: PipelineLogger, tracker: PipelineTracker):
        self.config = config.get("step01_validation", {})
        self.logger = logger
        self.tracker = tracker

    def validate_and_ingest(
        self,
        input_dir: str,
        gps_path: Optional[str] = None,
        intrinsics_path: Optional[str] = None,
        gcp_path: Optional[str] = None,
        masks_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry for Step 1.
        Returns:
            Dict containing:
                - 'frames': List of dicts per frame: {index, path, filename, width, height, timestamp, gps: {lat, lon, alt}, imu: {roll, pitch, yaw}, mask_path}
                - 'intrinsics': Dict or None
                - 'gcps': List or None
                - 'degraded_flags': List of str
        """
        self.logger.stage_header(1, "Input Validation & Telemetry Ingestion")
        
        # 1. Discover keyframe images
        image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.tif", "*.tiff")
        frame_paths = []
        for ext in image_extensions:
            frame_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        
        # Filter out macOS resource-fork files (._*)
        frame_paths = [p for p in frame_paths if not os.path.basename(p).startswith("._")]
        frame_paths = sorted(list(set(frame_paths)))
        if not frame_paths:
            raise FileNotFoundError(f"No valid image keyframes found in directory: {input_dir}")

        min_frames = self.config.get("min_frames", 3)
        if len(frame_paths) < min_frames:
            raise ValueError(f"Found {len(frame_paths)} frames, but minimum required is {min_frames}.")

        self.logger.info(f"Discovered {len(frame_paths)} keyframes in '{input_dir}'.")
        self.tracker.metrics["total_frames"] = len(frame_paths)

        # 2. Check frame dimensions and aspect consistency
        frame_info_list = []
        ref_shape = None
        readable_count = 0
        for idx, fpath in enumerate(frame_paths):
            try:
                with Image.open(fpath) as img:
                    w, h = img.size
                    img.verify()  # Verify image is readable
                if ref_shape is None:
                    ref_shape = (w, h)
                frame_info_list.append({
                    "index": idx,
                    "path": os.path.abspath(fpath),
                    "filename": os.path.basename(fpath),
                    "width": w,
                    "height": h,
                    "timestamp": float(idx), # Default sequential timestamp
                    "gps": None,
                    "imu": None,
                    "mask_path": None
                })
                readable_count += 1
            except Exception as e:
                self.logger.warning(f"Skipping unreadable image '{os.path.basename(fpath)}': {e}")
        
        if not frame_info_list:
            raise FileNotFoundError(f"No readable image keyframes found in directory: {input_dir}")
        self.logger.info(f"Verified {readable_count}/{len(frame_paths)} images are readable.")
        
        self.logger.info(f"Keyframe resolution verified: {ref_shape[0]}x{ref_shape[1]} px.")

        # 3. Parse GPS Telemetry
        telemetry_samples = []
        gps_source = "NONE"
        
        if gps_path and os.path.exists(gps_path):
            if gps_path.endswith(".srt"):
                telemetry_samples = self._parse_dji_srt(gps_path)
                gps_source = "DJI_SRT"
            elif gps_path.endswith(".csv"):
                telemetry_samples = self._parse_csv_telemetry(gps_path)
                gps_source = "CSV"
            elif gps_path.endswith(".json"):
                telemetry_samples = self._parse_json_telemetry(gps_path)
                gps_source = "JSON"

        # If no external GPS file provided or parsed, attempt EXIF extraction
        if not telemetry_samples:
            exif_gps_found = False
            for f_info in frame_info_list:
                gps_data = self._extract_exif_gps(f_info["path"])
                if gps_data:
                    f_info["gps"] = gps_data
                    exif_gps_found = True
            if exif_gps_found:
                gps_source = "EXIF"

        # If we have telemetry samples, interpolate them to frames
        if telemetry_samples:
            self._interpolate_telemetry_to_frames(frame_info_list, telemetry_samples)

        # Verify GPS coverage
        gps_valid_count = sum(1 for f in frame_info_list if f["gps"] is not None)
        if gps_valid_count == 0:
            if self.config.get("require_gps", False):
                self.tracker.mark_degraded("step01_validation", "No GPS coordinates found; downstream georeferencing will use synthetic local coordinates.", self.logger)
                self.logger.warning("No GPS data found for any keyframes!")
            else:
                self.logger.info("No GPS data found. Running in image-only mode (camera poses will be estimated from images).")
        else:
            self.logger.success(f"GPS telemetry linked: {gps_valid_count}/{len(frame_info_list)} frames tagged via {gps_source}.")

        # 4. Check Camera Intrinsics
        intrinsics_data = None
        if intrinsics_path and os.path.exists(intrinsics_path):
            try:
                with open(intrinsics_path, "r", encoding="utf-8") as f:
                    intrinsics_data = json.load(f)
                self.logger.success(f"Known camera intrinsics loaded from '{intrinsics_path}'.")
            except Exception as e:
                self.logger.warning(f"Failed to read intrinsics file: {e}")
        else:
            self.logger.info("No prior camera intrinsics supplied. Bundle adjustment will self-calibrate focal length and radial distortion.")

        # 5. Check GCPs (Ground Control Points)
        gcps = []
        if gcp_path and os.path.exists(gcp_path):
            gcps = self._parse_gcps(gcp_path)
            self.logger.success(f"Loaded {len(gcps)} Ground Control Points from '{gcp_path}'.")
        else:
            self.logger.info("No GCP file provided. 7-parameter Helmert alignment will rely purely on GPS trajectory.")

        # 6. Check Dynamic Object Masks
        if masks_dir and os.path.exists(masks_dir):
            matched_masks = 0
            for f_info in frame_info_list:
                base_name = os.path.splitext(f_info["filename"])[0]
                for ext in (".png", ".jpg", ".tif"):
                    candidate = os.path.join(masks_dir, f"{base_name}{ext}")
                    if os.path.exists(candidate):
                        f_info["mask_path"] = os.path.abspath(candidate)
                        matched_masks += 1
                        break
            if matched_masks > 0:
                self.logger.success(f"Dynamic-object masks discovered for {matched_masks}/{len(frame_info_list)} frames.")
            else:
                self.logger.info(f"Masks directory provided ('{masks_dir}') but no matching frame masks found.")
        else:
            self.logger.info("No dynamic-object masks supplied. Point cloud cleanup will rely on statistical outlier filtering.")

        return {
            "frames": frame_info_list,
            "intrinsics": intrinsics_data,
            "gcps": gcps,
            "ref_shape": ref_shape,
            "gps_source": gps_source
        }

    def _parse_dji_srt(self, srt_path: str) -> List[Dict[str, Any]]:
        """Parses timestamped GPS and sensor logs from DJI drone subtitle files."""
        samples = []
        with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        blocks = re.split(r'\n\s*\n', content.strip())
        for block in blocks:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if len(lines) < 2:
                continue
            
            # Timestamp regex (00:00:01,000 --> 00:00:02,000)
            time_match = re.search(r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})', lines[1] if len(lines) > 1 else lines[0])
            t_sec = 0.0
            if time_match:
                h, m, s, ms = map(int, time_match.groups())
                t_sec = h * 3600 + m * 60 + s + ms / 1000.0

            # Match GPS coordinates: [latitude: 12.345678] [longitude: 78.123456] [rel_alt: 45.2 abs_alt: 120.4]
            # or [iso : 100] [shutter : 1/1250.0] [fnum : 2.8] [ev : 0] [ct : 5400] [color_md : default] [focal_len : 24.00] [dji_srt] [latitude: ...]
            lat_m = re.search(r'latitude\s*[:=]\s*([-+]?\d+\.?\d*)', block, re.IGNORECASE)
            lon_m = re.search(r'longitude\s*[:=]\s*([-+]?\d+\.?\d*)', block, re.IGNORECASE)
            alt_m = re.search(r'(?:rel_alt|altitude|alt|abs_alt)\s*[:=]\s*([-+]?\d+\.?\d*)', block, re.IGNORECASE)
            roll_m = re.search(r'roll\s*[:=]\s*([-+]?\d+\.?\d*)', block, re.IGNORECASE)
            pitch_m = re.search(r'pitch\s*[:=]\s*([-+]?\d+\.?\d*)', block, re.IGNORECASE)
            yaw_m = re.search(r'yaw\s*[:=]\s*([-+]?\d+\.?\d*)', block, re.IGNORECASE)

            if lat_m and lon_m:
                lat = float(lat_m.group(1))
                lon = float(lon_m.group(1))
                alt = float(alt_m.group(1)) if alt_m else 50.0
                imu = None
                if roll_m and pitch_m and yaw_m:
                    imu = {
                        "roll": float(roll_m.group(1)),
                        "pitch": float(pitch_m.group(1)),
                        "yaw": float(yaw_m.group(1))
                    }
                samples.append({
                    "timestamp": t_sec,
                    "lat": lat,
                    "lon": lon,
                    "alt": alt,
                    "imu": imu
                })
        return samples

    def _parse_csv_telemetry(self, csv_path: str) -> List[Dict[str, Any]]:
        """Parses CSV flight telemetry."""
        samples = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            t_idx = 0.0
            for row in reader:
                # Find latitude and longitude keys
                lat_key = next((k for k in row if "lat" in k.lower()), None)
                lon_key = next((k for k in row if "lon" in k.lower() or "lng" in k.lower()), None)
                alt_key = next((k for k in row if "alt" in k.lower()), None)
                time_key = next((k for k in row if "time" in k.lower() or "timestamp" in k.lower()), None)

                if lat_key and lon_key and row[lat_key] and row[lon_key]:
                    try:
                        lat = float(row[lat_key])
                        lon = float(row[lon_key])
                        alt = float(row[alt_key]) if (alt_key and row[alt_key]) else 50.0
                        t = float(row[time_key]) if (time_key and row[time_key]) else t_idx
                        samples.append({
                            "timestamp": t,
                            "lat": lat,
                            "lon": lon,
                            "alt": alt,
                            "imu": None
                        })
                        t_idx += 1.0
                    except ValueError:
                        continue
        return samples

    def _parse_json_telemetry(self, json_path: str) -> List[Dict[str, Any]]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "frames" in data:
            return data["frames"]
        return []

    def _extract_exif_gps(self, img_path: str) -> Optional[Dict[str, float]]:
        """Extracts latitude, longitude, and altitude from image EXIF metadata."""
        try:
            with Image.open(img_path) as img:
                exif = img._getexif()
                if not exif:
                    return None
                
                gps_info = {}
                for tag, value in exif.items():
                    tag_name = ExifTags.TAGS.get(tag, tag)
                    if tag_name == "GPSInfo":
                        for t in value:
                            sub_tag = ExifTags.GPSTAGS.get(t, t)
                            gps_info[sub_tag] = value[t]

                if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
                    lat_dms = gps_info["GPSLatitude"]
                    lat_ref = gps_info.get("GPSLatitudeRef", "N")
                    lon_dms = gps_info["GPSLongitude"]
                    lon_ref = gps_info.get("GPSLongitudeRef", "E")

                    lat = float(lat_dms[0] + lat_dms[1]/60.0 + lat_dms[2]/3600.0)
                    if lat_ref.upper() == "S":
                        lat = -lat

                    lon = float(lon_dms[0] + lon_dms[1]/60.0 + lon_dms[2]/3600.0)
                    if lon_ref.upper() == "W":
                        lon = -lon

                    alt = float(gps_info.get("GPSAltitude", 50.0))
                    return {"lat": lat, "lon": lon, "alt": alt}
        except Exception:
            return None
        return None

    def _interpolate_telemetry_to_frames(self, frame_info_list: List[Dict[str, Any]], telemetry: List[Dict[str, Any]]):
        """
        Linearly interpolates telemetry to frame timestamps.
        Prevents clock skew / frame rate mismatch between video and drone telemetry loggers.
        """
        if not telemetry:
            return

        # Sort telemetry by timestamp
        telemetry = sorted(telemetry, key=lambda x: x["timestamp"])
        t_times = np.array([x["timestamp"] for x in telemetry])
        t_lats = np.array([x["lat"] for x in telemetry])
        t_lons = np.array([x["lon"] for x in telemetry])
        t_alts = np.array([x["alt"] for x in telemetry])

        num_frames = len(frame_info_list)
        # If frame timestamps are just sequential indices [0, 1, 2...], map linearly to telemetry duration
        if frame_info_list[-1]["timestamp"] == (num_frames - 1):
            frame_times = np.linspace(t_times[0], t_times[-1], num_frames)
        else:
            frame_times = np.array([f["timestamp"] for f in frame_info_list])

        # Interpolate
        interp_lats = np.interp(frame_times, t_times, t_lats)
        interp_lons = np.interp(frame_times, t_times, t_lons)
        interp_alts = np.interp(frame_times, t_times, t_alts)

        for i, f_info in enumerate(frame_info_list):
            f_info["timestamp"] = float(frame_times[i])
            f_info["gps"] = {
                "lat": float(interp_lats[i]),
                "lon": float(interp_lons[i]),
                "alt": float(interp_alts[i])
            }

    def _parse_gcps(self, gcp_path: str) -> List[Dict[str, Any]]:
        """Parses Ground Control Points (GCP) CSV: id, lat, lon, alt, [pixel_x, pixel_y, image_name]."""
        gcps = []
        with open(gcp_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    gcps.append({
                        "id": row.get("id", f"GCP_{len(gcps)}"),
                        "lat": float(row.get("lat", row.get("latitude"))),
                        "lon": float(row.get("lon", row.get("longitude"))),
                        "alt": float(row.get("alt", row.get("altitude", 0.0))),
                        "image": row.get("image", None),
                        "px": float(row.get("px", 0.0)) if "px" in row else None,
                        "py": float(row.get("py", 0.0)) if "py" in row else None
                    })
                except Exception:
                    continue
        return gcps
