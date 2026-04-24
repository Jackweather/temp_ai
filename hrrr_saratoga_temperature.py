from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pygrib
import requests


NOMADS_DIR_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod"
NOMADS_FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
BASE_DIR = Path("/var/data")
PLOTS_DIR_NAME = "plots"
ARCHIVES_DIR_NAME = "archives"
LATEST_RUN_FILENAME = "latest_run.json"
DEFAULT_LOCATION = "Saratoga Springs, NY"
DEFAULT_LAT = 43.0831
DEFAULT_LON = -73.7846
DEFAULT_BOX_PADDING = 0.05
DEFAULT_LOOKBACK_DAYS = 2
DEFAULT_TIMEOUT = 30
PLOT_RETENTION_COUNT = 24
ARCHIVE_RETENTION_DAYS = 30
LOCAL_TIMEZONE = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class HrrrRun:
    run_date: str
    cycle_hour: int
    forecast_hours: tuple[int, ...]

    @property
    def file_prefix(self) -> str:
        return f"hrrr.t{self.cycle_hour:02d}z.wrfsfcf"

    @property
    def run_time_utc(self) -> datetime:
        return datetime.strptime(
            f"{self.run_date}{self.cycle_hour:02d}", "%Y%m%d%H"
        ).replace(tzinfo=timezone.utc)

    @property
    def run_label(self) -> str:
        return f"{self.run_date}_{self.cycle_hour:02d}z"


@dataclass(frozen=True)
class RunArtifacts:
    run: HrrrRun
    location_name: str
    plot_path: Path
    archive_path: Path
    point_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the latest available HRRR run, extract the 2 m temperature near "
            "Saratoga Springs, NY, save a line chart under /var/data, and archive the run."
        )
    )
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="Latitude.")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON, help="Longitude.")
    parser.add_argument(
        "--location",
        default=DEFAULT_LOCATION,
        help="Display name for the plotted location.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="How many UTC days to search when discovering the latest available run.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=DEFAULT_BOX_PADDING,
        help="Bounding-box padding, in degrees, around the target point.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Base directory for plots, archives, and metadata.",
    )
    return parser.parse_args()


def ensure_storage_dirs(base_dir: Path) -> tuple[Path, Path]:
    plots_dir = base_dir / PLOTS_DIR_NAME
    archives_dir = base_dir / ARCHIVES_DIR_NAME
    plots_dir.mkdir(parents=True, exist_ok=True)
    archives_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, archives_dir


def build_run_stem(run: HrrrRun) -> str:
    return f"saratoga_springs_hrrr_{run.run_label}"


def latest_manifest_path(base_dir: Path) -> Path:
    return base_dir / LATEST_RUN_FILENAME


def discover_latest_run(session: requests.Session, lookback_days: int) -> HrrrRun:
    today_utc = datetime.now(timezone.utc).date()
    run_pattern = re.compile(r"hrrr\.t(\d{2})z\.wrfsfcf00\.grib2")

    for offset in range(lookback_days + 1):
        run_date = (today_utc - timedelta(days=offset)).strftime("%Y%m%d")
        directory_url = f"{NOMADS_DIR_BASE}/hrrr.{run_date}/conus/"
        response = session.get(directory_url, timeout=DEFAULT_TIMEOUT)
        if response.status_code != 200:
            continue

        cycle_hours = sorted({int(match) for match in run_pattern.findall(response.text)})
        for cycle_hour in reversed(cycle_hours):
            forecast_pattern = re.compile(
                rf"hrrr\.t{cycle_hour:02d}z\.wrfsfcf(\d{{2}})\.grib2"
            )
            forecast_hours = tuple(
                sorted({int(match) for match in forecast_pattern.findall(response.text)})
            )
            if forecast_hours:
                return HrrrRun(
                    run_date=run_date,
                    cycle_hour=cycle_hour,
                    forecast_hours=forecast_hours,
                )

    raise RuntimeError("Could not find an available HRRR run in the requested lookback window.")


def request_temperature_grib(
    session: requests.Session,
    run: HrrrRun,
    forecast_hour: int,
    lat: float,
    lon: float,
    padding: float,
) -> bytes:
    params = {
        "file": f"{run.file_prefix}{forecast_hour:02d}.grib2",
        "dir": f"/hrrr.{run.run_date}/conus",
        "var_TMP": "on",
        "lev_2_m_above_ground": "on",
        "subregion": "",
        "leftlon": lon - padding,
        "rightlon": lon + padding,
        "toplat": lat + padding,
        "bottomlat": lat - padding,
    }

    response = session.get(NOMADS_FILTER_URL, params=params, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    if not response.content.startswith(b"GRIB"):
        raise RuntimeError(
            f"NOMADS returned a non-GRIB response for forecast hour f{forecast_hour:02d}."
        )
    return response.content


def extract_temperature_at_point(grib_bytes: bytes, lat: float, lon: float) -> tuple[datetime, float]:
    with NamedTemporaryFile(suffix=".grib2", delete=False) as tmp_file:
        tmp_file.write(grib_bytes)
        tmp_path = Path(tmp_file.name)

    try:
        with pygrib.open(str(tmp_path)) as gribs:
            message = gribs.message(1)
            lats, lons = message.latlons()
            distances = np.hypot(lats - lat, lons - lon)
            nearest_index = np.unravel_index(np.argmin(distances), distances.shape)
            temperature_kelvin = float(message.values[nearest_index])
            valid_time = message.validDate.replace(tzinfo=timezone.utc)
    finally:
        tmp_path.unlink(missing_ok=True)

    temperature_fahrenheit = (temperature_kelvin - 273.15) * 9 / 5 + 32
    return valid_time, temperature_fahrenheit


def collect_temperature_series(
    session: requests.Session,
    run: HrrrRun,
    lat: float,
    lon: float,
    padding: float,
) -> list[tuple[datetime, float]]:
    series: list[tuple[datetime, float]] = []

    for forecast_hour in run.forecast_hours:
        print(f"Downloading f{forecast_hour:02d} from {run.run_date} {run.cycle_hour:02d}Z")
        grib_bytes = request_temperature_grib(session, run, forecast_hour, lat, lon, padding)
        valid_time, temperature_fahrenheit = extract_temperature_at_point(grib_bytes, lat, lon)
        series.append((valid_time, temperature_fahrenheit))

    return series


def archive_temperature_series(
    times_and_temps: Iterable[tuple[datetime, float]],
    run: HrrrRun,
    location_name: str,
    archive_path: Path,
) -> None:
    points = sorted(times_and_temps, key=lambda item: item[0])
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "location": location_name,
        "run_date": run.run_date,
        "cycle_hour": run.cycle_hour,
        "run_label": run.run_label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "series": [
            {
                "forecast_hour": forecast_hour,
                "valid_time_utc": valid_time.isoformat(),
                "valid_time_local": valid_time.astimezone(LOCAL_TIMEZONE).isoformat(),
                "temperature_f": round(temperature_fahrenheit, 2),
            }
            for forecast_hour, (valid_time, temperature_fahrenheit) in enumerate(points)
        ],
    }
    archive_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_temperature_series(
    times_and_temps: Iterable[tuple[datetime, float]],
    run: HrrrRun,
    location_name: str,
    output_path: Path,
) -> None:
    points = sorted(times_and_temps, key=lambda item: item[0])
    valid_times_local = [valid_time.astimezone(LOCAL_TIMEZONE) for valid_time, _ in points]
    temperatures = [temperature for _, temperature in points]
    point_count = len(valid_times_local)

    if point_count <= 12:
        tick_interval = 1
    elif point_count <= 24:
        tick_interval = 2
    else:
        tick_interval = 3

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(valid_times_local, temperatures, marker="o", linewidth=2.2, color="#0b7285")
    ax.set_title(
        f"HRRR 2 m Temperature Forecast\n{location_name} | Run {run.run_date} {run.cycle_hour:02d}Z"
    )
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=tick_interval, tz=LOCAL_TIMEZONE))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %I:%M %p %Z", tz=LOCAL_TIMEZONE))
    ax.set_xlabel("Forecast valid time (Eastern Time)")
    ax.set_ylabel("Temperature (°F)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def prune_plot_images(plots_dir: Path, keep_count: int = PLOT_RETENTION_COUNT) -> None:
    plot_files = sorted(plots_dir.glob("*.png"), key=lambda path: path.stat().st_mtime, reverse=True)
    for old_plot in plot_files[keep_count:]:
        old_plot.unlink(missing_ok=True)


def parse_run_label_from_stem(stem: str) -> datetime | None:
    match = re.search(r"_(\d{8})_(\d{2})z$", stem)
    if not match:
        return None
    run_date, cycle_hour = match.groups()
    return datetime.strptime(f"{run_date}{cycle_hour}", "%Y%m%d%H").replace(
        tzinfo=timezone.utc
    )


def prune_archives(archives_dir: Path, retention_days: int = ARCHIVE_RETENTION_DAYS) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    for archive_path in archives_dir.glob("*.json"):
        run_time = parse_run_label_from_stem(archive_path.stem)
        if run_time is not None and run_time < cutoff:
            archive_path.unlink(missing_ok=True)


def write_latest_manifest(
    base_dir: Path,
    artifacts: RunArtifacts,
    times_and_temps: Iterable[tuple[datetime, float]],
) -> None:
    points = sorted(times_and_temps, key=lambda item: item[0])
    payload = {
        "location": artifacts.location_name,
        "run_date": artifacts.run.run_date,
        "cycle_hour": artifacts.run.cycle_hour,
        "run_label": artifacts.run.run_label,
        "plot_file": artifacts.plot_path.name,
        "archive_file": artifacts.archive_path.name,
        "plot_path": str(artifacts.plot_path),
        "archive_path": str(artifacts.archive_path),
        "point_count": artifacts.point_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "series": [
            {
                "valid_time_utc": valid_time.isoformat(),
                "valid_time_local": valid_time.astimezone(LOCAL_TIMEZONE).isoformat(),
                "temperature_f": round(temperature_fahrenheit, 2),
            }
            for valid_time, temperature_fahrenheit in points
        ],
    }
    latest_manifest_path(base_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_latest_manifest(base_dir: Path) -> dict | None:
    manifest_path = latest_manifest_path(base_dir)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def generate_latest_run_artifacts(
    base_dir: Path = BASE_DIR,
    location_name: str = DEFAULT_LOCATION,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    padding: float = DEFAULT_BOX_PADDING,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> RunArtifacts:
    plots_dir, archives_dir = ensure_storage_dirs(base_dir)

    with requests.Session() as session:
        run = discover_latest_run(session, lookback_days)
        print(f"Latest available run: {run.run_date} {run.cycle_hour:02d}Z")
        print(f"Forecast hours found: {len(run.forecast_hours)}")

        series = collect_temperature_series(
            session=session,
            run=run,
            lat=lat,
            lon=lon,
            padding=padding,
        )

    run_stem = build_run_stem(run)
    plot_path = plots_dir / f"{run_stem}.png"
    archive_path = archives_dir / f"{run_stem}.json"

    plot_temperature_series(series, run, location_name, plot_path)
    archive_temperature_series(series, run, location_name, archive_path)
    prune_plot_images(plots_dir)
    prune_archives(archives_dir)

    artifacts = RunArtifacts(
        run=run,
        location_name=location_name,
        plot_path=plot_path,
        archive_path=archive_path,
        point_count=len(series),
    )
    write_latest_manifest(base_dir, artifacts, series)
    return artifacts


def main() -> None:
    args = parse_args()
    artifacts = generate_latest_run_artifacts(
        base_dir=args.base_dir,
        location_name=args.location,
        lat=args.lat,
        lon=args.lon,
        padding=args.padding,
        lookback_days=args.lookback_days,
    )
    print(f"Saved chart to: {artifacts.plot_path.resolve()}")
    print(f"Saved archive to: {artifacts.archive_path.resolve()}")


if __name__ == "__main__":
    main()