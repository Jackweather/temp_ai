from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys
import threading

from flask import Flask, abort, render_template, send_from_directory, url_for
import requests

from hrrr_saratoga_temperature import (
    ARCHIVES_DIR_NAME,
    BASE_DIR,
    DEFAULT_LAT,
    DEFAULT_LOCATION,
    DEFAULT_LON,
    LOCAL_TIMEZONE,
    PLOTS_DIR_NAME,
    ensure_storage_dirs,
    generate_latest_run_artifacts,
    load_latest_manifest,
)


app = Flask(__name__)
APP_DIR = Path(__file__).resolve().parent
LOGS_DIR = APP_DIR / "logs"
NWS_DIR_NAME = "nws"
NWS_LATEST_FILENAME = "latest_nws_hourly.json"
NWS_USER_AGENT = "hrrr-saratoga-temps/1.0"
NWS_RETENTION_DAYS = 30
NWS_ARCHIVE_PREFIX = "saratoga_springs_nws"
TRAINING_DIR_NAME = "training"
TRAINING_DATA_FILENAME = "hrrr_nws_training_pairs.json"


@app.template_filter("format_eastern_time")
def format_eastern_time(value: str | None) -> str:
    if not value:
        return ""
    dt = datetime.fromisoformat(value)
    return dt.astimezone(LOCAL_TIMEZONE).strftime("%m-%d-%Y %I:%M %p %Z")


@app.template_filter("pretty_json")
def pretty_json(value: dict | list | None) -> str:
    if value is None:
        return ""
    return json_dumps(value)


def run_scripts(
    scripts: list[tuple[str, str]],
    task_id: int,
    parallel: bool = True,
    max_parallel: int = 1,
) -> None:
    del parallel
    del max_parallel
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    for index, (script_path, working_dir) in enumerate(scripts, start=1):
        script_name = Path(script_path).stem
        log_path = LOGS_DIR / f"task{task_id}_{index}_{script_name}.log"
        with log_path.open("a", encoding="utf-8") as log_handle:
            subprocess.run(
                [sys.executable, script_path],
                cwd=working_dir,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )


def list_recent_archives(base_dir: Path) -> list[dict[str, str]]:
    _, archives_dir = ensure_storage_dirs(base_dir)
    archive_files = sorted(
        archives_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True
    )
    return [
        {
            "name": archive_path.name,
            "url": url_for("serve_archive", filename=archive_path.name),
        }
        for archive_path in archive_files
    ]


def list_nws_archives(base_dir: Path) -> list[dict[str, str]]:
    nws_dir = ensure_nws_dir(base_dir)
    archive_files = sorted(
        nws_dir.glob(f"{NWS_ARCHIVE_PREFIX}_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return [
        {
            "name": archive_path.name,
            "url": url_for("serve_nws_archive", filename=archive_path.name),
        }
        for archive_path in archive_files
    ]


def ensure_training_dir(base_dir: Path) -> Path:
    training_dir = base_dir / TRAINING_DIR_NAME
    training_dir.mkdir(parents=True, exist_ok=True)
    return training_dir


def training_pairs_path(base_dir: Path = BASE_DIR) -> Path:
    return ensure_training_dir(base_dir) / TRAINING_DATA_FILENAME


def ensure_nws_dir(base_dir: Path) -> Path:
    nws_dir = base_dir / NWS_DIR_NAME
    nws_dir.mkdir(parents=True, exist_ok=True)
    return nws_dir


def latest_nws_path(base_dir: Path) -> Path:
    return ensure_nws_dir(base_dir) / NWS_LATEST_FILENAME


def build_nws_hour_label(valid_time_utc: datetime) -> str:
    return valid_time_utc.astimezone(timezone.utc).strftime("%Y%m%d_%Hz").lower()


def build_nws_archive_path(base_dir: Path, valid_time_utc: datetime) -> Path:
    hour_label = build_nws_hour_label(valid_time_utc)
    return ensure_nws_dir(base_dir) / f"{NWS_ARCHIVE_PREFIX}_{hour_label}.json"


def prune_nws_archives(base_dir: Path, retention_days: int = NWS_RETENTION_DAYS) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    for archive_path in ensure_nws_dir(base_dir).glob(f"{NWS_ARCHIVE_PREFIX}_*.json"):
        match = archive_path.stem.removeprefix(f"{NWS_ARCHIVE_PREFIX}_")
        try:
            archive_hour = datetime.strptime(match, "%Y%m%d_%Hz").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if archive_hour < cutoff:
            archive_path.unlink(missing_ok=True)


def select_current_hourly_period(periods: list[dict]) -> tuple[dict, int]:
    now_local = datetime.now(LOCAL_TIMEZONE)
    current_hour_start = now_local.replace(minute=0, second=0, microsecond=0)

    for forecast_hour, period in enumerate(periods):
        period_start = datetime.fromisoformat(period["startTime"]).astimezone(LOCAL_TIMEZONE)
        period_end_text = period.get("endTime")
        if period_end_text:
            period_end = datetime.fromisoformat(period_end_text).astimezone(LOCAL_TIMEZONE)
            if period_start <= current_hour_start < period_end:
                return period, forecast_hour
        elif period_start == current_hour_start:
            return period, forecast_hour

    future_periods = []
    for forecast_hour, period in enumerate(periods):
        period_start = datetime.fromisoformat(period["startTime"]).astimezone(LOCAL_TIMEZONE)
        if period_start >= current_hour_start:
            future_periods.append((period_start, forecast_hour, period))

    if future_periods:
        _, forecast_hour, period = min(future_periods, key=lambda item: item[0])
        return period, forecast_hour

    return periods[-1], len(periods) - 1


def fetch_nws_hourly_forecast(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    location_name: str = DEFAULT_LOCATION,
    base_dir: Path = BASE_DIR,
) -> dict:
    generated_at = datetime.now(timezone.utc)
    headers = {
        "Accept": "application/geo+json",
        "User-Agent": NWS_USER_AGENT,
    }
    session = requests.Session()
    session.headers.update(headers)

    points_response = session.get(
        f"https://api.weather.gov/points/{lat},{lon}",
        timeout=30,
    )
    points_response.raise_for_status()
    points_payload = points_response.json()

    hourly_url = points_payload["properties"]["forecastHourly"]
    hourly_response = session.get(hourly_url, timeout=30)
    hourly_response.raise_for_status()
    hourly_payload = hourly_response.json()

    periods = hourly_payload["properties"].get("periods", [])
    issued_at = hourly_payload["properties"].get("updateTime")
    if not periods:
        raise RuntimeError("NWS hourly forecast response did not contain any periods.")

    selected_period, selected_forecast_hour = select_current_hourly_period(periods)

    valid_time_local = datetime.fromisoformat(selected_period["startTime"]).astimezone(LOCAL_TIMEZONE)
    valid_time_utc = valid_time_local.astimezone(timezone.utc)
    temperature_value = selected_period.get("temperature")
    temperature_unit = selected_period.get("temperatureUnit", "F")
    temperature_f = temperature_value
    if temperature_value is not None and temperature_unit == "C":
        temperature_f = round((temperature_value * 9 / 5) + 32, 2)

    current_hour = {
        "forecast_hour": selected_forecast_hour,
        "valid_time_utc": valid_time_utc.isoformat(),
        "valid_time_local": valid_time_local.isoformat(),
        "temperature_f": temperature_f,
        "temperature_unit": "F",
        "short_forecast": selected_period.get("shortForecast"),
        "is_daytime": selected_period.get("isDaytime"),
    }

    payload = {
        "location": location_name,
        "source": "NWS",
        "product": "forecastHourly",
        "generated_at": generated_at.isoformat(),
        "issued_at": issued_at,
        "grid_id": points_payload["properties"].get("gridId"),
        "grid_x": points_payload["properties"].get("gridX"),
        "grid_y": points_payload["properties"].get("gridY"),
        "verification_hour_label": build_nws_hour_label(valid_time_utc),
        "current_hour": current_hour,
    }
    payload_text = json_dumps(payload)
    latest_nws_path(base_dir).write_text(payload_text, encoding="utf-8")
    build_nws_archive_path(base_dir, valid_time_utc).write_text(payload_text, encoding="utf-8")
    prune_nws_archives(base_dir)
    return payload


def load_or_fetch_nws_hourly(base_dir: Path = BASE_DIR) -> dict:
    nws_path = latest_nws_path(base_dir)
    if nws_path.exists():
        payload = json_loads(nws_path.read_text(encoding="utf-8"))
        current_hour = payload.get("current_hour")
        if current_hour:
            stored_local_time = datetime.fromisoformat(current_hour["valid_time_local"]).astimezone(
                LOCAL_TIMEZONE
            )
            now_local = datetime.now(LOCAL_TIMEZONE)
            if stored_local_time.year == now_local.year and stored_local_time.month == now_local.month and stored_local_time.day == now_local.day and stored_local_time.hour == now_local.hour:
                return payload
    return fetch_nws_hourly_forecast(base_dir=base_dir)


def run_hourly_pipeline() -> None:
    fetch_nws_hourly_forecast(base_dir=BASE_DIR)
    scripts = [
        ("/opt/render/project/src/hrrr_saratoga_temperature.py", "/opt/render/project/src"),
    ]
    run_scripts(scripts, 1, parallel=True, max_parallel=1)
    build_training_pairs_for_current_hour(BASE_DIR)


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2)


def json_loads(payload_text: str) -> dict:
    import json

    return json.loads(payload_text)


def load_training_pairs(base_dir: Path = BASE_DIR) -> list[dict]:
    dataset_path = training_pairs_path(base_dir)
    if not dataset_path.exists():
        return []
    return json_loads(dataset_path.read_text(encoding="utf-8"))


def save_training_pairs(pairs: list[dict], base_dir: Path = BASE_DIR) -> None:
    training_pairs_path(base_dir).write_text(json_dumps(pairs), encoding="utf-8")


def latest_training_pairs_info(base_dir: Path = BASE_DIR) -> dict | None:
    dataset_path = training_pairs_path(base_dir)
    if not dataset_path.exists():
        return None
    return {
        "name": dataset_path.name,
        "url": url_for("serve_training_pairs"),
    }


def build_training_pairs_for_current_hour(base_dir: Path = BASE_DIR) -> list[dict]:
    latest_nws = load_or_fetch_nws_hourly(base_dir)
    current_hour = latest_nws.get("current_hour")
    if not current_hour:
        return load_training_pairs(base_dir)

    valid_time_utc = current_hour["valid_time_utc"]
    nws_temp_f = current_hour.get("temperature_f")
    _, archives_dir = ensure_storage_dirs(base_dir)

    existing_pairs = load_training_pairs(base_dir)
    pair_index = {pair["pair_id"]: pair for pair in existing_pairs}

    for archive_path in archives_dir.glob("saratoga_springs_hrrr_*.json"):
        archive_payload = json_loads(archive_path.read_text(encoding="utf-8"))
        run_label = archive_payload.get("run_label")

        for point in archive_payload.get("series", []):
            if point.get("valid_time_utc") != valid_time_utc:
                continue

            forecast_hour = point.get("forecast_hour")
            hrrr_temp_f = point.get("temperature_f")
            pair_id = f"{run_label}_{forecast_hour:02d}_{valid_time_utc}"
            pair_index[pair_id] = {
                "pair_id": pair_id,
                "location": archive_payload.get("location", DEFAULT_LOCATION),
                "valid_time_utc": valid_time_utc,
                "valid_time_local": point.get("valid_time_local"),
                "verification_hour_label": latest_nws.get("verification_hour_label"),
                "hrrr_run_label": run_label,
                "hrrr_run_date": archive_payload.get("run_date"),
                "hrrr_cycle_hour": archive_payload.get("cycle_hour"),
                "forecast_hour": forecast_hour,
                "lead_hours": forecast_hour,
                "hrrr_temperature_f": hrrr_temp_f,
                "nws_temperature_f": nws_temp_f,
                "temperature_error_f": None if hrrr_temp_f is None or nws_temp_f is None else round(nws_temp_f - hrrr_temp_f, 2),
                "nws_short_forecast": current_hour.get("short_forecast"),
                "nws_is_daytime": current_hour.get("is_daytime"),
                "nws_generated_at": latest_nws.get("generated_at"),
            }

    updated_pairs = sorted(
        pair_index.values(),
        key=lambda item: (item["valid_time_utc"], item["hrrr_run_label"], item["forecast_hour"]),
    )
    save_training_pairs(updated_pairs, base_dir)
    return updated_pairs


def get_manifest_or_generate() -> dict:
    manifest = load_latest_manifest(BASE_DIR)
    if manifest is not None:
        return manifest

    artifacts = generate_latest_run_artifacts(base_dir=BASE_DIR)
    manifest = load_latest_manifest(BASE_DIR)
    if manifest is None:
        raise RuntimeError(f"Failed to create manifest for run {artifacts.run.run_label}.")
    return manifest


@app.get("/")
def index():
    manifest = get_manifest_or_generate()
    archives = list_recent_archives(BASE_DIR)
    nws_archives = list_nws_archives(BASE_DIR)
    nws_hourly = load_or_fetch_nws_hourly(BASE_DIR)
    nws_current = nws_hourly.get("current_hour")
    training_pairs = latest_training_pairs_info(BASE_DIR)
    return render_template(
        "index.html",
        manifest=manifest,
        image_url=url_for("serve_plot", filename=manifest["plot_file"]),
        archive_url=url_for("serve_archive", filename=manifest["archive_file"]),
        archives=archives,
        nws_archives=nws_archives,
        nws_hourly=nws_hourly,
        nws_current=nws_current,
        training_pairs=training_pairs,
    )


@app.route("/run-task1")
def run_task1():
    threading.Thread(
        target=run_hourly_pipeline,
        daemon=True,
    ).start()
    return "Task started in background! Check logs folder for output.", 200


@app.get("/plots/<path:filename>")
def serve_plot(filename: str):
    plots_dir = BASE_DIR / PLOTS_DIR_NAME
    if not (plots_dir / filename).exists():
        abort(404)
    return send_from_directory(plots_dir, filename)


@app.get("/archives/<path:filename>")
def serve_archive(filename: str):
    archives_dir = BASE_DIR / ARCHIVES_DIR_NAME
    if not (archives_dir / filename).exists():
        abort(404)
    return send_from_directory(archives_dir, filename, as_attachment=True)


@app.get("/nws/latest")
def serve_latest_nws_hourly():
    nws_path = latest_nws_path(BASE_DIR)
    if not nws_path.exists():
        fetch_nws_hourly_forecast(base_dir=BASE_DIR)
    return send_from_directory(nws_path.parent, nws_path.name, as_attachment=True)


@app.get("/nws/archive/<path:filename>")
def serve_nws_archive(filename: str):
    nws_dir = ensure_nws_dir(BASE_DIR)
    if not (nws_dir / filename).exists():
        abort(404)
    return send_from_directory(nws_dir, filename, as_attachment=True)


@app.get("/training/pairs")
def serve_training_pairs():
    dataset_path = training_pairs_path(BASE_DIR)
    if not dataset_path.exists():
        abort(404)
    return send_from_directory(dataset_path.parent, dataset_path.name, as_attachment=True)


if __name__ == "__main__":
    ensure_storage_dirs(BASE_DIR)
    ensure_nws_dir(BASE_DIR)
    ensure_training_dir(BASE_DIR)
    app.run(host="0.0.0.0", port=5000, debug=True)
