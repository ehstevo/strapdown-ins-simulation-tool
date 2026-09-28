"""Configuration loading and validation for Monte Carlo INS simulations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tomllib
from typing import Any, Mapping

import numpy as np


class ConfigurationError(ValueError):
    """Raised when a simulation configuration is incomplete or invalid."""


def _vector(table: dict, key: str, *, unit: str) -> np.ndarray:
    try:
        values = np.asarray(table[key], dtype=float)
    except KeyError as error:
        raise ConfigurationError(f"Missing '{key}' ({unit}).") from error

    if values.shape != (3,):
        raise ConfigurationError(f"'{key}' must contain exactly three {unit} values.")
    return values


def _nonnegative(values: np.ndarray, name: str) -> np.ndarray:
    if np.any(values < 0.0):
        raise ConfigurationError(f"'{name}' must be non-negative.")
    return values


def _positive(values: np.ndarray, name: str) -> np.ndarray:
    if np.any(values <= 0.0):
        raise ConfigurationError(f"'{name}' must be strictly positive.")
    return values


@dataclass(frozen=True)
class GaussianVector:
    """Independent three-axis Gaussian distribution."""

    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class SensorErrorModel:
    """Stochastic error settings for one three-axis IMU sensor."""

    constant_bias: GaussianVector
    fogm_sigma: np.ndarray
    fogm_tau_s: np.ndarray
    white_measurement_noise_density: np.ndarray
    bias_rate_random_walk_intensity: np.ndarray


@dataclass(frozen=True)
class InitialErrorModel:
    """Run-to-run initial navigation-state error distributions."""

    position: GaussianVector
    velocity: GaussianVector
    attitude_ned_rad: GaussianVector


@dataclass(frozen=True)
class ScenarioConfig:
    """Truth-trajectory settings shared by every Monte Carlo run."""

    name: str
    origin_lat_deg: float
    origin_lon_deg: float
    origin_hae_m: float
    radius_m: float
    altitude_amplitude_m: float
    bank_rad: float
    pitch_rad: float


@dataclass(frozen=True)
class SimulationConfig:
    """Complete validated configuration for one Monte Carlo experiment."""

    name: str
    runs: int
    seed: int
    sample_period_s: float
    duration_s: float
    altitude_aiding: bool
    output_dir: Path
    scenario: ScenarioConfig
    initial_error: InitialErrorModel
    gyro: SensorErrorModel
    accel: SensorErrorModel


def _zero_mean_gaussian_vector(table: dict, prefix: str, unit: str) -> GaussianVector:
    """Read a three-axis standard deviation and fix its distribution mean at zero."""

    std = _nonnegative(_vector(table, f"{prefix}_std", unit=unit), f"{prefix}_std")
    return GaussianVector(mean=np.zeros(3), std=std)


def _sensor_error_model(table: dict, sensor_name: str) -> SensorErrorModel:
    if sensor_name == "gyro":
        bias_unit = "rad/s"
        density_unit = "rad/sqrt(s)"
        random_walk_unit = "rad/s/sqrt(s)"
        white_noise_key = "angle_random_walk"
    else:
        bias_unit = "m/s^2"
        density_unit = "m/s/sqrt(s)"
        random_walk_unit = "m/s^2/sqrt(s)"
        white_noise_key = "velocity_random_walk"

    return SensorErrorModel(
        constant_bias=_zero_mean_gaussian_vector(table, "constant_bias", bias_unit),
        fogm_sigma=_nonnegative(_vector(table, "fogm_sigma", unit=bias_unit), "fogm_sigma"),
        fogm_tau_s=_positive(_vector(table, "fogm_tau_s", unit="s"), "fogm_tau_s"),
        white_measurement_noise_density=_nonnegative(
            _vector(table, white_noise_key, unit=density_unit),
            white_noise_key,
        ),
        bias_rate_random_walk_intensity=_nonnegative(
            _vector(table, "bias_rate_random_walk", unit=random_walk_unit),
            "bias_rate_random_walk",
        ),
    )


def config_from_mapping(raw: Mapping[str, Any], default_name: str = "simulation") -> SimulationConfig:
    """Validate a TOML- or JSON-compatible mapping as a simulation configuration."""

    try:
        simulation = raw["simulation"]
        scenario = raw["scenario"]
        initial_error = raw["initial_error"]
        sensors = raw["sensors"]
    except KeyError as error:
        raise ConfigurationError(f"Missing required section: [{error.args[0]}]") from error

    try:
        runs = int(simulation["runs"])
        sample_period_s = float(simulation["sample_period_s"])
        duration_s = float(simulation["duration_s"])
    except KeyError as error:
        raise ConfigurationError(f"Missing simulation setting: {error.args[0]}") from error

    if runs < 1:
        raise ConfigurationError("simulation.runs must be at least one.")
    if sample_period_s <= 0.0:
        raise ConfigurationError("simulation.sample_period_s must be positive.")
    if duration_s <= 0.0:
        raise ConfigurationError("simulation.duration_s must be positive.")
    if not np.isclose(duration_s / sample_period_s, round(duration_s / sample_period_s)):
        raise ConfigurationError("duration_s must be an integer multiple of sample_period_s.")

    scenario_name = str(scenario.get("name", ""))
    if scenario_name not in {"figure_eight", "circle", "stationary"}:
        raise ConfigurationError("scenario.name must be figure_eight, circle, or stationary.")

    scenario_config = ScenarioConfig(
        name=scenario_name,
        origin_lat_deg=float(scenario["origin_lat_deg"]),
        origin_lon_deg=float(scenario["origin_lon_deg"]),
        origin_hae_m=float(scenario["origin_hae_m"]),
        radius_m=float(scenario.get("radius_m", 0.0)),
        altitude_amplitude_m=float(scenario.get("altitude_amplitude_m", 0.0)),
        bank_rad=np.deg2rad(float(scenario.get("bank_deg", 0.0))),
        pitch_rad=np.deg2rad(float(scenario.get("pitch_deg", 0.0))),
    )
    if scenario_name != "stationary" and scenario_config.radius_m <= 0.0:
        raise ConfigurationError("scenario.radius_m must be positive for moving scenarios.")
    if scenario_config.altitude_amplitude_m < 0.0:
        raise ConfigurationError("scenario.altitude_amplitude_m must be non-negative.")

    return SimulationConfig(
        name=str(simulation.get("name", default_name)),
        runs=runs,
        seed=int(simulation.get("seed", 0)),
        sample_period_s=sample_period_s,
        duration_s=duration_s,
        altitude_aiding=bool(simulation.get("altitude_aiding", True)),
        output_dir=Path(simulation.get("output_dir", f"results/{default_name}")),
        scenario=scenario_config,
        initial_error=InitialErrorModel(
            position=_zero_mean_gaussian_vector(initial_error, "position", "m in NED"),
            velocity=_zero_mean_gaussian_vector(initial_error, "velocity", "m/s in NED"),
            attitude_ned_rad=GaussianVector(
                mean=np.zeros(3),
                std=np.deg2rad(
                    _nonnegative(
                        _vector(initial_error, "attitude_std_deg", unit="deg in NED"),
                        "attitude_std_deg",
                    )
                ),
            ),
        ),
        gyro=_sensor_error_model(sensors["gyro"], "gyro"),
        accel=_sensor_error_model(sensors["accel"], "accel"),
    )


def load_config(path: str | Path) -> SimulationConfig:
    """Load and validate a Monte Carlo simulation TOML file."""

    path = Path(path)
    try:
        with path.open("rb") as config_file:
            raw = tomllib.load(config_file)
    except FileNotFoundError as error:
        raise ConfigurationError(f"Configuration file not found: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigurationError(f"Invalid TOML in {path}: {error}") from error

    return config_from_mapping(raw, default_name=path.stem)


def _python_vector(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


def _sensor_mapping(model: SensorErrorModel, sensor_name: str) -> dict[str, list[float]]:
    white_noise_key = "angle_random_walk" if sensor_name == "gyro" else "velocity_random_walk"
    return {
        "constant_bias_std": _python_vector(model.constant_bias.std),
        "fogm_sigma": _python_vector(model.fogm_sigma),
        "fogm_tau_s": _python_vector(model.fogm_tau_s),
        white_noise_key: _python_vector(model.white_measurement_noise_density),
        "bias_rate_random_walk": _python_vector(model.bias_rate_random_walk_intensity),
    }


def config_to_mapping(config: SimulationConfig) -> dict[str, Any]:
    """Convert a validated configuration into JSON- and TOML-compatible values."""

    return {
        "simulation": {
            "name": config.name,
            "runs": config.runs,
            "seed": config.seed,
            "sample_period_s": config.sample_period_s,
            "duration_s": config.duration_s,
            "altitude_aiding": config.altitude_aiding,
            "output_dir": str(config.output_dir),
        },
        "scenario": {
            "name": config.scenario.name,
            "origin_lat_deg": config.scenario.origin_lat_deg,
            "origin_lon_deg": config.scenario.origin_lon_deg,
            "origin_hae_m": config.scenario.origin_hae_m,
            "radius_m": config.scenario.radius_m,
            "altitude_amplitude_m": config.scenario.altitude_amplitude_m,
            "bank_deg": float(np.rad2deg(config.scenario.bank_rad)),
            "pitch_deg": float(np.rad2deg(config.scenario.pitch_rad)),
        },
        "initial_error": {
            "position_std": _python_vector(config.initial_error.position.std),
            "velocity_std": _python_vector(config.initial_error.velocity.std),
            "attitude_std_deg": _python_vector(np.rad2deg(config.initial_error.attitude_ned_rad.std)),
        },
        "sensors": {
            "gyro": _sensor_mapping(config.gyro, "gyro"),
            "accel": _sensor_mapping(config.accel, "accel"),
        },
    }


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return repr(value)


def config_to_toml(config: SimulationConfig) -> str:
    """Serialize a validated simulation configuration to readable TOML."""

    mapping = config_to_mapping(config)
    sections = (
        ("simulation", mapping["simulation"]),
        ("scenario", mapping["scenario"]),
        ("initial_error", mapping["initial_error"]),
        ("sensors.gyro", mapping["sensors"]["gyro"]),
        ("sensors.accel", mapping["sensors"]["accel"]),
    )
    lines: list[str] = []
    for section_name, values in sections:
        lines.append(f"[{section_name}]")
        lines.extend(f"{key} = {_toml_value(value)}" for key, value in values.items())
        lines.append("")
    return "\n".join(lines)


def save_config(config: SimulationConfig, path: str | Path) -> Path:
    """Write a validated configuration as a standalone TOML file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config_to_toml(config))
    return path
