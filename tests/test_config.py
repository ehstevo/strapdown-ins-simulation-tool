from pathlib import Path

from ins.config import config_to_mapping, config_to_toml, load_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_figure_eight_configuration_loads() -> None:
    config = load_config(REPOSITORY_ROOT / "configs" / "figure_eight.toml")

    assert config.name == "Figure-Eight INS Error Propagation"
    assert config.runs == 100
    assert config.scenario.name == "figure_eight"
    assert config.altitude_aiding is True
    assert config.initial_error.position.std.shape == (3,)
    assert config.gyro.fogm_tau_s.shape == (3,)


def test_distribution_means_are_fixed_at_zero() -> None:
    config = load_config(REPOSITORY_ROOT / "configs" / "figure_eight.toml")

    assert not config.initial_error.position.mean.any()
    assert not config.initial_error.velocity.mean.any()
    assert not config.initial_error.attitude_ned_rad.mean.any()
    assert not config.gyro.constant_bias.mean.any()
    assert not config.accel.constant_bias.mean.any()


def test_configuration_round_trips_through_toml(tmp_path: Path) -> None:
    original = load_config(REPOSITORY_ROOT / "configs" / "circle.toml")
    output = tmp_path / "round_trip.toml"
    output.write_text(config_to_toml(original))

    recovered = load_config(output)

    assert config_to_mapping(recovered) == config_to_mapping(original)
