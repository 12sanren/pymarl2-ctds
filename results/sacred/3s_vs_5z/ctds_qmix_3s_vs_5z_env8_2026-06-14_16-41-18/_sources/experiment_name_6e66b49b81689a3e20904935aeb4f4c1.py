import datetime
import os
import re
from os.path import abspath, dirname


def apply_cli_overrides(config, params):
    """Apply Sacred `with key=value` overrides before building log names."""
    if "with" not in params:
        return config

    with_idx = params.index("with")
    for arg in params[with_idx + 1:]:
        if arg.startswith("-"):
            break
        if "=" not in arg:
            continue

        key, raw_value = arg.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        if key == "env_args.map_name":
            config.setdefault("env_args", {})["map_name"] = raw_value
            continue

        if key == "batch_size_run":
            config[key] = int(raw_value)
            continue

        if raw_value.lower() in ("true", "false"):
            config[key] = raw_value.lower() == "true"
            continue

        try:
            config[key] = int(raw_value)
        except ValueError:
            try:
                config[key] = float(raw_value)
            except ValueError:
                config[key] = raw_value

    return config


def get_results_root():
    return dirname(dirname(dirname(abspath(__file__))))


def get_map_name(config):
    env_args = config.get("env_args", {})
    return env_args.get("map_name") or env_args.get("key", "unknown")


def get_algorithm_name(config):
    if config.get("config_name"):
        return str(config["config_name"])
    name = config.get("name", "unknown")
    if "_env=" in name:
        return name.split("_env=")[0]
    return name


def _sanitize_token(value):
    return re.sub(r"[^\w\-.]+", "_", str(value))


def build_experiment_name(config, timestamp=None):
    """Build log dir name: {algorithm}_{map}_env{N}_{timestamp}."""
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    algo = _sanitize_token(get_algorithm_name(config))
    map_name = _sanitize_token(get_map_name(config))
    batch_size_run = config.get("batch_size_run", 1)

    return "{}_{}_env{}_{}".format(algo, map_name, batch_size_run, timestamp)


def setup_experiment_logging(config, args, logger):
    experiment_name = config.get("experiment_name") or build_experiment_name(config)
    config["experiment_name"] = experiment_name
    args.experiment_name = experiment_name
    args.unique_token = experiment_name

    if args.use_tensorboard:
        tb_exp_direc = os.path.join(
            get_results_root(), "results", "tb_logs", experiment_name
        )
        logger.setup_tb(tb_exp_direc)

    return experiment_name
