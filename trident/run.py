import importlib.util
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

cwd = Path.cwd()


def get_train_module_path(cwd: Path) -> None | Path:
    """Return the path to train.py if it exists in either $CWD or $CWD/src, otherwise return None."""
    possible_paths = [cwd.joinpath("train.py"), cwd.joinpath("src", "train.py")]
    for path in possible_paths:
        if path.exists():
            return path
    return None


@hydra.main(
    version_base="1.3",
    config_path=str(cwd.joinpath("configs/")),
    config_name="config.yaml",
)
def main(cfg: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    train_module_path = get_train_module_path(cwd)
    if train_module_path:
        # Create a unique module name based on the train_module_path
        module_name = train_module_path.stem + "." + train_module_path.parent.name
        spec = importlib.util.spec_from_file_location(module_name, train_module_path)
        assert spec is not None
        train_mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = train_mod
        if exec_module := getattr(spec.loader, "exec_module"):
            exec_module(train_mod)
        train = train_mod.train
    else:
        from trident.train import train
    from trident.utils.runner import extras, print_config

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    extras(cfg)
    # Init lightning datamodule

    # Pretty print config using Rich library
    if cfg.get("print_config"):
        print_config(cfg, resolve=True)

    # Train model
    return train(cfg)


if __name__ == "__main__":
    main()
