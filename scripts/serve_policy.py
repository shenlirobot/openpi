import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


class LiberoModelType(enum.Enum):
    """LIBERO model types."""

    LIBERO = "libero"  # Uses pi05_libero config and checkpoint
    BASE = "base"      # Uses pi05_base with LIBERO normalization stats


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # For LIBERO environment, specify which model type to use
    libero_model_type: LiberoModelType = LiberoModelType.BASE

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | None = None


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}

# LIBERO-specific checkpoints
LIBERO_CHECKPOINTS: dict[LiberoModelType, Checkpoint] = {
    LiberoModelType.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
    LiberoModelType.BASE: Checkpoint(
        config="pi05_base_libero_norm",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None, libero_model_type: LiberoModelType = LiberoModelType.BASE) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if env == EnvMode.LIBERO:
        checkpoint = LIBERO_CHECKPOINTS[libero_model_type]
    elif checkpoint := DEFAULT_CHECKPOINT.get(env):
        pass
    else:
        raise ValueError(f"Unsupported environment mode: {env}")
    
    train_config = _config.get_config(checkpoint.config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    
    # Load normalization stats from AssetsConfig if available
    norm_stats = None
    if hasattr(data_config, 'norm_stats') and data_config.norm_stats is not None:
        norm_stats = data_config.norm_stats
    
    return _policy_config.create_trained_policy(
        train_config, checkpoint.dir, default_prompt=default_prompt, norm_stats=norm_stats
    )


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    if args.policy is not None:
        return _policy_config.create_trained_policy(
            _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
        )
    else:
        return create_default_policy(
            args.env, 
            default_prompt=args.default_prompt,
            libero_model_type=args.libero_model_type
        )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
