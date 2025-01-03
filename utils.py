import jax
from jax.experimental.compilation_cache import compilation_cache
from pathlib import Path


def jax_jit_setup(config):
    cache_path = "/tmp/jax_cache"

    # jax.config.update('jax_log_compiles', True)

    # setup compilation cache
    jax.config.update("jax_compilation_cache_dir", cache_path)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_explain_cache_misses", config.jax_explain_cache_misses)

    compilation_cache.initialize_cache(cache_path)
    # jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


EXPERIMENT_PATH = Path(__file__).parent.joinpath("experiments")
EXPERIMENT_PATH.mkdir(exist_ok=True)

DATASET_PATH = Path(__file__).parent.joinpath("data")
DATASET_PATH.mkdir(exist_ok=True)

CONFIG_PATH = Path(__file__).parent.joinpath("configs")
CONFIG_PATH.mkdir(exist_ok=True)
