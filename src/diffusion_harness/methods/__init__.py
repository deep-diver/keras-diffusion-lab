"""Research methods plug-in directory.

Each method is a self-contained module that provides:
  - build_model(config) -> keras.Model
  - build_trainer(config) -> BaseTrainer
  - build_sampler(model, config) -> BaseSampler
"""

import importlib

_METHODS = {
    "unconditional": "diffusion_harness.methods.unconditional",
    "class_conditional": "diffusion_harness.methods.class_conditional",
}


def get_method(name):
    """Get a method module by name."""
    if name not in _METHODS:
        raise ValueError(f"Unknown method: {name}. Available: {list(_METHODS.keys())}")
    return importlib.import_module(_METHODS[name])


def list_methods():
    """List available method names."""
    return list(_METHODS.keys())
