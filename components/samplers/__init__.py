__SAMPLER__ = {}


def register_sampler(name: str):
    def wrapper(cls):
        if name in __SAMPLER__:
            raise ValueError(f"Sampler '{name}' already registered.")
        __SAMPLER__[name] = cls
        return cls
    return wrapper


def get_sampler_class(name: str):
    # Lazy import so all sampler modules register on first call
    if not __SAMPLER__:
        from . import ddim, euler, dpm  # noqa: F401
    if name not in __SAMPLER__:
        raise ValueError(f"Unknown sampler '{name}'. Available: {list(__SAMPLER__)}")
    return __SAMPLER__[name]
