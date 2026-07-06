"""Full-gradient audit package."""

__all__ = ["SequentialBackwardDomainGradientTracker"]


def __getattr__(name: str) -> object:
    if name == "SequentialBackwardDomainGradientTracker":
        from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker

        return SequentialBackwardDomainGradientTracker
    raise AttributeError(name)
