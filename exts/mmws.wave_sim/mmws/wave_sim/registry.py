"""Name -> class registries for the swappable parts of the simulator.

The whole point of this project is being able to switch wave spectrum, wave
field and force model independently, including at runtime from a UI dropdown.
That needs exactly one thing: a mapping from a human-readable name to a class.

Adding a new implementation is then three lines and no edits to existing code:

    from ..registry import SPECTRA

    @SPECTRA.register("my_spectrum")
    class MySpectrum(WaveSpectrum):
        ...

then import the module from `mmws/wave_sim/__init__.py` so the decorator runs.
"""

from __future__ import annotations

from typing import Callable, Dict, Type, TypeVar

T = TypeVar("T")


class Registry:
    """A case-insensitive name -> class map with helpful failure messages."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: Dict[str, type] = {}

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Class decorator. Also stamps `cls.name` so a class knows its own key."""

        def decorator(cls: Type[T]) -> Type[T]:
            key = name.lower()
            if key in self._entries:
                raise ValueError(
                    f"{self._kind} {name!r} is already registered by "
                    f"{self._entries[key].__name__}"
                )
            cls.name = name          # type: ignore[attr-defined]
            self._entries[key] = cls
            return cls

        return decorator

    def get(self, name: str) -> type:
        try:
            return self._entries[name.lower()]
        except KeyError:
            available = ", ".join(self.names()) or "(none registered)"
            raise KeyError(
                f"unknown {self._kind} {name!r}. Available: {available}"
            ) from None

    def create(self, name: str, **kwargs):
        """Look the class up and construct it. The swap, in one call."""
        return self.get(name)(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._entries

    def __repr__(self) -> str:
        return f"<Registry {self._kind}: {', '.join(self.names()) or 'empty'}>"


# The three axes of the "multi-modal" in the project name.
SPECTRA = Registry("spectrum")
FIELDS = Registry("wave field")
FORCES = Registry("force model")
