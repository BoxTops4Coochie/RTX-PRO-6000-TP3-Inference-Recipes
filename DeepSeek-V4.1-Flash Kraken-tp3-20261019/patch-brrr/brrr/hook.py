import sys
from collections.abc import Mapping, Sequence
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
from types import ModuleType

from brrr.patches import PATCHES, Patch


class PatchingLoader:
    def __init__(self, inner: Loader, patch: Patch) -> None:
        self.inner = inner
        self.patch = patch

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        return self.inner.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.inner.exec_module(module)
        self.patch(module)


class PatchingFinder:
    def __init__(self, patches: Mapping[str, Patch]) -> None:
        self.patches = patches

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        patch = self.patches.get(fullname)
        if patch is None:
            return None
        spec = next(found for finder in sys.meta_path if finder is not self and (found := finder.find_spec(fullname, path, target)) is not None)
        spec.loader = PatchingLoader(spec.loader, patch)
        return spec


sys.meta_path.insert(0, PatchingFinder(PATCHES))
