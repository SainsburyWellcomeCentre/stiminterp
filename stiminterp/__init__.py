from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("stiminterp")
except PackageNotFoundError:
    # package is not installed
    pass

from .stim_interpolate import remove_photostim_artefacts, StimInterpConfig
# from .pipeline import FullPipeline
