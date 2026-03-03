# https://github.com/AllenNeuralDynamics/aind-ophys-mesoscope-image-splitter/blob/c034d3d893dc7365498b61e5353337c1a4e45fb5/code/tiff_metadata.py#L19

import copy
import pathlib
from typing import List, Union

import tifffile


def _read_metadata(tiff_path: pathlib.Path):
    """
    Calls tifffile.read_scanimage_metadata on the specified
    path and returns the result. This method was factored
    out so that it could be easily mocked in unit tests.
    """
    return tifffile.read_scanimage_metadata(open(tiff_path, "rb"))


class ScanImageMetadata(object):
    """
    A class to handle reading and parsing the metadata that
    comes with the TIFF files produced by ScanImage

    Parameters
    ----------
    tiff_path: pathlib.Path
        Path to the TIFF file whose metadata we are parsing
    """

    def __init__(self, tiff_path: pathlib.Path):
        self._file_path = tiff_path
        if not tiff_path.is_file():
            raise ValueError(f"{tiff_path.resolve().absolute()} is not a file")
        self._metadata = _read_metadata(tiff_path)

    @property
    def file_path(self) -> pathlib.Path:
        return self._file_path

    @property
    def raw_metadata(self) -> tuple:
        """
        Return a copy of the raw metadata as read by
        tifffile.read_scanimage_metadata.
        """
        return copy.deepcopy(self._metadata)

    @property
    def numVolumes(self) -> int:
        """
        The metadata field representing the number of volumes
        recorded by the rig
        """
        if not hasattr(self, "_numVolumes"):
            value = self._metadata[0]["SI.hStackManager.actualNumVolumes"]
            if not isinstance(value, int):
                raise ValueError(
                    f"in {self._file_path}\n"
                    "SI.hStackManager.actualNumVolumes is a "
                    f"{type(value)}; expected int"
                )

            self._numVolumes = value

        return self._numVolumes

    @property
    def numSlices(self) -> int:
        """
        The metadata field representing the number of slices
        recorded by the rig
        """
        if not hasattr(self, "_numSlices"):
            value = self._metadata[0]["SI.hStackManager.actualNumSlices"]
            if not isinstance(value, int):
                raise ValueError(
                    f"in {self._file_path}\n"
                    "SI.hStackManager.actualNumSlices is a "
                    f"{type(value)}; expected int"
                )
            self._numSlices = value

        return self._numSlices

    @property
    def channelSave(self) -> Union[int, List[int]]:
        """
        The metadata field representing which channels were saved
        in this TIFF. Either 1 or [1, 2]
        """
        if not hasattr(self, "_channelSave"):
            self._channelSave = self._metadata[0]["SI.hChannels.channelSave"]
        return self._channelSave

    @property
    def defined_rois(self) -> List[dict]:
        """
        Get the ROIs defined in this TIFF file

        This is list of dicts, each dict containing the ScanImage
        metadata for a given ROI

        In this context, an ROI is a 3-dimensional volume of the brain
        that was scanned by the microscope.
        """
        if not hasattr(self, "_defined_rois"):
            roi_parent = self._metadata[1]["RoiGroups"]
            roi_group = roi_parent["imagingRoiGroup"]["rois"]
            if isinstance(roi_group, dict):
                self._defined_rois = [
                    roi_group,
                ]
            elif isinstance(roi_group, list):
                self._defined_rois = roi_group
            else:
                msg = "unable to parse "
                msg += "self._metadata[1]['RoiGroups']"
                msg += "['imagingROIGroup']['rois'] "
                msg += f"of type {type(roi_group)}"
                raise RuntimeError(msg)

        # use copy to make absolutely sure self._defined_rois
        # is not accidentally changed downstream
        return copy.deepcopy(self._defined_rois)

    @property
    def n_rois(self) -> int:
        """
        Number of ROIs defined in the metadata for this TIFF file.
        """
        if not hasattr(self, "_n_rois"):
            self._n_rois = len(self.defined_rois)
        return self._n_rois

    def zs_for_roi(self, i_roi: int) -> List[int]:
        """
        Return a list of the z-values at which the specified
        ROI was scanned
        """
        if i_roi >= self.n_rois:
            msg = f"You asked for ROI {i_roi}; "
            msg += f"there are only {self.n_rois} "
            msg += "specified in this TIFF file"
            raise ValueError(msg)
        return self.defined_rois[i_roi]["zs"]
