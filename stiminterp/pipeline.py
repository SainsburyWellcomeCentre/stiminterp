from pathlib import Path

from ScanImageTiffReader import ScanImageTiffReader
from tifffile import imwrite

from stiminterp import remove_photostim_artefacts
from stiminterp.load_data.custom_data_loader import get_artefact_dfs
from stiminterp.load_data.scanimage_metadata import ScanImageMetadata


def run_stiminterp(
    input_tif: str,
    input_h5: str | None = None,
    output_tif: str | None = None,
):
    tif_path = Path(input_tif)
    sim = ScanImageMetadata(tif_path)

    # infer h5 if not provided
    if input_h5 is None:
        h5_path = tif_path.with_suffix(".h5")
    else:
        h5_path = Path(input_h5)

    # determine output path
    if output_tif is None:
        out_path = tif_path.with_name(f"{tif_path.stem}_corrected.tif")
    else:
        out_tmp = Path(output_tif)
        if out_tmp.is_dir():
            out_path = out_tmp / f"{tif_path.stem}_corrected.tif"
        else:
            out_path = out_tmp

    vol = ScanImageTiffReader(input_tif).data()

    df_frames, df_stims = get_artefact_dfs(
        h5_path,
        "FrameTTL",
        "SatsumaGateTTL",
    )

    corrected, bad_mask, df_split = remove_photostim_artefacts(
        vol,
        df_frames,
        df_stims,
        frame_gap=sim.n_rois - 1,
        num_channel=sim.n_chans,
    )

    imwrite(str(out_path), corrected)
