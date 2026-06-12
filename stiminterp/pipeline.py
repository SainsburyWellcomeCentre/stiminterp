from pathlib import Path

from stiminterp import remove_photostim_artefacts
from stiminterp.load_data.custom_data_loader import get_artefact_dfs
from stiminterp.load_data.scanimage_metadata import ScanImageMetadata


def run_stiminterp(
    input_tif: str,
    input_h5: str | None = None,
    output_tif: str | None = None,
    save_stim_df: bool = True,
    skip_noh5: bool = True,
):
    tif_path = Path(input_tif)
    sim = ScanImageMetadata(tif_path)

    # infer h5 if not provided
    h5_path = (
        tif_path.with_suffix(".h5") if input_h5 is None else Path(input_h5)
    )

    # determine output path
    if output_tif is None:
        out_path = tif_path.with_name(f"{tif_path.stem}_corrected.tif")
    else:
        out_tmp = Path(output_tif)
        if out_tmp.is_dir():
            out_path = out_tmp / f"{tif_path.stem}_corrected.tif"
        else:
            out_path = out_tmp

    if not h5_path.exists():
        if skip_noh5:
            if not out_path.exists():
                out_path.symlink_to(tif_path.resolve())
            return None

    df_frames, df_stims = get_artefact_dfs(
        h5_path, "FrameTTL", "SatsumaGateTTL"
    )

    df_split = remove_photostim_artefacts(
        input_tif,
        str(out_path),
        df_frames,
        df_stims,
        frame_gap=sim.n_rois - 1,
        num_channel=sim.n_chans,
    )

    # Save csv
    if save_stim_df:
        csv_path = out_path.with_name(f"{tif_path.stem}_stim.csv")
        df_split.to_csv(csv_path, index=False)

    return None
