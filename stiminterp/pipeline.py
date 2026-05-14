from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ScanImageTiffReader import ScanImageTiffReader
from tifffile import imwrite

from stiminterp import remove_photostim_artefacts
from stiminterp.load_data.custom_data_loader import get_artefact_dfs
from stiminterp.load_data.scanimage_metadata import ScanImageMetadata
from stiminterp.plotting_hooks.sanity_check import (
    create_sanitycheck_axes,
    plot_removal,
)


def run_stiminterp(
    input_tif: str,
    input_h5: str | None = None,
    output_tif: str | None = None,
    save_stim_df: bool = True,
    save_sanity_plot: bool = True,
    skip_noh5: bool = True,
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

    if not h5_path.exists():
        if skip_noh5:
            if not out_path.exists():
                out_path.symlink_to(tif_path.resolve())
            return None

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

    # Save tif
    imwrite(str(out_path), corrected)

    # Save csv
    if save_stim_df:
        csv_path = out_path.with_name(
            out_path.name.replace("_corrected.tif", "_stim.csv")
        )
        df_split.to_csv(csv_path, index=False)

    # Save sanity check
    if save_sanity_plot:
        pdf_path = out_path.with_name(
            out_path.name.replace("_corrected.tif", "_sanitycheck.pdf")
        )
        with PdfPages(pdf_path) as pdf:
            for i in range(len(df_split)):
                fig, axes = create_sanitycheck_axes(sim.n_chans)
                for ch in range(sim.n_chans):
                    ttl_frame = df_split.frame[i]
                    movie_frame = ttl_frame * sim.n_chans + ch
                    plot_removal(
                        axes_row=axes[ch],
                        frame=movie_frame,
                        y_frac_start=df_split.frac_start[i],
                        y_frac_stop=df_split.frac_stop[i],
                        uncorrected=vol,
                        corrected=corrected,
                        bad_mask=bad_mask,
                        channel=ch,
                    )
                pdf.savefig(fig)
                plt.close(fig)

    return None
