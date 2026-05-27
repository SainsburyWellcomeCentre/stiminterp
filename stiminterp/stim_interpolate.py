"""
stim_interpolate.py

Photostimulation artefact removal via 1D interpolation.

Pipeline
--------
1. Convert input TIFF to a plain contiguous BigTIFF at a temp path using
   tifffile page-by-page (works for any TIFF including ScanImage with
   per-frame metadata gaps that are not memmap-able).
2. Open the temp copy as a writable memmap (always works: contiguous BigTIFF).
3. Compute artefact regions: frame index + scanline fractions from
   frame/stim timing DataFrames.
4. For each bad TTL frame, find the nearest good flanking frame on each side
   within the same plane.
5. Load [flank_before, bad_frame, flank_after] (≤3 pages), NaN the
   contaminated scanlines, linearly interpolate with _interp_block_numpy,
   write the corrected frame back into the memmap in place.
6. Flush, close, rename temp → output_tif.

Design rationale
----------------
Artefacts are sparse: only a handful of TTL frames are contaminated.
The copy step pays one full sequential read; after that only bad frames are
touched. No ScanImageLazyTiff dependency — tifffile.TiffFile handles all
ScanImage variants natively.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tifffile

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class StimInterpConfig:
    """Configuration for stim artefact removal."""

    pad_rows: int = 5
    """Extra scanline rows to mark bad above/below the artefact boundary."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def remove_photostim_artefacts(
    input_tif: str,
    output_tif: str,
    df_frames: pd.DataFrame,
    df_stims: pd.DataFrame,
    frame_gap: Optional[int] = None,
    num_channel: Optional[int] = None,
    cfg: Optional[StimInterpConfig] = None,
) -> pd.DataFrame:
    """
    In-place sparse photostim artefact removal.

    Converts ``input_tif`` to a plain contiguous BigTIFF at a temp path,
    patches only the bad frames directly in the memmap, then renames to
    ``output_tif``.  A crash never leaves a half-patched file at the output.

    Parameters
    ----------
    input_tif : str
        Path to the ScanImage TIFF/BigTIFF (any format readable by tifffile).
    output_tif : str
        Path for the corrected copy.  Must not equal ``input_tif``.
    df_frames : pd.DataFrame
        Columns ``["start", "stop"]`` — one row per TTL frame.
    df_stims : pd.DataFrame
        Columns ``["start", "stop"]`` — one row per photostim interval.
    frame_gap : int, optional
        Plane interleave: ``num_planes = frame_gap + 1``.
        ``None`` → single plane.
    num_channel : int, optional
        Number of interleaved channels (channel-fast ordering).
        If ``None``, inferred as ``T // n_ttl``.
    cfg : StimInterpConfig, optional

    Returns
    -------
    df_split : pd.DataFrame
    """
    if cfg is None:
        cfg = StimInterpConfig()

    input_tif = str(input_tif)
    output_tif = str(output_tif)
    if Path(input_tif).resolve() == Path(output_tif).resolve():
        raise ValueError("input_tif and output_tif must be different paths.")

    # Step 1: copy as plain contiguous BigTIFF → always memmap-writable
    tmp_tif = Path(output_tif).with_suffix(".tmp.tif")
    _copy_as_contiguous_tiff(input_tif, tmp_tif)

    # Step 2: open temp copy as writable memmap
    try:
        out = tifffile.memmap(str(tmp_tif), mode="r+")
    except (ValueError, MemoryError) as exc:
        tmp_tif.unlink(missing_ok=True)
        raise RuntimeError(
            f"Cannot memmap '{tmp_tif}' even after contiguous copy. "
            f"Original error: {exc}"
        ) from exc

    if out.ndim > 3:
        out = out.reshape(-1, out.shape[-2], out.shape[-1])
    T, Y, X = out.shape

    n_ttl = len(df_frames)
    if n_ttl <= 0:
        raise ValueError("df_frames is empty.")
    nchan = int(num_channel) if num_channel is not None else T // n_ttl
    if nchan <= 0 or (n_ttl * nchan) != T:
        raise ValueError(
            f"movie T={T} must equal len(df_frames)*nchan "
            f"({n_ttl}*{nchan}={n_ttl * nchan})."
        )

    # Step 3: compute artefact regions
    df_split = _artefact_regions(df_frames, df_stims)
    if df_split.empty:
        _flush_close(out)
        tmp_tif.rename(output_tif)
        return df_split

    # Plane assignment for every TTL frame
    num_planes = 1 if frame_gap is None else int(frame_gap) + 1
    plane_ids = np.arange(n_ttl) % num_planes

    # Set of all bad TTL frames
    bad_ttl_all = set(
        int(t) for t in df_split["frame"].unique() if 0 <= int(t) < n_ttl
    )

    # Bad-scanline mask: (n_ttl, Y) bool
    bad_lines = _build_bad_line_mask(
        df_split, T=n_ttl, Y=Y, pad_rows=cfg.pad_rows
    )

    # Steps 4 + 5: patch each bad TTL frame
    skipped = []
    try:
        for bad_ttl in sorted(bad_ttl_all):
            for c in range(nchan):
                bad_movie = bad_ttl * nchan + c

                flank_before, flank_after = _find_flanks(
                    bad_ttl, plane_ids, bad_ttl_all, n_ttl
                )

                if flank_before is None and flank_after is None:
                    skipped.append(bad_movie)
                    continue

                # Load ≤3 frames as float32
                flanks = [flank_before, bad_ttl, flank_after]
                frames_to_load = [
                    f * nchan + c for f in flanks if f is not None
                ]
                chunk = np.array(out[frames_to_load], dtype=np.float32)

                # bad frame is at index 0 if there's no flank_before, else 1
                bad_idx = 0 if flank_before is None else 1

                # NaN the bad scanlines
                bl = bad_lines[bad_ttl]
                if bl.any():
                    chunk[bad_idx, bl, :] = np.nan

                # x-coordinates: true TTL frame indices
                x_coords = np.array(
                    [f for f in flanks if f is not None], dtype=np.float32
                )

                # Interpolate in place
                flat = chunk.reshape(len(frames_to_load), -1)
                _interp_block_numpy(
                    block=flat,
                    x=x_coords,
                    donor_mask=np.ones(len(frames_to_load), dtype=bool),
                    require_n_good=1,
                )

                # Clip to dtype range and write back
                dinfo = np.iinfo(out.dtype)
                corrected = np.clip(flat[bad_idx], dinfo.min, dinfo.max)
                out[bad_movie] = corrected.reshape(Y, X).astype(out.dtype)

    except Exception:
        _flush_close(out)
        tmp_tif.unlink(missing_ok=True)
        raise

    if skipped:
        print(
            f"Warning: {len(skipped)} frames had no good flanking frame "
            f"in the same plane and were left uncorrected: {skipped}"
        )

    # Step 6: flush, close, rename temp → final output
    _flush_close(out)
    tmp_tif.rename(output_tif)
    return df_split


# ---------------------------------------------------------------------------
# TIFF copy helper
# ---------------------------------------------------------------------------


def _copy_as_contiguous_tiff(input_tif: str, output_tif: Path) -> None:
    """
    Read ``input_tif`` page by page with tifffile and write a plain
    contiguous BigTIFF.  Works for ScanImage files with per-frame metadata
    gaps that are not directly memmap-able.
    """
    with (
        tifffile.TiffFile(input_tif) as tf,
        tifffile.TiffWriter(str(output_tif), bigtiff=True) as tw,
    ):
        for page in tf.pages:
            tw.write(page.asarray(), contiguous=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flush_close(arr: np.ndarray) -> None:
    mm = getattr(arr, "_mmap", None)
    if mm is not None:
        mm.flush()
        mm.close()


# ---------------------------------------------------------------------------
# Flanking frame search
# ---------------------------------------------------------------------------


def _find_flanks(
    bad_ttl: int,
    plane_ids: np.ndarray,
    bad_set: set,
    n_ttl: int,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Return the nearest good TTL frame index before and after ``bad_ttl``
    within the same plane.  Walks the full movie if needed.
    Returns ``None`` for a side if no good frame exists there at all.
    """
    p = plane_ids[bad_ttl]
    flank_before: Optional[int] = None
    flank_after: Optional[int] = None

    for t in range(bad_ttl - 1, -1, -1):
        if plane_ids[t] == p and t not in bad_set:
            flank_before = t
            break

    for t in range(bad_ttl + 1, n_ttl):
        if plane_ids[t] == p and t not in bad_set:
            flank_after = t
            break

    return flank_before, flank_after


# ---------------------------------------------------------------------------
# Region detection (timing -> per-frame fractions)
# ---------------------------------------------------------------------------


def _artefact_regions(
    df_frames: pd.DataFrame, df_stims: pd.DataFrame
) -> pd.DataFrame:
    df_frames = df_frames.sort_values("start").reset_index(drop=True)
    df_stims = df_stims.sort_values("start")

    if df_stims.empty:
        return pd.DataFrame(columns=["frame", "frac_start", "frac_stop"])

    t0 = df_frames["start"].iloc[0]
    t1 = df_frames["stop"].iloc[-1]
    df_stims = df_stims[(df_stims["stop"] > t0) & (df_stims["start"] < t1)]
    if df_stims.empty:
        return pd.DataFrame(columns=["frame", "frac_start", "frac_stop"])

    all_bounds = np.empty(2 * len(df_frames), dtype=float)
    all_bounds[0::2] = df_frames["start"].to_numpy()
    all_bounds[1::2] = df_frames["stop"].to_numpy()

    frame_start, frac_start = _map_times_to_frame_frac(
        times=df_stims["start"].to_numpy(),
        frame_boundaries=df_frames["stop"].to_numpy(),
        all_boundaries=all_bounds,
        fill=0.0,
        offset=1,
    )
    frame_stop, frac_stop = _map_times_to_frame_frac(
        times=df_stims["stop"].to_numpy(),
        frame_boundaries=df_frames["start"].to_numpy(),
        all_boundaries=all_bounds,
        fill=1.0,
        offset=0,
    )

    df = pd.DataFrame(
        {
            "frame_start": frame_start,
            "frac_start": frac_start,
            "frame_stop": frame_stop,
            "frac_stop": frac_stop,
        },
        index=df_stims.index,
    )
    return _split_multi_frame_stims(df)


def _map_times_to_frame_frac(
    times: np.ndarray,
    frame_boundaries: np.ndarray,
    all_boundaries: np.ndarray,
    fill: float,
    offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    frame = (
        np.interp(
            times,
            frame_boundaries,
            np.arange(len(frame_boundaries)),
            left=-offset,
        )
        + offset
    )
    frame = frame.astype(int)

    all_idx = np.interp(times, all_boundaries, np.arange(len(all_boundaries)))
    out_of_frame = (all_idx.astype(int) % 2) == 1

    frac_template = np.tile([0.0, 1.0], len(frame_boundaries))
    frac = np.interp(times, all_boundaries, frac_template)
    frac[out_of_frame] = float(fill)

    return frame, np.clip(frac, 0.0, 1.0)


def _split_multi_frame_stims(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in df.itertuples():
        if r.frame_start == r.frame_stop:
            out.append(
                (int(r.frame_start), float(r.frac_start), float(r.frac_stop))
            )
            continue
        out.append((int(r.frame_start), float(r.frac_start), 1.0))
        for f in range(int(r.frame_start) + 1, int(r.frame_stop)):
            out.append((f, 0.0, 1.0))
        out.append((int(r.frame_stop), 0.0, float(r.frac_stop)))

    return pd.DataFrame(out, columns=["frame", "frac_start", "frac_stop"])


# ---------------------------------------------------------------------------
# Mask building (fractions -> scanlines)
# ---------------------------------------------------------------------------


def _build_bad_line_mask(
    df_split: pd.DataFrame,
    T: int,
    Y: int,
    pad_rows: int = 0,
) -> np.ndarray:
    bad = np.zeros((T, Y), dtype=bool)
    for r in df_split.itertuples(index=False):
        t = int(r.frame)
        if t < 0 or t >= T:
            continue
        y0 = max(0, int(np.floor(float(r.frac_start) * Y)) - pad_rows)
        y1 = min(Y, int(np.ceil(float(r.frac_stop) * Y)) + pad_rows)
        if y1 > y0:
            bad[t, y0:y1] = True
    return bad


# ---------------------------------------------------------------------------
# Nearest-neighbor / linear 1D interpolation
# ---------------------------------------------------------------------------


def _interp_block_numpy(
    block: np.ndarray,
    x: np.ndarray,
    donor_mask: np.ndarray,
    require_n_good: int,
) -> None:
    """In-place linear interpolation of NaNs along axis 0 using np.interp."""
    N, P = block.shape
    for j in range(P):
        y = block[:, j]
        nans = np.isnan(y)
        if not nans.any():
            continue
        good = donor_mask & ~nans
        if good.sum() < require_n_good:
            continue
        x_good = x[good]
        y_good = y[good]
        order = np.argsort(x_good)
        y[nans] = np.interp(x[nans], x_good[order], y_good[order])


def interpolate_nan(
    movie_float: np.ndarray,
    frame_index: np.ndarray,
    donor_mask: np.ndarray,
    require_n_good: int = 2,
    num_channel: int = 1,
    frame_gap: Optional[int] = None,
) -> np.ndarray:
    """
    Full-movie NaN fill using np.interp.

    Retained as the reference implementation.  Useful for validation and for
    callers that already hold the movie in RAM.
    """
    T, Y, X = movie_float.shape
    n_ttl = len(frame_index)

    if T != n_ttl * num_channel:
        raise ValueError(
            f"T ({T}) must equal len(frame_index)*num_channel "
            f"({n_ttl * num_channel})"
        )

    x = frame_index.astype(np.float32)
    flat = movie_float.reshape(T, -1)

    if frame_gap is None:
        num_planes = 1
        plane_ids = np.zeros(n_ttl, dtype=np.int32)
    else:
        num_planes = int(frame_gap) + 1
        plane_ids = (np.arange(n_ttl) % num_planes).astype(np.int32)

    for c in range(num_channel):
        ys = flat[c::num_channel, :]
        if num_planes == 1:
            _interp_block_numpy(ys, x, donor_mask, require_n_good)
            continue
        for p in range(num_planes):
            idx = np.where(plane_ids == p)[0]
            if len(idx) == 0:
                continue
            block = ys[idx, :]
            _interp_block_numpy(block, x[idx], donor_mask[idx], require_n_good)
            ys[idx, :] = block
        flat[c::num_channel, :] = ys

    return flat.reshape(T, Y, X)
