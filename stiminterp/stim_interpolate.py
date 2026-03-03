"""
stim_interpolate.py

Photostimulation artefact removal via 1D interpolation.

Pipeline
--------
1) Use frame timing + stim timing to determine artefact regions
   (frame index + fraction within frame).
2) Convert fractions -> scanline rows (Y).
3) Set contaminated pixels to NaN.
4) Fill NaNs per pixel using nearest-neighbor in *frame_index* space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class StimInterpConfig:
    """Configuration for stim artefact removal."""

    pad_rows: int = 1
    require_n_good: int = 1  # nearest fill needs only 1 neighbor


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def remove_photostim_artefacts(
    movie: np.ndarray,
    df_frames: pd.DataFrame,
    df_stims: pd.DataFrame,
    frame_gap: Optional[int] = None,
    num_channel: Optional[int] = None,
    cfg: Optional[StimInterpConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Remove photostimulation artefacts by nearest-neighbor temporal filling.

    Parameters
    ----------
    movie : np.ndarray
        Imaging movie, shape (T, Y, X).
        If multiple channels are saved interleaved per TTL frame, then:
            T = len(df_frames) * num_channel
    df_frames : pd.DataFrame
        Columns ["start", "stop"] per TTL frame.
        NOTE: TTL frames should already include plane interleaving if present.
    df_stims : pd.DataFrame
        Columns ["start", "stop"] per stim interval.
    frame_gap : int, optional
        Plane interleave indicator:
          - 1 => 2 planes interleaved (plane_id = ttl_frame % 2)
          - 2 => 3 planes interleaved (plane_id = ttl_frame % 3)
        If None, treat as single plane for interpolation grouping i.e, 0.
    num_channel : int, optional
        Number of interleaved channels in the TIFF (channel-fast ordering).
        If None, inferred as movie.shape[0] // len(df_frames).
    cfg : StimInterpConfig, optional

    Returns
    -------
    corrected : np.ndarray
        Corrected movie (float32).
    bad_mask : np.ndarray
        Boolean mask (T, Y, X) indicating pixels replaced.
    df_split : pd.DataFrame
        Per-TTL-frame stim segments: ["frame", "frac_start", "frac_stop"].
    """
    if cfg is None:
        cfg = StimInterpConfig()

    movie = np.asarray(movie)
    if movie.ndim != 3:
        raise ValueError("movie must have shape (T, Y, X)")
    T, Y, X = movie.shape

    n_ttl = len(df_frames)
    if n_ttl <= 0:
        raise ValueError("df_frames is empty")

    if T % n_ttl != 0:
        raise ValueError(
            f"movie.shape[0]={T} must be a multiple of len(df_frames)={n_ttl}."
        )

    inferred_num_channel = T // n_ttl
    nchan = (
        int(num_channel)
        if num_channel is not None
        else int(inferred_num_channel)
    )

    if nchan <= 0 or (n_ttl * nchan) != T:
        raise ValueError(
            f"movie.shape[0]={T} must be a multiple of len(df_frames)={n_ttl}."
        )

    # --- stim regions in TTL frame space ---
    df_split = _artefact_regions(df_frames, df_stims)
    if df_split.empty:
        return (
            movie.astype(np.float32, copy=True),
            np.zeros_like(movie, dtype=bool),
            df_split,
        )

    # --- bad scanlines in TTL frame space: (n_ttl, Y) ---
    bad_lines_ttl = _build_bad_line_mask(
        df_split=df_split,
        T=n_ttl,
        Y=Y,
        pad_rows=cfg.pad_rows,
    )

    # --- expand across channels to movie time axis: (T, Y) ---
    # TTL frame t corresponds to movie indices t*nchan + c for each channel c
    bad_lines = np.repeat(bad_lines_ttl, repeats=nchan, axis=0)
    bad_mask = np.repeat(bad_lines[:, :, None], X, axis=2)

    corrected = movie.astype(np.float32, copy=True)
    corrected[bad_mask] = np.nan

    frame_index = np.arange(n_ttl, dtype=np.int32)
    donor_mask = np.ones(n_ttl, dtype=bool)

    corrected = interpolate_nan(
        corrected,
        frame_index=frame_index,
        donor_mask=donor_mask,
        require_n_good=cfg.require_n_good,
        num_channel=nchan,
        frame_gap=frame_gap,
    )

    return corrected, bad_mask, df_split


# -----------------------------------------------------------------------------
# Region detection (timing -> per-frame fractions)
# -----------------------------------------------------------------------------


def _artefact_regions(
    df_frames: pd.DataFrame, df_stims: pd.DataFrame
) -> pd.DataFrame:
    df_frames = df_frames.sort_values("start").reset_index(drop=True)
    df_stims = df_stims.sort_values("start")

    if df_stims.empty:
        return pd.DataFrame(columns=["frame", "frac_start", "frac_stop"])

    # Remove stims outside acquisition span
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


# -----------------------------------------------------------------------------
# Mask building (fractions -> scanlines)
# -----------------------------------------------------------------------------


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

        y0 = int(np.floor(float(r.frac_start) * Y))
        y1 = int(np.ceil(float(r.frac_stop) * Y))

        y0 = max(0, y0 - pad_rows)
        y1 = min(Y, y1 + pad_rows)

        if y1 > y0:
            bad[t, y0:y1] = True

    return bad


# -----------------------------------------------------------------------------
# Nearest-neighbor 1D interpolation
# -----------------------------------------------------------------------------


def interpolate_nan(
    movie_float: np.ndarray,
    frame_index: np.ndarray,
    donor_mask: np.ndarray,
    require_n_good: int = 2,
    num_channel: int = 1,
    frame_gap: Optional[int] = None,
) -> np.ndarray:
    """
    Fill NaNs using np.interp
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

    # Plane grouping
    if frame_gap is None:
        num_planes = 1
        plane_ids = np.zeros(n_ttl, dtype=np.int32)
    else:
        num_planes = int(frame_gap) + 1
        plane_ids = np.arange(n_ttl) % num_planes

    for c in range(num_channel):
        ys = flat[c::num_channel, :]  # (n_ttl, n_pixels)

        if num_planes == 1:
            _interp_block_numpy(ys, x, donor_mask, require_n_good)
            continue

        for p in range(num_planes):
            idx = np.where(plane_ids == p)[0]
            if len(idx) == 0:
                continue

            block = ys[idx, :]
            _interp_block_numpy(
                block,
                x[idx],
                donor_mask[idx],
                require_n_good,
            )
            ys[idx, :] = block

        flat[c::num_channel, :] = ys

    return flat.reshape(T, Y, X)


def _interp_block_numpy(
    block: np.ndarray,
    x: np.ndarray,
    donor_mask: np.ndarray,
    require_n_good: int,
) -> None:
    """
    In-place linear interpolation using np.interp.
    """

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

        # NumPy requires sorted x
        # frame_index already sorted, but keep safe
        order = np.argsort(x_good)
        x_good = x_good[order]
        y_good = y_good[order]

        y[nans] = np.interp(x[nans], x_good, y_good)
