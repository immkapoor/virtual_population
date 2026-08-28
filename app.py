#!/usr/bin/env python3
"""
Streamlit GUI for multivariate RNN virtual seal trajectories.

The multivariate checkpoint directly conditions neural prediction on
physical traits when those columns were used in training.

Virtual mode:
    selected sex + mass + length
    selected haulout
    generated history
    nearest observed dynamic context at generated position
    autoregressive RNN rollout
    land + empirical movement constraints

Evaluation is intentionally separate from virtual generation.
"""

import os
import math
import importlib.util

import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go


st.set_page_config(
    page_title="Multivariate Virtual Seal Generator",
    page_icon="🦭",
    layout="wide",
)

st.title(
    "Multivariate RNN Virtual Seal Generator"
)

st.caption(
    "Physical traits directly condition the RNN; dynamic environmental/social "
    "context is updated during autoregressive generation."
)


# ============================================================
# Import training/backend module
# ============================================================

def import_backend(
    path,
):
    spec = importlib.util.spec_from_file_location(
        "multi_rnn_backend",
        os.path.abspath(
            path
        ),
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def import_univariate_backend(
    path,
):
    spec = importlib.util.spec_from_file_location(
        "univariate_geo_backend",
        os.path.abspath(
            path
        ),
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def quantile(
    x,
    p,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    x = x[
        np.isfinite(
            x
        )
    ]

    return float(
        np.quantile(
            x,
            p,
        )
    )


# ============================================================
# Checkpoint
# ============================================================

def load_checkpoint(
    backend,
    pt_file,
    device_choice,
):
    ckpt = torch.load(
        pt_file,
        map_location="cpu",
        weights_only=False,
    )

    required = [
        "model",
        "window",
        "input_dim",
        "hidden_dim",
        "input_cols",
        "dynamic_cols",
        "static_cols",
        "x_mean",
        "x_std",
        "y_mean",
        "y_std",
    ]

    missing = [
        x
        for x in required
        if x not in ckpt
    ]

    if missing:
        raise KeyError(
            "Multivariate checkpoint missing: "
            + ", ".join(
                missing
            )
        )

    model = backend.MultivariateRNN(
        input_dim=int(
            ckpt[
                "input_dim"
            ]
        ),
        hidden_dim=int(
            ckpt[
                "hidden_dim"
            ]
        ),
        num_layers=int(
            ckpt.get(
                "num_layers",
                1,
            )
        ),
        dropout=float(
            ckpt.get(
                "dropout",
                0.0,
            )
        ),
    )

    model.load_state_dict(
        ckpt[
            "model"
        ]
    )

    device = torch.device(
        "cuda"
        if (
            device_choice == "CUDA"
            and torch.cuda.is_available()
        )
        else "cpu"
    )

    model = model.to(
        device
    ).eval()

    xs = backend.Standardizer()
    xs.mean = np.asarray(
        ckpt[
            "x_mean"
        ],
        dtype=float,
    )
    xs.std = np.asarray(
        ckpt[
            "x_std"
        ],
        dtype=float,
    )

    ys = backend.Standardizer()
    ys.mean = np.asarray(
        ckpt[
            "y_mean"
        ],
        dtype=float,
    )
    ys.std = np.asarray(
        ckpt[
            "y_std"
        ],
        dtype=float,
    )

    return (
        ckpt,
        model,
        xs,
        ys,
        device,
    )


# ============================================================
# Dynamic-context lookup
# ============================================================

class ContextLookup:
    """
    Approximate spatial context for autonomous generation.

    At each generated position, use the median dynamic features of the
    K nearest observed rows in lat/lon space. This avoids replaying an
    observed trajectory while providing location-dependent exogenous inputs.
    """

    def __init__(
        self,
        df,
        dynamic_cols,
        max_rows=100000,
        seed=42,
    ):
        use = df[
            [
                "lat",
                "lon",
            ]
            + dynamic_cols
        ].copy()

        use = use.dropna(
            subset=[
                "lat",
                "lon",
            ]
        )

        if len(
            use
        ) > max_rows:
            use = use.sample(
                max_rows,
                random_state=seed,
            )

        self.lat = use[
            "lat"
        ].to_numpy(
            dtype=float
        )

        self.lon = use[
            "lon"
        ].to_numpy(
            dtype=float
        )

        self.dynamic_cols = list(
            dynamic_cols
        )

        self.values = (
            use[
                self.dynamic_cols
            ].to_numpy(
                dtype=float
            )
            if self.dynamic_cols
            else np.empty(
                (
                    len(
                        use
                    ),
                    0,
                )
            )
        )

        self.cos_ref = math.cos(
            math.radians(
                float(
                    np.nanmedian(
                        self.lat
                    )
                )
            )
        )

    def query(
        self,
        lat,
        lon,
        k=20,
    ):
        if not self.dynamic_cols:
            return np.empty(
                0,
                dtype=float,
            )

        d2 = (
            (
                self.lat
                - float(
                    lat
                )
            )
            ** 2
            + (
                (
                    self.lon
                    - float(
                        lon
                    )
                )
                * self.cos_ref
            )
            ** 2
        )

        k = min(
            int(
                k
            ),
            len(
                d2
            ),
        )

        idx = np.argpartition(
            d2,
            k - 1,
        )[
            :k
        ]

        vals = self.values[
            idx
        ]

        return np.nanmedian(
            vals,
            axis=0,
        )


# ============================================================
# Empirical movement priors
# ============================================================

def extract_movements(
    geo,
    df,
    max_gap,
):
    steps = []
    turns = []
    trip_steps = []
    trip_dist = []

    for _, g in df.groupby(
        "_trip"
    ):
        g = g.sort_values(
            "d_date"
        )

        s = []

        for i in range(
            1,
            len(
                g
            ),
        ):
            dt = (
                g[
                    "d_date"
                ].iloc[
                    i
                ]
                - g[
                    "d_date"
                ].iloc[
                    i - 1
                ]
            ).total_seconds() / 3600.0

            if (
                dt <= 0
                or dt
                > max_gap
            ):
                continue

            s.append(
                geo.ll_to_ne(
                    g[
                        "lat"
                    ].iloc[
                        i - 1
                    ],
                    g[
                        "lon"
                    ].iloc[
                        i - 1
                    ],
                    g[
                        "lat"
                    ].iloc[
                        i
                    ],
                    g[
                        "lon"
                    ].iloc[
                        i
                    ],
                )
            )

        if len(
            s
        ) < 2:
            continue

        s = np.asarray(
            s,
            dtype=float,
        )

        lengths = np.linalg.norm(
            s,
            axis=1,
        )

        steps.extend(
            s.tolist()
        )

        turns.extend(
            [
                geo.signed_turn(
                    s[
                        j - 1
                    ],
                    s[
                        j
                    ],
                )
                for j in range(
                    1,
                    len(
                        s
                    ),
                )
            ]
        )

        trip_steps.append(
            len(
                s
            )
        )

        trip_dist.append(
            float(
                lengths.sum()
            )
        )

    return {
        "steps": np.asarray(
            steps,
            dtype=float,
        ),
        "step_lengths": np.linalg.norm(
            np.asarray(
                steps,
                dtype=float,
            ),
            axis=1,
        ),
        "turns": np.asarray(
            turns,
            dtype=float,
        ),
        "trip_steps": np.asarray(
            trip_steps,
            dtype=int,
        ),
        "trip_distance": np.asarray(
            trip_dist,
            dtype=float,
        ),
    }


class Constraints:
    def __init__(
        self,
        geo,
        priors,
        step_low,
        step_high,
        turn_low,
        turn_high,
    ):
        self.geo = geo

        self.step_samples = priors[
            "step_lengths"
        ]

        self.turn_samples = priors[
            "turns"
        ]

        self.step_lo = float(
            step_low
        )

        self.step_hi = float(
            step_high
        )

        self.step_med = float(
            np.median(
                self.step_samples
            )
        )

        self.turn_lo = float(
            turn_low
        )

        self.turn_hi = float(
            turn_high
        )

    def constrain(
        self,
        prev,
        prop,
    ):
        v = np.asarray(
            prop,
            dtype=float,
        )

        L = np.linalg.norm(
            v
        )

        if (
            not np.isfinite(
                L
            )
            or L < 1e-10
        ):
            v = self.geo.rescale(
                prev,
                self.step_med,
            )

            L = self.step_med

        L = float(
            np.clip(
                L,
                self.step_lo,
                self.step_hi,
            )
        )

        v = self.geo.rescale(
            v,
            L,
        )

        turn = self.geo.signed_turn(
            prev,
            v,
        )

        turn2 = float(
            np.clip(
                turn,
                self.turn_lo,
                self.turn_hi,
            )
        )

        if abs(
            turn
            - turn2
        ) > 1e-12:
            v = self.geo.rescale(
                self.geo.rotate(
                    prev,
                    turn2,
                ),
                L,
            )

        return v


# ============================================================
# Land plot
# ============================================================

def add_land(
    fig,
    land,
    lon,
    lat,
):
    if (
        land is None
        or not getattr(
            land,
            "available",
            False,
        )
    ):
        return

    lon = np.asarray(
        lon,
        dtype=float,
    )

    lat = np.asarray(
        lat,
        dtype=float,
    )

    xmin = float(
        np.nanmin(
            lon
        )
    )

    xmax = float(
        np.nanmax(
            lon
        )
    )

    ymin = float(
        np.nanmin(
            lat
        )
    )

    ymax = float(
        np.nanmax(
            lat
        )
    )

    dx = max(
        0.1,
        (
            xmax
            - xmin
        )
        * 0.2,
    )

    dy = max(
        0.1,
        (
            ymax
            - ymin
        )
        * 0.2,
    )

    idxs = list(
        land.sindex.intersection(
            (
                xmin - dx,
                ymin - dy,
                xmax + dx,
                ymax + dy,
            )
        )
    )

    xs = []
    ys = []

    def ring(
        coords,
    ):
        for x, y in coords:
            xs.append(
                float(
                    x
                )
            )

            ys.append(
                float(
                    y
                )
            )

        xs.append(
            None
        )

        ys.append(
            None
        )

    for idx in idxs:
        geom = land.land.geometry.iloc[
            idx
        ]

        try:
            if geom.geom_type == "Polygon":
                ring(
                    geom.exterior.coords
                )

            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    ring(
                        poly.exterior.coords
                    )
        except Exception:
            pass

    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                fillcolor="rgba(215,215,215,0.70)",
                line=dict(
                    color="rgba(90,90,90,0.9)",
                    width=0.7,
                ),
                name="Land",
                hoverinfo="skip",
            )
        )


def plot_virtual(
    df,
    haulout,
    land,
):
    fig = go.Figure()

    add_land(
        fig,
        land,
        np.concatenate(
            [
                df[
                    "lon"
                ].to_numpy(
                    float
                ),
                [
                    haulout[
                        "lon"
                    ]
                ],
            ]
        ),
        np.concatenate(
            [
                df[
                    "lat"
                ].to_numpy(
                    float
                ),
                [
                    haulout[
                        "lat"
                    ]
                ],
            ]
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=df[
                "lon"
            ],
            y=df[
                "lat"
            ],
            mode="lines",
            name="Virtual trajectory",
            line=dict(
                color="#243B53",
                width=3,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[
                haulout[
                    "lon"
                ]
            ],
            y=[
                haulout[
                    "lat"
                ]
            ],
            mode="markers",
            name="Haulout",
            marker=dict(
                symbol="square",
                size=12,
                color="#243B53",
            ),
        )
    )

    fig.update_layout(
        height=650,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        legend=dict(
            orientation="h",
        ),
        plot_bgcolor="white",
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


# ============================================================
# Autoregressive generation
# ============================================================

def make_feature_vector(
    move,
    dynamic,
    static_values,
):
    return np.concatenate(
        [
            np.asarray(
                move,
                dtype=float,
            ),
            np.asarray(
                dynamic,
                dtype=float,
            ),
            np.asarray(
                static_values,
                dtype=float,
            ),
        ]
    )


@torch.no_grad()
def generate(
    backend,
    geo,
    model,
    x_scaler,
    y_scaler,
    context,
    dynamic_cols,
    static_cols,
    static_map,
    constraints,
    haulout,
    land,
    window,
    rollout_steps,
    max_distance,
    device,
    seed,
    context_k,
):
    rng = np.random.default_rng(
        int(
            seed
        )
    )

    coords, init_moves = geo.synthetic_departure(
        float(
            haulout[
                "lat"
            ]
        ),
        float(
            haulout[
                "lon"
            ]
        ),
        int(
            window
        ),
        constraints,
        land,
        rng,
    )

    history_moves = [
        x.copy()
        for x in init_moves
    ]

    history_features = []

    for i, move in enumerate(
        history_moves
    ):
        lat, lon = coords[
            i + 1
        ]

        dyn = context.query(
            lat,
            lon,
            k=context_k,
        )

        stat = [
            static_map[
                c
            ]
            for c in static_cols
        ]

        history_features.append(
            make_feature_vector(
                move,
                dyn,
                stat,
            )
        )

    total_distance = float(
        np.linalg.norm(
            init_moves,
            axis=1,
        ).sum()
    )

    interventions = 0

    while len(
        history_moves
    ) < rollout_steps:
        x = np.asarray(
            history_features[
                -window:
            ],
            dtype=float,
        )

        x_scaled = x_scaler.transform(
            x
        ).astype(
            np.float32
        )

        pred_scaled = model(
            torch.tensor(
                x_scaled[
                    None,
                    :,
                    :
                ],
                dtype=torch.float32,
                device=device,
            )
        ).cpu().numpy()

        proposed = y_scaler.inverse(
            pred_scaled
        )[
            0
        ]

        prev = history_moves[
            -1
        ]

        lat, lon = coords[
            -1
        ]

        (
            nlat,
            nlon,
            move,
            changed,
        ) = geo.repair_move(
            float(
                lat
            ),
            float(
                lon
            ),
            prev,
            proposed,
            constraints,
            land,
            False,
        )

        interventions += int(
            changed
        )

        d = float(
            np.linalg.norm(
                move
            )
        )

        if d < 1e-10:
            fallback = geo.rescale(
                geo.rotate(
                    prev,
                    float(
                        rng.choice(
                            constraints.turn_samples
                        )
                    ),
                ),
                float(
                    rng.choice(
                        constraints.step_samples
                    )
                ),
            )

            (
                nlat,
                nlon,
                move,
                changed2,
            ) = geo.repair_move(
                float(
                    lat
                ),
                float(
                    lon
                ),
                prev,
                fallback,
                constraints,
                land,
                False,
            )

            interventions += int(
                changed2
            )

            d = float(
                np.linalg.norm(
                    move
                )
            )

        if d < 1e-10:
            break

        if (
            max_distance is not None
            and total_distance
            + d
            > max_distance
        ):
            break

        coords = np.vstack(
            [
                coords,
                [
                    nlat,
                    nlon,
                ],
            ]
        )

        history_moves.append(
            move
        )

        dyn = context.query(
            nlat,
            nlon,
            k=context_k,
        )

        stat = [
            static_map[
                c
            ]
            for c in static_cols
        ]

        history_features.append(
            make_feature_vector(
                move,
                dyn,
                stat,
            )
        )

        total_distance += d

    df = pd.DataFrame(
        {
            "point_index": np.arange(
                len(
                    coords
                )
            ),
            "lat": coords[
                :,
                0,
            ],
            "lon": coords[
                :,
                1,
            ],
        }
    )

    return (
        df,
        np.asarray(
            history_moves
        ),
        interventions,
        total_distance,
    )


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header(
        "Model and data"
    )

    multi_backend_file = st.text_input(
        "Python file",
        "app.py",
    )

    geo_backend_file = st.text_input(
        "Geographic constraint backend",
        "univariate_rnn_virtual_trips.py",
    )

    checkpoint_file = st.text_input(
        "Multivariate .pt",
        "multivariate_rnn_model.pt",
    )

    trip_file = st.text_input(
        "Trip/context CSV",
        "social_cues_2010_13_haulout_trips.csv",
    )

    haulout_file = st.text_input(
        "Haulout CSV",
        "Clean_IRO_HAULOUT_2010-13_spatial.csv",
    )

    land_file = st.text_input(
        "Land shapefile",
        "./Land_polygons/highres_map.shp",
    )

    device_choice = st.selectbox(
        "Device",
        [
            "CUDA",
            "CPU",
        ],
    )

    load = st.button(
        "Load multivariate model",
        type="primary",
        use_container_width=True,
    )


if load:
    try:
        with st.spinner(
            "Loading..."
        ):
            backend = import_backend(
                multi_backend_file
            )

            geo = import_univariate_backend(
                geo_backend_file
            )

            (
                ckpt,
                model,
                xs,
                ys,
                device,
            ) = load_checkpoint(
                backend,
                checkpoint_file,
                device_choice,
            )

            df = backend.load_trips(
                trip_file,
                gap_hours=24.0,
            )

            for c in (
                ckpt[
                    "dynamic_cols"
                ]
                + ckpt[
                    "static_cols"
                ]
            ):
                if c in df.columns:
                    df[c] = pd.to_numeric(
                        df[c],
                        errors="coerce",
                    )

                    fallback = float(
                        ckpt.get(
                            "feature_medians",
                            {},
                        ).get(
                            c,
                            0.0,
                        )
                    )

                    df[c] = df[c].fillna(
                        fallback
                    )

            context = ContextLookup(
                df,
                ckpt[
                    "dynamic_cols"
                ],
            )

            priors = extract_movements(
                geo,
                df,
                max_gap=24.0,
            )

            haulouts = geo.load_haulouts(
                haulout_file
            )

            land = (
                geo.LandMask(
                    land_file
                )
                if land_file.strip()
                else None
            )

            st.session_state[
                "multi_assets"
            ] = {
                "backend": backend,
                "geo": geo,
                "ckpt": ckpt,
                "model": model,
                "xs": xs,
                "ys": ys,
                "device": device,
                "df": df,
                "context": context,
                "priors": priors,
                "haulouts": haulouts,
                "land": land,
            }

        st.success(
            "Loaded."
        )

    except Exception as exc:
        st.exception(
            exc
        )


if "multi_assets" not in st.session_state:
    st.info(
        "Load the trained multivariate checkpoint from the sidebar."
    )
    st.stop()


A = st.session_state[
    "multi_assets"
]

backend = A[
    "backend"
]

geo = A[
    "geo"
]

ckpt = A[
    "ckpt"
]

model = A[
    "model"
]

xs = A[
    "xs"
]

ys = A[
    "ys"
]

device = A[
    "device"
]

df = A[
    "df"
]

context = A[
    "context"
]

priors = A[
    "priors"
]

haulouts = A[
    "haulouts"
]

land = A[
    "land"
]


# ============================================================
# Tabs
# ============================================================

virtual_tab, eval_tab = st.tabs(
    [
        "Virtual generation",
        "Sliding-window evaluation",
    ]
)


with virtual_tab:
    st.subheader(
        "Virtual physical profile"
    )

    static_cols = ckpt[
        "static_cols"
    ]

    p1, p2, p3 = st.columns(
        3
    )

    static_map = {}

    with p1:
        if "sex_code" in static_cols:
            sex = st.selectbox(
                "Sex",
                [
                    "Female",
                    "Male",
                ],
            )

            static_map[
                "sex_code"
            ] = (
                0.0
                if sex == "Female"
                else 1.0
            )

        else:
            st.info(
                "sex_code was not used during training."
            )

    with p2:
        if "body_mass" in static_cols:
            vals = pd.to_numeric(
                df[
                    "body_mass"
                ],
                errors="coerce",
            ).dropna()

            static_map[
                "body_mass"
            ] = st.slider(
                "Body mass",
                float(
                    vals.min()
                ),
                float(
                    vals.max()
                ),
                float(
                    vals.median()
                ),
            )

        else:
            st.info(
                "body_mass was not used during training."
            )

    with p3:
        if "total_length" in static_cols:
            vals = pd.to_numeric(
                df[
                    "total_length"
                ],
                errors="coerce",
            ).dropna()

            static_map[
                "total_length"
            ] = st.slider(
                "Total length",
                float(
                    vals.min()
                ),
                float(
                    vals.max()
                ),
                float(
                    vals.median()
                ),
            )

        else:
            st.info(
                "total_length was not used during training."
            )

    for c in static_cols:
        if c not in static_map:
            static_map[
                c
            ] = float(
                ckpt.get(
                    "feature_medians",
                    {},
                ).get(
                    c,
                    0.0,
                )
            )

    st.subheader(
        "Initial haulout"
    )

    labels = [
        (
            f"H{i:04d} | "
            f"{float(r['lat']):.5f}, "
            f"{float(r['lon']):.5f}"
        )
        for i, r in haulouts.iterrows()
    ]

    chosen_label = st.selectbox(
        "Haulout location",
        labels,
    )

    hidx = labels.index(
        chosen_label
    )

    haulout = haulouts.iloc[
        hidx
    ]

    st.subheader(
        "History and rollout"
    )

    c1, c2 = st.columns(
        2
    )

    trained_window = int(
        ckpt[
            "window"
        ]
    )

    with c1:
        window = st.slider(
            "Sliding/history window",
            3,
            max(
                100,
                trained_window
                * 4,
            ),
            trained_window,
        )

        if window != trained_window:
            st.warning(
                f"Checkpoint trained with window={trained_window}."
            )

    with c2:
        rollout_steps = st.slider(
            "Autoregressive rollout steps",
            window,
            1000,
            max(
                window,
                200,
            ),
        )

    st.subheader(
        "Trip-derived constraints"
    )

    c1, c2 = st.columns(
        2
    )

    with c1:
        step_q = st.slider(
            "Step-length percentile interval",
            0.0,
            100.0,
            (
                0.5,
                99.5,
            ),
            0.5,
        )

        step_low = quantile(
            priors[
                "step_lengths"
            ],
            step_q[
                0
            ]
            / 100.0,
        )

        step_high = quantile(
            priors[
                "step_lengths"
            ],
            step_q[
                1
            ]
            / 100.0,
        )

        st.caption(
            f"{step_low:.4f}–{step_high:.4f} km"
        )

        turn_deg = np.degrees(
            priors[
                "turns"
            ]
        )

        turn_q = st.slider(
            "Turning-angle percentile interval",
            0.0,
            100.0,
            (
                0.5,
                99.5,
            ),
            0.5,
        )

        turn_low = quantile(
            turn_deg,
            turn_q[
                0
            ]
            / 100.0,
        )

        turn_high = quantile(
            turn_deg,
            turn_q[
                1
            ]
            / 100.0,
        )

        st.caption(
            f"{turn_low:.1f}°–{turn_high:.1f}°"
        )

    with c2:
        distance_q = st.slider(
            "Maximum total trip-distance percentile",
            50.0,
            100.0,
            99.0,
            1.0,
        )

        max_distance = quantile(
            priors[
                "trip_distance"
            ],
            distance_q
            / 100.0,
        )

        st.caption(
            f"{max_distance:.2f} km"
        )

        context_k = st.slider(
            "Nearest observations for dynamic context",
            1,
            100,
            20,
        )

    constraints = Constraints(
        geo,
        priors,
        step_low,
        step_high,
        math.radians(
            turn_low
        ),
        math.radians(
            turn_high
        ),
    )

    seed = st.number_input(
        "Random seed",
        0,
        2_000_000_000,
        42,
    )

    if st.button(
        "Generate multivariate virtual trajectory",
        type="primary",
    ):
        try:
            result = generate(
                backend,
                geo,
                model,
                xs,
                ys,
                context,
                ckpt[
                    "dynamic_cols"
                ],
                ckpt[
                    "static_cols"
                ],
                static_map,
                constraints,
                haulout,
                land,
                window,
                rollout_steps,
                max_distance,
                device,
                int(
                    seed
                ),
                context_k,
            )

            st.session_state[
                "multi_result"
            ] = result

        except Exception as exc:
            st.exception(
                exc
            )

    if "multi_result" in st.session_state:
        traj, moves, interventions, total = st.session_state[
            "multi_result"
        ]

        st.plotly_chart(
            plot_virtual(
                traj,
                {
                    "lat": float(
                        haulout[
                            "lat"
                        ]
                    ),
                    "lon": float(
                        haulout[
                            "lon"
                        ]
                    ),
                },
                land,
            ),
            use_container_width=True,
        )

        a, b, c = st.columns(
            3
        )

        a.metric(
            "Generated steps",
            len(
                moves
            ),
        )

        b.metric(
            "Total distance",
            f"{total:.2f} km",
        )

        c.metric(
            "Constraint intervention",
            f"{100 * interventions / max(1, len(moves)):.1f}%",
        )

        for k, v in static_map.items():
            traj[
                k
            ] = v

        traj[
            "haulout_lat"
        ] = float(
            haulout[
                "lat"
            ]
        )

        traj[
            "haulout_lon"
        ] = float(
            haulout[
                "lon"
            ]
        )

        st.download_button(
            "Download virtual trajectory CSV",
            traj.to_csv(
                index=False
            ).encode(
                "utf-8"
            ),
            "multivariate_virtual_trajectory.csv",
            "text/csv",
        )


with eval_tab:
    st.markdown(
        """
This tab is reserved for **observed-vs-predicted sliding-window evaluation**.
It is intentionally separate from virtual generation so observed trajectories
are never shown as part of a virtual rollout.
"""
    )

    st.info(
        "The virtual-generation workflow above does not use an observed trajectory path."
    )
