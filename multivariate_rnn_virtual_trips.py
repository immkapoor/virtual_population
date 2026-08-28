#!/usr/bin/env python3
"""
Multivariate RNN for seal trajectory prediction and virtual generation.

Input per timestep
------------------
[next movement context]
    north_km, east_km
[dynamic environmental/social variables]
    available columns from DEFAULT_DYNAMIC_COLS
[static physical variables repeated over the window]
    sex_code, body_mass, total_length

Target
------
next north/east movement vector in km.

The checkpoint stores all feature metadata/scalers required by the GUI.

Example
-------
python multivariate_rnn_virtual_trips.py \
  --trip_csv social_cues_2010_13_haulout_trips.csv \
  --out_dir multivariate_rnn_2010_13 \
  --window 20 \
  --epochs 100 \
  --device cuda
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


KM_LAT = 110.574
KM_LON = 111.320

TRIP_IDS = [
    "trip_id",
    "trip_number",
    "trip",
    "haulout_number",
]

DEFAULT_DYNAMIC_COLS = [
    "sst",
    "salinity",
    "current_u",
    "current_v",
    "depth",
    "distance_to_shore_km",
    "nearby_seal_count",
    "nearest_seal_distance_km",
    "mean_nearby_distance_km",
    "same_sex_nearby_count",
    "opposite_sex_nearby_count",
    "nearby_male_count",
    "nearby_female_count",
    "social_density_score",
]

DEFAULT_STATIC_COLS = [
    "sex_code",
    "body_mass",
    "total_length",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ll_to_ne(lat1, lon1, lat2, lon2):
    mid = 0.5 * (float(lat1) + float(lat2))

    north = (
        float(lat2)
        - float(lat1)
    ) * KM_LAT

    east = (
        float(lon2)
        - float(lon1)
    ) * KM_LON * math.cos(
        math.radians(mid)
    )

    return np.array(
        [north, east],
        dtype=float,
    )


class Standardizer:
    def fit(self, x):
        x = np.asarray(
            x,
            dtype=float,
        )

        self.mean = np.nanmean(
            x,
            axis=0,
        )

        self.std = np.nanstd(
            x,
            axis=0,
        )

        self.std[
            self.std < 1e-8
        ] = 1.0

        return self

    def transform(self, x):
        return (
            np.asarray(
                x,
                dtype=float,
            )
            - self.mean
        ) / self.std

    def inverse(self, x):
        return (
            np.asarray(
                x,
                dtype=float,
            )
            * self.std
            + self.mean
        )


def load_trips(
    path,
    gap_hours=24.0,
):
    df = pd.read_csv(
        path
    )

    required = [
        "seal",
        "d_date",
        "lat",
        "lon",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df[
        "d_date"
    ] = pd.to_datetime(
        df[
            "d_date"
        ],
        errors="coerce",
        utc=True,
    )

    for c in [
        "lat",
        "lon",
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    df = df.dropna(
        subset=required
    ).copy()

    trip_col = next(
        (
            c
            for c in TRIP_IDS
            if c in df.columns
        ),
        None,
    )

    if trip_col:
        df[
            "_trip"
        ] = (
            df[
                "seal"
            ].astype(str)
            + "_"
            + df[
                trip_col
            ].astype(str)
        )

        print(
            "Trip ID column:",
            trip_col,
        )

    else:
        pieces = []

        for seal, g in (
            df.sort_values(
                [
                    "seal",
                    "d_date",
                ]
            )
            .groupby(
                "seal"
            )
        ):
            g = g.copy()

            gaps = (
                g[
                    "d_date"
                ]
                .diff()
                .dt.total_seconds()
                .div(3600.0)
            )

            block = (
                gaps
                .gt(
                    gap_hours
                )
                .fillna(
                    False
                )
                .cumsum()
            )

            g[
                "_trip"
            ] = (
                str(
                    seal
                )
                + "_"
                + block.astype(
                    str
                )
            )

            pieces.append(
                g
            )

        df = pd.concat(
            pieces,
            ignore_index=True,
        )

    return df.sort_values(
        [
            "_trip",
            "d_date",
        ]
    ).reset_index(
        drop=True
    )


def available_feature_columns(
    df,
):
    dynamic = [
        c
        for c in DEFAULT_DYNAMIC_COLS
        if c in df.columns
    ]

    static = [
        c
        for c in DEFAULT_STATIC_COLS
        if c in df.columns
    ]

    return (
        dynamic,
        static,
    )


def fill_numeric_features(
    df,
    columns,
):
    medians = {}

    for c in columns:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

        med = float(
            df[c].median()
        )

        if not np.isfinite(
            med
        ):
            med = 0.0

        medians[c] = med

        df[c] = df[c].fillna(
            med
        )

    return medians


def build_trip_sequences(
    df,
    dynamic_cols,
    static_cols,
    max_gap_hours,
    min_points,
):
    sequences = []

    for trip_id, g in df.groupby(
        "_trip"
    ):
        g = g.sort_values(
            "d_date"
        ).reset_index(
            drop=True
        )

        if len(
            g
        ) < min_points:
            continue

        rows = []

        for i in range(
            1,
            len(
                g
            ),
        ):
            dt_h = (
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
                not np.isfinite(
                    dt_h
                )
                or dt_h <= 0
                or dt_h
                > max_gap_hours
            ):
                continue

            move = ll_to_ne(
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

            dyn = (
                g.loc[
                    i,
                    dynamic_cols,
                ].to_numpy(
                    dtype=float
                )
                if dynamic_cols
                else np.empty(
                    0,
                    dtype=float,
                )
            )

            stat = (
                g.loc[
                    i,
                    static_cols,
                ].to_numpy(
                    dtype=float
                )
                if static_cols
                else np.empty(
                    0,
                    dtype=float,
                )
            )

            x = np.concatenate(
                [
                    move,
                    dyn,
                    stat,
                ]
            )

            rows.append(
                {
                    "x": x,
                    "move": move,
                    "lat": float(
                        g[
                            "lat"
                        ].iloc[
                            i
                        ]
                    ),
                    "lon": float(
                        g[
                            "lon"
                        ].iloc[
                            i
                        ]
                    ),
                }
            )

        if len(
            rows
        ) < 3:
            continue

        x_seq = np.asarray(
            [
                r[
                    "x"
                ]
                for r in rows
            ],
            dtype=float,
        )

        moves = np.asarray(
            [
                r[
                    "move"
                ]
                for r in rows
            ],
            dtype=float,
        )

        sequences.append(
            {
                "trip_id": str(
                    trip_id
                ),
                "seal": str(
                    g[
                        "seal"
                    ].iloc[
                        0
                    ]
                ),
                "x": x_seq,
                "moves": moves,
                "n_steps": len(
                    rows
                ),
            }
        )

    if not sequences:
        raise RuntimeError(
            "No usable multivariate trip sequences."
        )

    return sequences


class MultiWindowDataset(
    Dataset
):
    def __init__(
        self,
        sequences,
        window,
        x_scaler,
        y_scaler,
    ):
        self.samples = []

        for seq in sequences:
            x = x_scaler.transform(
                seq[
                    "x"
                ]
            ).astype(
                np.float32
            )

            y = y_scaler.transform(
                seq[
                    "moves"
                ]
            ).astype(
                np.float32
            )

            for i in range(
                window,
                len(
                    x
                ),
            ):
                self.samples.append(
                    (
                        x[
                            i - window:i
                        ],
                        y[
                            i
                        ],
                    )
                )

        if not self.samples:
            raise RuntimeError(
                "No training windows; reduce --window."
            )

    def __len__(
        self
    ):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        i,
    ):
        x, y = self.samples[
            i
        ]

        return (
            torch.tensor(
                x
            ),
            torch.tensor(
                y
            ),
        )


class MultivariateRNN(
    nn.Module
):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                2,
            ),
        )

    def forward(
        self,
        x,
    ):
        out, _ = self.rnn(
            x
        )

        return self.head(
            out[
                :,
                -1,
                :,
            ]
        )


def train_model(
    model,
    loader,
    device,
    epochs,
    lr,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    criterion = nn.MSELoss()

    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        losses = []

        for x, y in loader:
            x = x.to(
                device
            )

            y = y.to(
                device
            )

            optimizer.zero_grad()

            pred = model(
                x
            )

            loss = criterion(
                pred,
                y,
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            losses.append(
                float(
                    loss.item()
                )
            )

        mse = float(
            np.mean(
                losses
            )
        )

        history.append(
            {
                "epoch": epoch,
                "train_mse_scaled": mse,
            }
        )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):
            print(
                f"Epoch {epoch:03d}/{epochs} "
                f"MSE={mse:.6f}"
            )

    return pd.DataFrame(
        history
    )


def main(
    args
):
    set_seed(
        args.seed
    )

    out = Path(
        args.out_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        args.device
        if args.device != "auto"
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    print(
        "Device:",
        device,
    )

    df = load_trips(
        args.trip_csv,
        args.trip_gap_hours,
    )

    dynamic_cols, static_cols = (
        available_feature_columns(
            df
        )
    )

    if args.no_social:
        dynamic_cols = [
            c
            for c in dynamic_cols
            if c not in {
                "nearby_seal_count",
                "nearest_seal_distance_km",
                "mean_nearby_distance_km",
                "same_sex_nearby_count",
                "opposite_sex_nearby_count",
                "nearby_male_count",
                "nearby_female_count",
                "social_density_score",
            }
        ]

    print(
        "Dynamic features:",
        dynamic_cols,
    )

    print(
        "Static features:",
        static_cols,
    )

    if not static_cols:
        print(
            "WARNING: no static physical traits found."
        )

    feature_medians = fill_numeric_features(
        df,
        dynamic_cols
        + static_cols,
    )

    sequences = build_trip_sequences(
        df,
        dynamic_cols,
        static_cols,
        args.max_step_gap_hours,
        max(
            5,
            args.window + 2,
        ),
    )

    all_x = np.vstack(
        [
            x[
                "x"
            ]
            for x in sequences
        ]
    )

    all_y = np.vstack(
        [
            x[
                "moves"
            ]
            for x in sequences
        ]
    )

    x_scaler = Standardizer().fit(
        all_x
    )

    y_scaler = Standardizer().fit(
        all_y
    )

    dataset = MultiWindowDataset(
        sequences,
        args.window,
        x_scaler,
        y_scaler,
    )

    print(
        f"Usable trips: {len(sequences):,}"
    )

    print(
        f"Training windows: {len(dataset):,}"
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    input_cols = (
        [
            "north_km",
            "east_km",
        ]
        + dynamic_cols
        + static_cols
    )

    model = MultivariateRNN(
        input_dim=len(
            input_cols
        ),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(
        device
    )

    history = train_model(
        model,
        loader,
        device,
        args.epochs,
        args.lr,
    )

    history.to_csv(
        out
        / "training_history.csv",
        index=False,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "window": args.window,
            "input_dim": len(
                input_cols
            ),
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "input_cols": input_cols,
            "dynamic_cols": dynamic_cols,
            "static_cols": static_cols,
            "x_mean": x_scaler.mean,
            "x_std": x_scaler.std,
            "y_mean": y_scaler.mean,
            "y_std": y_scaler.std,
            "feature_medians": feature_medians,
        },
        out
        / "multivariate_rnn_model.pt",
    )

    metadata = pd.DataFrame(
        [
            {
                "window": args.window,
                "input_dim": len(
                    input_cols
                ),
                "hidden_dim": args.hidden_dim,
                "n_trips": len(
                    sequences
                ),
                "n_training_windows": len(
                    dataset
                ),
                "dynamic_cols": "|".join(
                    dynamic_cols
                ),
                "static_cols": "|".join(
                    static_cols
                ),
            }
        ]
    )

    metadata.to_csv(
        out
        / "model_metadata.csv",
        index=False,
    )

    print(
        "\nSaved:",
        out
        / "multivariate_rnn_model.pt",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument(
        "--trip_csv",
        required=True,
    )

    p.add_argument(
        "--out_dir",
        default="multivariate_rnn",
    )

    p.add_argument(
        "--window",
        type=int,
        default=20,
    )

    p.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
    )

    p.add_argument(
        "--num_layers",
        type=int,
        default=1,
    )

    p.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    p.add_argument(
        "--trip_gap_hours",
        type=float,
        default=24.0,
    )

    p.add_argument(
        "--max_step_gap_hours",
        type=float,
        default=24.0,
    )

    p.add_argument(
        "--no_social",
        action="store_true",
        help=(
            "Train trajectory + environment + physical traits, "
            "excluding social variables."
        ),
    )

    p.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    main(
        p.parse_args()
    )
