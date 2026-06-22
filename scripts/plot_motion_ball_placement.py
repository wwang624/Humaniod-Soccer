#!/usr/bin/env python3
"""Visualize per-motion soccer ball placement regions.

The plot mirrors the training command logic used by MultiMotionSoccerCommand:
for each motion, the nominal ball location is the kick-foot position at
`contact_phase`, expressed relative to the first-frame anchor body position.
The feasible region is then the configured radius offset and angular offset.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _as_str(value) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def _body_index(body_names: list[str], name: str, motion_file: Path) -> int:
    try:
        return body_names.index(name)
    except ValueError as exc:
        raise ValueError(f"{motion_file} has no body '{name}'. Available bodies: {body_names}") from exc


def _sector_polygon(angle: float, radius: float, radius_range: tuple[float, float], arc_angle: float, samples: int = 32):
    r_min = max(0.0, radius + radius_range[0])
    r_max = max(0.0, radius + radius_range[1])
    a0 = angle - arc_angle
    a1 = angle + arc_angle
    outer_angles = np.linspace(a0, a1, samples)
    inner_angles = np.linspace(a1, a0, samples)
    outer = np.column_stack((r_max * np.cos(outer_angles), r_max * np.sin(outer_angles)))
    inner = np.column_stack((r_min * np.cos(inner_angles), r_min * np.sin(inner_angles)))
    return np.vstack((outer, inner))


def collect_ball_placements(
    motion_dir: Path,
    *,
    anchor_body: str,
    left_foot_body: str,
    right_foot_body: str,
    contact_phase: float,
):
    rows = []
    for path in sorted(motion_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        if "body_pos_w" not in data or "body_names" not in data:
            continue
        body_names = [_as_str(x) for x in data["body_names"]]
        anchor_idx = _body_index(body_names, anchor_body, path)
        kick_leg = _as_str(data["kick_leg"]).lower() if "kick_leg" in data else ""
        if kick_leg == "left":
            foot_body = left_foot_body
        elif kick_leg == "right":
            foot_body = right_foot_body
        else:
            foot_body = anchor_body
        foot_idx = _body_index(body_names, foot_body, path)

        body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)
        motion_len = body_pos_w.shape[0]
        contact_idx = int(round((motion_len - 1) * contact_phase))
        contact_idx = max(0, min(motion_len - 1, contact_idx))

        first_anchor_xy = body_pos_w[0, anchor_idx, :2]
        contact_foot_xy = body_pos_w[contact_idx, foot_idx, :2]
        nominal_xy = contact_foot_xy - first_anchor_xy
        radius = float(np.linalg.norm(nominal_xy))
        angle = float(math.atan2(nominal_xy[1], nominal_xy[0])) if radius > 1e-12 else 0.0
        rows.append(
            {
                "file": path.name,
                "kick_leg": kick_leg or "unknown",
                "contact_idx": contact_idx,
                "motion_len": motion_len,
                "x": float(nominal_xy[0]),
                "y": float(nominal_xy[1]),
                "radius": radius,
                "angle": angle,
            }
        )
    return rows


def plot_ball_placements(rows, output: Path, *, radius_range: tuple[float, float], arc_angle: float, title: str):
    if not rows:
        raise RuntimeError("No valid motion npz files found.")

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(rows), 1)))
    fig, ax = plt.subplots(figsize=(9.0, 7.0))

    for i, row in enumerate(rows):
        color = colors[i % len(colors)]
        polygon = _sector_polygon(row["angle"], row["radius"], radius_range, arc_angle)
        ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.12)
        ax.plot(polygon[:, 0], polygon[:, 1], color=color, linestyle="--", linewidth=0.8, alpha=0.75)
        ax.plot([0.0, row["x"]], [0.0, row["y"]], color=color, linewidth=1.0, alpha=0.75)
        marker = "o" if row["kick_leg"] == "right" else "s"
        ax.scatter(row["x"], row["y"], color=color, edgecolors="black", linewidths=0.35, s=34, marker=marker, zorder=3)
        ax.text(row["x"], row["y"], f" {row['file'].replace('.npz', '')}", fontsize=7, color=color)

    xs = np.array([r["x"] for r in rows])
    ys = np.array([r["y"] for r in rows])
    margin = 0.35
    ax.set_xlim(float(xs.min()) - margin, float(xs.max()) + margin)
    ax.set_ylim(float(ys.min()) - margin, float(ys.max()) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.6, alpha=0.35)
    ax.scatter([0.0], [0.0], color="black", s=20, zorder=4)
    ax.set_xlabel("X from first-frame anchor (m)")
    ax.set_ylabel("Y from first-frame anchor (m)")
    ax.set_title(title)

    right_count = sum(1 for r in rows if r["kick_leg"] == "right")
    left_count = sum(1 for r in rows if r["kick_leg"] == "left")
    summary = (
        f"motions={len(rows)}  left={left_count}  right={right_count}\n"
        f"x=[{xs.min():.3f}, {xs.max():.3f}]  y=[{ys.min():.3f}, {ys.max():.3f}]\n"
        f"radius=[{min(r['radius'] for r in rows):.3f}, {max(r['radius'] for r in rows):.3f}] m"
    )
    ax.text(0.01, 0.99, summary, transform=ax.transAxes, va="top", ha="left", fontsize=9)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_dir", type=Path, help="Directory containing motion .npz files.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path. Defaults to <motion_dir>/ball_placement.png.")
    parser.add_argument("--anchor-body", default="torso_link")
    parser.add_argument("--left-foot-body", default="left_ankle_roll_link")
    parser.add_argument("--right-foot-body", default="right_ankle_roll_link")
    parser.add_argument("--contact-phase", type=float, default=0.9)
    parser.add_argument("--radius-min-offset", type=float, default=-0.05)
    parser.add_argument("--radius-max-offset", type=float, default=0.05)
    parser.add_argument("--arc-angle", type=float, default=0.05, help="Angular randomization in radians.")
    args = parser.parse_args()

    output = args.output if args.output is not None else args.motion_dir / "ball_placement.png"
    rows = collect_ball_placements(
        args.motion_dir,
        anchor_body=args.anchor_body,
        left_foot_body=args.left_foot_body,
        right_foot_body=args.right_foot_body,
        contact_phase=args.contact_phase,
    )
    plot_ball_placements(
        rows,
        output,
        radius_range=(args.radius_min_offset, args.radius_max_offset),
        arc_angle=args.arc_angle,
        title=f"Ball Placement: {args.motion_dir}",
    )

    print(f"saved: {output}")
    print("file,kick_leg,contact_idx,motion_len,x,y,radius,angle_deg")
    for row in rows:
        print(
            f"{row['file']},{row['kick_leg']},{row['contact_idx']},{row['motion_len']},"
            f"{row['x']:.6f},{row['y']:.6f},{row['radius']:.6f},{math.degrees(row['angle']):.3f}"
        )


if __name__ == "__main__":
    main()
