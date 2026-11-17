"""Stage 8: figures for the report (matplotlib) and the interactive map (Folium).

Every function takes plain pandas tables, so the figures can be rebuilt without Spark.
Colours: one fixed palette. Blue is always "night / dark complaints / the score", orange is
always "day / placebo complaints", grey is context. Text is never drawn in a data colour.
"""
from __future__ import annotations

import json
from pathlib import Path

import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
VIOLET = "#6250d6"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE_RAMP = ["#fbd9cb", "#f6b79d", "#f1936f", "#eb6834", "#c4501f", "#9a3d16", "#6f2b0e"]
VIOLET_RAMP = ["#dcd8f7", "#bcb4ef", "#9a8fe6", "#7667dc", "#5443c2", "#40329a", "#2d2370"]


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
            "font.family": "sans-serif", "font.size": 10,
            "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
            "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
            "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold",
            "axes.titlelocation": "left", "legend.frameon": False, "lines.solid_capstyle": "round",
        }
    )


def _title(ax, title: str, subtitle: str = ""):
    ax.set_title(title, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- figures

def hour_month_heatmap(counts: pd.DataFrame, sun_monthly: pd.DataFrame, path: Path) -> Path:
    """counts: month, hour, per_100_days.  sun_monthly: month, sunrise_h, sunset_h (clock hours)."""
    _style()
    grid = counts.pivot(index="hour", columns="month", values="per_100_days").reindex(
        index=range(24), columns=range(1, 13)).fillna(0)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    cmap = LinearSegmentedColormap.from_list("blue", [SURFACE, *BLUE_RAMP])
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap=cmap, origin="upper",
                   extent=(0.5, 12.5, 24, 0), vmin=0)
    ax.grid(False)
    for name, col in (("sunrise", "sunrise_h"), ("sunset", "sunset_h")):
        m = sun_monthly.sort_values("month")
        ax.plot(m["month"], m[col], color=SURFACE, lw=4.5)
        ax.plot(m["month"], m[col], color=INK, lw=2)
        y = float(m[col].iloc[5])
        ax.annotate(name, (6, y), xytext=(0, -9 if name == "sunset" else 9), textcoords="offset points",
                    ha="center", va="center", color=INK, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85))
    ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)])
    ax.set_ylabel("Hour of day (clock time)")
    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label("Pedestrian crashes per 100 days", color=INK2)
    cb.outline.set_visible(False)
    _title(ax, "Pedestrian crashes by month and hour of day",
           "Crashes with a pedestrian injured or killed, by month and hour. Lines show average sunrise and sunset.")
    return _save(fig, path)


def forest(effects: pd.DataFrame, path: Path) -> Path:
    """One row per run: ratio = RR_night / RR_day with its 95% interval."""
    _style()
    e = effects.reset_index(drop=True)
    n = len(e)
    fig, ax = plt.subplots(figsize=(9.6, 0.52 * n + 1.9))
    y = np.arange(n)[::-1]
    colors = [ORANGE if k == "placebo" else BLUE for k in e["kind"]]
    ax.axvline(1.0, color=AXIS, lw=1.2, zorder=1)
    for yi, (_, r), c in zip(y, e.iterrows(), colors):
        ax.plot([r["ratio_lo"], r["ratio_hi"]], [yi, yi], color=c, lw=2, zorder=2)
        ax.plot(r["ratio"], yi, "o", ms=8.5, color=c, mec=SURFACE, mew=2, zorder=3)
    ax.set_yticks(y, e["label"])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    lo = min(0.8, float(np.nanmin(e["ratio_lo"])) - 0.05)
    hi = max(1.2, float(np.nanmax(e["ratio_hi"])) + 0.05)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Night rate ratio divided by day rate ratio (1.0 = no lighting effect)")
    tx = ax.get_yaxis_transform()
    ax.text(1.03, n - 0.3, "ratio [95% interval]", transform=tx, color=MUTED, fontsize=8.5, va="bottom")
    ax.text(1.33, n - 0.3, "night crashes\nduring outage", transform=tx, color=MUTED, fontsize=8.5, va="bottom")
    for yi, (_, r) in zip(y, e.iterrows()):
        ax.text(1.03, yi, f"{r['ratio']:.2f} [{r['ratio_lo']:.2f}, {r['ratio_hi']:.2f}]",
                transform=tx, va="center", color=INK2, fontsize=9)
        ax.text(1.33, yi, f"{int(r['night_crashes_outage']):,}", transform=tx, va="center", color=INK2, fontsize=9)
    ax.legend(
        handles=[Line2D([], [], marker="o", color=BLUE, lw=2, ms=7, label="Dark-lamp complaints"),
                 Line2D([], [], marker="o", color=ORANGE, lw=2, ms=7, label="Placebo complaints (lamp works)")],
        loc="upper left", bbox_to_anchor=(0, -0.16 - 0.15 / n), ncol=2,
    )
    _title(ax, "Change in crash risk at night while a streetlight complaint is open",
           "Each row is one way of running the analysis. Bars are 95% bootstrap intervals.")
    return _save(fig, path)


def event_study(es: pd.DataFrame, path: Path, kind: str = "dark") -> Path:
    """Night and day pedestrian crash rates around the report date, indexed to the early pre-period."""
    _style()
    d = es[es["kind"] == kind]
    w = int(d["rel"].max())
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.axhline(1.0, color=AXIS, lw=1.2)
    ax.axvline(0, color=AXIS, lw=1.2)
    ends = {}
    for period, c in (("day", ORANGE), ("night", BLUE)):
        s = d[d["period"] == period].sort_values("rel").set_index("rel")["per_1000"]
        base = s.loc[-w:-8].mean()
        idx = s / base if base else s
        ax.plot(idx.index, idx.values, "o", ms=3, color=c, alpha=0.3, mew=0)
        smooth = idx.rolling(7, center=True, min_periods=4).mean()
        ax.plot(smooth.index, smooth.values, color=c, lw=2, label=f"{period.capitalize()} crashes")
        ends[period] = (smooth.dropna().index[-1], float(smooth.dropna().iloc[-1]), c)
    gap = abs(ends["day"][1] - ends["night"][1])
    for period, (x, yv, c) in ends.items():
        dy = 0 if gap > 0.04 else (7 if yv >= max(v[1] for v in ends.values()) else -7)
        ax.annotate(period, (x, yv), xytext=(8, dy), textcoords="offset points", va="center",
                    color=INK2, fontsize=9)
        ax.plot(x, yv, "o", ms=7, color=c, mec=SURFACE, mew=2)
    ax.set_xlim(-w - 1, w + 4)
    ax.set_xlabel("Nights relative to the complaint (0 = first night after the report)")
    ax.set_ylabel(f"Crash rate, relative to nights -{w} to -8")
    ax.text(0.4, ax.get_ylim()[1], "reported", color=MUTED, fontsize=8.5, va="top")
    ax.legend(loc="lower left")
    label = "dark-lamp" if kind == "dark" else "placebo"
    _title(ax, f"Pedestrian crashes near {label} complaints, before and after the report",
           "Dots are single nights; lines are 7-night moving averages. Index 1.0 = early pre-report level.")
    return _save(fig, path)


def capture(curve: pd.DataFrame, summary: dict, path: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    x = curve["top_share"] * 100
    ax.plot(x, curve["perfect"] * 100, color=AXIS, lw=1.5, label="Perfect ranking (upper bound)")
    ax.plot(x, curve["arrival"] * 100, color=MUTED, lw=2, label="Arrival order")
    ax.plot(x, curve["score"] * 100, color=BLUE, lw=2, label="Priority score")
    s20, a20 = summary["score_top20"] * 100, summary["arrival_top20"] * 100
    ax.plot([20, 20], [a20, s20], color=INK2, lw=1)
    ax.plot(20, s20, "o", ms=8, color=BLUE, mec=SURFACE, mew=2)
    ax.plot(20, a20, "o", ms=8, color=MUTED, mec=SURFACE, mew=2)
    ax.annotate(f"first 20% by score: {s20:.0f}% of crashes", (20, s20), xytext=(10, 4),
                textcoords="offset points", color=INK, fontsize=9)
    ax.annotate(f"first 20% by arrival: {a20:.0f}%", (20, a20), xytext=(10, -12),
                textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Share of complaints handled first (%)")
    ax.set_ylabel("Share of nearby night pedestrian crashes covered (%)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=3, columnspacing=1.2, handlelength=1.6)
    _title(ax, "Crashes covered when complaints are handled by score vs. by arrival order",
           f"{int(summary['test_complaints']):,} complaints after the training period; crashes within 7 nights of the report.")
    return _save(fig, path)


def model_bars(metrics: pd.DataFrame, path: Path, target: str) -> Path:
    _style()
    names = {
        "gradient_boosted_trees": "Gradient-boosted trees",
        "logistic_regression": "Logistic regression",
        "priority_score_baseline": "Priority score (stage 6)",
        "trailing_count_only": "Last year's crash count only",
    }
    m = metrics.assign(name=metrics["model"].map(names)).sort_values("top10_lift")
    fig, ax = plt.subplots(figsize=(7.4, 0.55 * len(m) + 1.7))
    ax.barh(m["name"], m["top10_lift"], height=0.3, color=BLUE)
    ax.axvline(1.0, color=AXIS, lw=1.2)
    for yv, (_, r) in enumerate(m.iterrows()):
        ax.text(r["top10_lift"] + 0.05, yv, f"{r['top10_lift']:.1f}x   (PR area {r['auc_pr']:.3f}, ROC {r['auc_roc']:.2f})",
                va="center", color=INK2, fontsize=9)
    ax.set_xlim(0, float(m["top10_lift"].max()) * 1.75)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Lift in the top 10% of complaints (1.0 = no better than chance)")
    base = float(m["base_rate"].iloc[0]) * 100
    _title(ax, "How well each ranking finds complaints followed by a crash",
           f"Target: {target}. Base rate {base:.1f}% of test complaints.")
    return _save(fig, path)


# ---------------------------------------------------------------- map

def _hex_features(df: pd.DataFrame, value: str, ramp: list[str], unit: str, digits: int) -> tuple[dict, list]:
    """GeoJSON hexagons coloured by quantile class of ``value``; returns (geojson, legend rows)."""
    v = df[value].to_numpy(dtype=float)
    edges = np.unique(np.quantile(v, np.linspace(0, 1, len(ramp) + 1)))
    if len(edges) < 2:
        edges = np.array([v.min(), v.max() + 1e-9])
    cls = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)
    colors = [ramp[int(round(i * (len(ramp) - 1) / max(len(edges) - 2, 1)))] for i in range(len(edges) - 1)]
    feats = []
    for (_, r), k in zip(df.iterrows(), cls):
        ring = [[lon, lat] for lat, lon in h3.cell_to_boundary(r["cell"])]
        ring.append(ring[0])
        props = {"fill": colors[k], "value": f"{r[value]:,.{digits}f} {unit}"}
        for extra in ("score", "outage_nights", "expected"):
            if extra in df.columns:
                props[extra] = round(float(r[extra]), 3)
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "Polygon", "coordinates": [ring]}})
    legend = [(colors[i], f"{edges[i]:,.{digits}f} to {edges[i + 1]:,.{digits}f}") for i in range(len(edges) - 1)]
    return {"type": "FeatureCollection", "features": feats}, legend


def build_map(cells: pd.DataFrame, open_ranked: pd.DataFrame, path: Path, top_cells: int = 3000,
              top_markers: int = 400) -> Path:
    """cells: cell, score (night pedestrian crashes per year in the footprint), outage_nights (per year),
    expected = score * outage_nights / 365 (crashes per year expected to fall on outage nights)."""
    import folium

    m = folium.Map(location=[40.73, -73.94], zoom_start=11, tiles="OpenStreetMap", prefer_canvas=True)
    layers = [
        ("Overlap: risk x outage nights", "expected", VIOLET_RAMP, "crashes/yr on outage nights", 3, True),
        ("Baseline night pedestrian risk", "score", BLUE_RAMP, "crashes/yr", 2, False),
        ("Outage nights per year", "outage_nights", ORANGE_RAMP, "nights/yr", 0, False),
    ]
    legend_html = []
    for name, col, ramp, unit, digits, show in layers:
        top = cells.nlargest(top_cells, col)
        top = top[top[col] > 0]
        gj, legend = _hex_features(top, col, ramp, unit, digits)
        fg = folium.FeatureGroup(name=name, show=show)
        folium.GeoJson(
            gj,
            style_function=lambda f: {"fillColor": f["properties"]["fill"], "color": f["properties"]["fill"],
                                      "weight": 0.6, "fillOpacity": 0.72},
            highlight_function=lambda f: {"weight": 2, "color": INK, "fillOpacity": 0.85},
            tooltip=folium.GeoJsonTooltip(
                fields=["score", "outage_nights", "expected"],
                aliases=["Night pedestrian crashes / yr (footprint)", "Outage nights / yr",
                         "Expected crashes / yr on outage nights"],
            ),
        ).add_to(fg)
        fg.add_to(m)
        rows = "".join(
            f'<div><span style="display:inline-block;width:12px;height:12px;background:{c};'
            f'margin-right:6px;border-radius:2px"></span>{lab}</div>' for c, lab in legend)
        legend_html.append(f'<div style="margin-top:8px"><b>{name}</b><br>'
                           f'<span style="color:{MUTED}">{unit}, top {len(top):,} cells</span>{rows}</div>')

    fg = folium.FeatureGroup(name=f"Open complaints: top {top_markers} by score", show=True)
    for _, r in open_ranked.head(top_markers).iterrows():
        where = " & ".join(str(x) for x in (r.get("intersection_street_1"), r.get("intersection_street_2"))
                           if isinstance(x, str) and x) or (r.get("incident_address") or "location not named")
        popup = (
            f"<b>{where}</b><br>{r['descriptor']}<br>"
            f"Open {int(r['days_open'])} days (reported {str(r['created_ts'])[:10]})<br>"
            f"Score: {r['score']:.2f} night pedestrian crashes / yr nearby<br>"
            f"Rank by score: <b>{int(r['rank_by_score']):,}</b> of {len(open_ranked):,}<br>"
            f"Rank by arrival: {int(r['rank_by_arrival']):,} of {len(open_ranked):,}"
        )
        folium.CircleMarker(
            [r["lat"], r["lon"]], radius=5, color=SURFACE, weight=2, fill=True, fill_color=INK,
            fill_opacity=0.9, popup=folium.Popup(popup, max_width=320),
            tooltip=f"#{int(r['rank_by_score'])} by score",
        ).add_to(fg)
    fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    box = (
        '<div style="position:fixed;bottom:18px;left:12px;z-index:9999;background:#fcfcfb;'
        'padding:10px 12px;border:1px solid rgba(11,11,11,0.1);border-radius:6px;max-width:270px;'
        f'font:12px system-ui,-apple-system,Segoe UI,sans-serif;color:{INK}">'
        '<b>Lights Out</b><br>Hexagons are H3 resolution-10 cells. Switch layers at top right. '
        'Black dots are streetlight complaints still open, ranked by score.'
        + "".join(legend_html) + "</div>"
    )
    m.get_root().html.add_child(folium.Element(box))
    path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(path))
    return path


def dump_table(df: pd.DataFrame, path: Path) -> Path:
    """Table twin of a figure, so every plotted value is also available as numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def dump_json(obj: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path
