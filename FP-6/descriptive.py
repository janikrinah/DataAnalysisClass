# descriptive.py
from IPython.display import display, Markdown
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# ---------- Utility ----------

def display_title(s, pref="Figure", num=1, center=False):
    ctag = "center" if center else "p"
    s_html = f"<{ctag}><span style='font-size: 1.2em;'><b>{pref} {num}</b>: {s}</span></{ctag}>"
    if pref == "Figure":
        s_html = f"{s_html}<br><br>"
    else:
        s_html = f"<br><br>{s_html}"
    display(Markdown(s_html))


# ---------- Summary statistics ----------

def central(x: pd.Series):
    x0 = x.mean()
    x1 = x.median()
    mode_vals = x.mode()
    x2 = mode_vals.iloc[0] if not mode_vals.empty else np.nan
    return x0, x1, x2


def dispersion(x: pd.Series):
    y0 = x.std()
    y1 = x.min()
    y2 = x.max()
    y3 = y2 - y1
    y4 = x.quantile(0.25)
    y5 = x.quantile(0.75)
    y6 = y5 - y4
    return y0, y1, y2, y3, y4, y5, y6


def _numeric_df(finaldata: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "Population",
        "Electricity demand",
        "GHG emissions",
        "FF electricity share",
        "RE electricity share",
    ]
    return finaldata[numeric_cols].copy()


def display_central_tendency_table(finaldata: pd.DataFrame, num: int = 1):
    df = _numeric_df(finaldata)
    df_central = df.apply(central, axis=0)
    df_central.index = ["mean", "median", "mode"]
    df_central = df_central.round({
        "Population": 0,
        "Electricity demand": 2,
        "GHG emissions": 2,
        "FF electricity share": 2,
        "RE electricity share": 2,
    })
    display_title("Central tendency summary statistics.", pref="Table", num=num)
    display(df_central)


def display_dispersion_table(finaldata: pd.DataFrame, num: int = 2):
    df = _numeric_df(finaldata)
    df_disp = df.apply(dispersion, axis=0)
    df_disp.index = ["st.dev.", "min", "max", "range", "25th", "75th", "IQR"]
    df_disp = df_disp.round({
        "Population": 0,
        "Electricity demand": 2,
        "GHG emissions": 2,
        "FF electricity share": 2,
        "RE electricity share": 2,
    })
    display_title("Dispersion summary statistics.", pref="Table", num=num)
    display(df_disp)


# ---------- Correlations & plotting ----------

def corrcoeff(x, y):
    return np.corrcoef(x, y)[0, 1]


def plot_regression_line(ax, x, y, **kwargs):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    if x_clean.size < 2:
        return
    a, b = np.polyfit(x_clean, y_clean, deg=1)
    x0, x1 = x_clean.min(), x_clean.max()
    y0, y1 = a * x0 + b, a * x1 + b
    ax.plot([x0, x1], [y0, y1], **kwargs)


def plot_descriptive(finaldata: pd.DataFrame, num: int = 1):
    """
    Reproduces your main descriptive figure:
    (a)–(c) FF share vs log(Pop, Demand, GHG)
    (d) GHG vs FF share by income group.
    """
    # -------- Panels (a–c): FF share vs population, demand, GHG --------
    df_main = finaldata[[
        "FF electricity share",
        "Population",
        "Electricity demand",
        "GHG emissions",
    ]].dropna()

    y_main   = df_main["FF electricity share"]
    pop_log  = np.log10(df_main["Population"] + 1)
    edem_log = np.log10(df_main["Electricity demand"] + 1)
    ghg_log  = np.log10(df_main["GHG emissions"] + 1)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)

    ivs    = [pop_log, edem_log, ghg_log]
    labels = [
        "log10(Population + 1)",
        "log10(Electricity demand + 1)",
        "log10(GHG emissions + 1)",
    ]
    colors = ["b", "r", "g"]

    for ax, x, lab, c in zip(axs.ravel()[:3], ivs, labels, colors):
        ax.scatter(x, y_main, alpha=0.5, color=c)
        plot_regression_line(ax, x, y_main, color="k", ls="-", lw=2)
        r = corrcoeff(x, y_main)
        ax.text(
            0.7, 0.3, f"r = {r:.3f}",
            color=c,
            transform=ax.transAxes,
            bbox=dict(color="0.8", alpha=0.7),
        )
        ax.set_xlabel(lab)

    axs[0, 0].set_ylabel("Fossil fuel electricity share (%)")
    axs[1, 0].set_ylabel("Fossil fuel electricity share (%)")
    [ax.set_yticklabels([]) for ax in axs[:, 1]]

    # -------- Panel (d): GHG vs FF share by income group --------
    df_inc = finaldata[[
        "FF electricity share",
        "GHG emissions",
        "Income Group FY23",
    ]].dropna()

    y_inc       = df_inc["FF electricity share"]
    ghg_inc_log = np.log10(df_inc["GHG emissions"] + 1)

    low_groups  = ["L", "LM"]
    high_groups = ["UM", "H"]

    i_low  = df_inc["Income Group FY23"].isin(low_groups)
    i_high = df_inc["Income Group FY23"].isin(high_groups)

    ax = axs[1, 1]

    for mask, c, label, yloc in zip(
        [i_low, i_high],
        ["m", "c"],
        ["Low & lower-middle income", "Upper-middle & high income"],
        [0.25, 0.65],
    ):
        ax.scatter(ghg_inc_log[mask], y_inc[mask], alpha=0.5, color=c, label=label)
        plot_regression_line(ax, ghg_inc_log[mask], y_inc[mask], color=c, ls="-", lw=2)
        r_group = corrcoeff(ghg_inc_log[mask], y_inc[mask])
        ax.text(
            0.7, yloc, f"r = {r_group:.3f}",
            color=c,
            transform=ax.transAxes,
            bbox=dict(color="0.8", alpha=0.7),
        )

    ax.set_xlabel("log10(GHG emissions + 1)")
    ax.legend()

    # Panel labels
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax_, s in zip(axs.ravel(), panel_labels):
        ax_.text(0.02, 0.92, s, size=12, transform=ax_.transAxes)

    plt.show()
    display_title("Correlations amongst main variables.", pref="Figure", num=num)


def run_all(finaldata: pd.DataFrame):
    """
    Convenience function: run the full descriptive analysis.
    """
    display_central_tendency_table(finaldata, num=1)
    display_dispersion_table(finaldata, num=2)
    plot_descriptive(finaldata, num=3)
