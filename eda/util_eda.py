"""
Utility helpers for EDA — Polymarket & Bitcoin analysis
Three static-method classes:
  PolymarketUtils   : epoch/timestamp helpers + category normalization / keyword fill
  PolymarketFeatures: DataFame-level feature engineering (join, index construction, interpolation)
  PolymarketPlots   : reusable chart functions shared across Crypto / Trump / US-Affairs sections
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


# PolymarketUtils — epoch conversion, category normalisation

class PolymarketUtils:
    """Low-level data-cleaning utilities: epoch parsing, category labelling."""

    # Rule: Title-Case words joined by hyphens, no spaces.
    # Canonical list:
    #   Crypto | Trump | Politics | Business | Tech | Coronavirus
    #   Pop-Culture | Ukraine-Russia | US-Current-Affairs | Global-Politics
    CATEGORY_CANONICAL: Dict[str, str] = {
        "us-current-affairs":  "US-Current-Affairs",
        "us current affairs":  "US-Current-Affairs",
        "global politics":     "Global-Politics",
        "global-politics":     "Global-Politics",
        "pop-culture":         "Pop-Culture",
        "coronavirus":         "Coronavirus",
        "coronavirus-":        "Coronavirus",
        "crypto":              "Crypto",
        "trump":               "Trump",
        "politics":            "Politics",
        "business":            "Business",
        "tech":                "Tech",
    }

    @staticmethod
    def infer_epoch_unit(series: pl.Series) -> str:
        """Infer the time unit of an epoch timestamp series based on its magnitude."""
        v = series.drop_nulls().cast(pl.Int64)[0]
        if v < 10 ** 11:
            return "s"
        if v < 10 ** 14:
            return "ms"
        if v < 10 ** 17:
            return "us"
        return "ns"

    @staticmethod
    def fix_epoch_cols(df: pl.DataFrame, cols: List[str]) -> pl.DataFrame:
        """Convert epoch timestamp columns to Polars datetime."""
        unit = PolymarketUtils.infer_epoch_unit(df.select(cols[0]).to_series())
        return df.with_columns(
            [pl.from_epoch(pl.col(c).cast(pl.Int64), time_unit=unit).alias(c) for c in cols]
        )

    @staticmethod
    def normalize_category(df: pl.DataFrame) -> pl.DataFrame:
        """
        Normalise pre-existing category values to the canonical Title-Case-Hyphen format.
        Strips whitespace and maps known variants; unknown values are left as-is.
        """
        canon = PolymarketUtils.CATEGORY_CANONICAL
        return df.with_columns(
            pl.col("category")
              .fill_null("")
              .str.strip_chars()
              .map_elements(
                  lambda v: canon.get(v.lower(), v),
                  return_dtype=pl.Utf8,
              )
              .alias("category")
        )

    @staticmethod
    def classify_markets_by_keywords(df: pl.DataFrame) -> pl.DataFrame:
        """
        Fill blank category values using keyword rules against the question text.
        All label strings follow the canonical Title-Case-Hyphen convention.
        Calls normalize_category() first.
        """
        rules: List[Tuple[str, str]] = [
            ("Ukraine & Russia",    r"\b(ukraine|russia|putin|zelensky|kiev|kyiv|donbas|luhansk|crimea|invasion|war in ukraine)\b"),
            ("Coronavirus",         r"\b(coronavirus|covid-19|covid19|covid|pandemic|sars-cov-2|vaccine|vaccination|omicron|delta|lockdown)\b"),
            ("Tech",                r"\b(tech|technology|ai|artificial intelligence|machine learning|ml|deep learning|semiconductor|chip|startup|silicon valley|nvidia|intel|gpu)\b"),
            ("Pop-Culture",         r"\b(movie|film|tv|television|album|song|singer|actor|actress|celebrity|k-pop|netflix|disney|oscars|grammys|pop culture)\b"),
            ("Business",            r"\b(stock|stocks|market|earnings|ipo|merger|acquisition|acquire|ceo|cfo|profit|revenue|dow jones|s&p|nasdaq)\b"),
            ("US-Current-Affairs",  r"\b(biden|trump|white house|congress|senate|house of representatives|supreme court|federal reserve|fbi|doj|department of justice|fed|interest)\b"),
            ("Global-Politics",     r"\b(united nations|united-nations|un|diplomacy|sanction|sanctions|nato|eu|european union|g20|foreign policy|geopolitic(s)?)\b"),
            ("Politics",            r"\b(politic(s)?|election(s)?|campaign|vote|ballot|parliament|minister|prime minister)\b"),
        ]

        df = PolymarketUtils.normalize_category(df)
        df = df.with_columns(
            pl.col("question").fill_null("").str.to_lowercase().alias("__q_lc"),
        )

        def _is_blank():
            return pl.col("category") == ""

        for label, pattern in rules:
            df = df.with_columns(
                pl.when(_is_blank() & pl.col("__q_lc").str.contains(pattern))
                  .then(pl.lit(label))
                  .otherwise(pl.col("category"))
                  .alias("category")
            )

        return df.drop("__q_lc")

    @staticmethod
    def apply_trump_override(df: pl.DataFrame) -> pl.DataFrame:
        """Force any market mentioning trump/donald trump → category = 'Trump'."""
        trump_pattern = r"\b(trump|donald trump|president trump)\b"
        df = df.with_columns(
            pl.col("question").fill_null("").str.to_lowercase().alias("__q_lc_trump")
        )
        df = df.with_columns(
            pl.when(pl.col("__q_lc_trump").str.contains(trump_pattern))
              .then(pl.lit("Trump"))
              .otherwise(pl.col("category"))
              .alias("category")
        )
        return df.drop("__q_lc_trump")

    @staticmethod
    def apply_crypto_override(df: pl.DataFrame) -> pl.DataFrame:
        """
        Tag any market with a crypto-keyword question as 'Crypto'
        only if it currently has a blank category.
        """
        df = df.with_columns(
            pl.col("category").fill_null("").alias("category"),
            pl.col("question").fill_null("").str.to_lowercase().alias("__q_lc"),
        )
        df = df.with_columns(
            pl.when(
                (pl.col("category") == "") &
                (pl.col("__q_lc").str.contains(r"\b(bitcoin|btc|crypto|eth|bsv|bch|ada|xrp|solana|ethereum)\b"))
            ).then(pl.lit("Crypto"))
            .otherwise(pl.col("category"))
            .alias("category")
        )
        return df.drop("__q_lc")


# PolymarketFeatures — DataFrame-level feature engineering

class PolymarketFeatures:
    """Feature engineering: joins, duration metrics, poly-index construction."""

    @staticmethod
    def backfill_summary_from_odds(
        df_summary: pl.DataFrame,
        df_odds: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Backfill first_trade / last_trade / trade_count in df_summary
        using per-market min/max timestamps from df_odds (which covers older dates).
        """
        odds_ts = (
            df_odds
            .group_by("market_id")
            .agg([
                pl.col("timestamp").min().alias("first_trade_odds"),
                pl.col("timestamp").max().alias("last_trade_odds"),
                pl.col("timestamp").count().cast(pl.Int64).alias("trade_count_odds"),
            ])
        )

        return (
            df_summary
            .with_columns(pl.col("trade_count").cast(pl.Int64))
            .join(odds_ts, on="market_id", how="left")
            .with_columns([
                pl.when(pl.col("first_trade").is_null())
                  .then(pl.col("first_trade_odds"))
                  .otherwise(pl.col("first_trade"))
                  .alias("first_trade"),
                pl.when(pl.col("last_trade").is_null())
                  .then(pl.col("last_trade_odds"))
                  .otherwise(pl.col("last_trade"))
                  .alias("last_trade"),
                pl.when(
                    (pl.col("trade_count").is_null()) | (pl.col("trade_count") == 0)
                ).then(pl.col("trade_count_odds"))
                 .otherwise(pl.col("trade_count"))
                 .alias("trade_count"),
            ])
            .drop(["first_trade_odds", "last_trade_odds", "trade_count_odds"])
        )

    @staticmethod
    def compute_std_token_price(df_odds: pl.DataFrame) -> pl.DataFrame:
        """Return per-market std of token price from odds tick data."""
        return df_odds.group_by("market_id").agg(
            pl.std("price").alias("std_token_price")
        )

    @staticmethod
    def join_markets_with_summary(
        markets_df: pl.DataFrame,
        summary_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Inner-join markets with the summary table; compute:
          - market_open_duration_minutes
          - actual_market_trade_duration_minutes (with null fallback)
          - active_trade_ratio = actual / open, clipped to [0, 1]
        Drops rows with null created_at or end_date after the join.
        """
        markets_with_dur = markets_df.with_columns(
            (pl.col("end_date") - pl.col("created_at"))
            .dt.total_minutes()
            .floor()
            .alias("market_open_duration_minutes")
        )

        return (
            markets_with_dur
            .join(summary_df, on="market_id", how="inner", suffix="_summary")
            .select([
                "market_id", "question", "slug", "volume", "trade_count",
                "created_at", "end_date", "market_open_duration_minutes",
                "first_trade", "last_trade",
            ])
            .with_columns(
                pl.when(pl.col("first_trade").is_null())
                  .then(pl.col("end_date"))
                  .otherwise(pl.col("first_trade"))
                  .alias("first_trade"),
                pl.when(pl.col("last_trade").is_null())
                  .then(pl.col("end_date"))
                  .otherwise(pl.col("last_trade"))
                  .alias("last_trade"),
            )
            .with_columns(
                (pl.col("last_trade") - pl.col("first_trade"))
                .dt.total_minutes()
                .floor()
                .alias("actual_market_trade_duration_minutes")
            )
            .with_columns(
                pl.when(
                    (pl.col("market_open_duration_minutes").is_null()) |
                    (pl.col("market_open_duration_minutes") <= 0)
                )
                .then(pl.lit(0))
                .otherwise(
                    (pl.col("actual_market_trade_duration_minutes") /
                     pl.col("market_open_duration_minutes"))
                    .clip(0.0, 1.0)
                )
                .alias("active_trade_ratio")
            )
            .drop_nulls(subset=["created_at", "end_date"])
        )

    @staticmethod
    def add_poly_index(
        df: pl.DataFrame,
        index_col: str,
        weights: Optional[Dict[str, float]] = None,
    ) -> pl.DataFrame:
        """
        Build a composite [0, 1] engagement index from:
          trade_count, volume, std_token_price, active_trade_ratio.
        Each metric is log1p-transformed then min-max normalised before
        weighted summation.

        Parameters
        ----------
        df        : DataFrame that includes the four source columns.
        index_col : Name of the output column (e.g. 'crypto_poly_index').
        weights   : dict with keys trade_count / volume / std_token_price /
                    active_trade_ratio. Must sum to 1.0. Defaults to 0.25 each.
        """
        if weights is None:
            weights = {
                "trade_count": 0.25,
                "volume": 0.25,
                "std_token_price": 0.25,
                "active_trade_ratio": 0.25,
            }

        def _minmax(col: str) -> pl.Expr:
            c = pl.col(col)
            return (
                pl.when(c.max() == c.min())
                  .then(pl.lit(0.0))
                  .otherwise((c - c.min()) / (c.max() - c.min()))
            )

        return (
            df
            .with_columns([
                pl.col("trade_count").log1p().alias("_tc_log"),
                pl.col("volume").log1p().alias("_vol_log"),
                pl.col("std_token_price").log1p().alias("_stp_log"),
            ])
            .with_columns([
                _minmax("_tc_log").alias("_tc_norm"),
                _minmax("_vol_log").alias("_vol_norm"),
                _minmax("_stp_log").alias("_stp_norm"),
                _minmax("active_trade_ratio").alias("_atr_norm"),
            ])
            .with_columns(
                (
                    pl.col("_tc_norm")  * weights["trade_count"]
                    + pl.col("_vol_norm") * weights["volume"]
                    + pl.col("_stp_norm") * weights["std_token_price"]
                    + pl.col("_atr_norm") * weights["active_trade_ratio"]
                )
                .clip(0.0, 1.0)
                .alias(index_col)
            )
            .drop(["_tc_log", "_vol_log", "_stp_log", "_tc_norm", "_vol_norm", "_stp_norm", "_atr_norm"])
            .with_columns(pl.col(index_col).fill_null(0.0))
        )

    @staticmethod
    def build_daily_index_ts(
        per_market_df: pd.DataFrame,
        index_col: str,
    ) -> pd.DataFrame:
        """
        Aggregate per-market Poly Index to daily mean, reindex to full calendar,
        and linearly interpolate missing dates.

        Returns a DataFrame with columns ['date', index_col].
        """
        daily_raw = (
            per_market_df
            .dropna(subset=["created_at", index_col])
            .assign(date=lambda d: pd.to_datetime(d["created_at"]).dt.normalize())
            .groupby("date")[index_col]
            .mean()
        )
        full_range = pd.date_range(
            start=daily_raw.index.min(),
            end=daily_raw.index.max(),
            freq="D",
            name="date",
        )
        ts = daily_raw.reindex(full_range).interpolate(method="linear").reset_index()
        ts.columns = ["date", index_col]
        ts["date"] = pd.to_datetime(ts["date"])
        return ts

    @staticmethod
    def build_daily_index_agg(
        ts_df: pd.DataFrame,
        index_col: str,
    ) -> pd.DataFrame:
        """
        Aggregate the output of build_daily_index_ts (or similar) to a
        date-indexed DataFrame with columns [min, mean, max, sum, std, n_markets].
        Fills any remaining date gaps with bfill/ffill so there are no NaN rows.
        """
        daily = (
            ts_df
            .assign(date=ts_df["date"].dt.date)
            .groupby("date")[index_col]
            .agg(["min", "mean", "max", "sum", "std"])
            .fillna(0.0)
            .reset_index()
            .set_index("date")
        )
        date_range = pd.date_range(
            start=daily.index.min(), end=daily.index.max(), freq="D"
        )
        return daily.reindex(date_range).bfill().ffill().fillna(0.0)

    @staticmethod
    def lag_corr(df: pd.DataFrame, col: str, lag: int) -> float:
        """Pearson correlation between df[col] and df['PriceUSD'] at a given lag.
        Negative lag → col leads price; positive lag → price leads col."""
        s = df[col]
        p = df["PriceUSD"]
        if lag == 0:
            return float(s.corr(p))
        elif lag > 0:
            return float(s.iloc[:-lag].reset_index(drop=True).corr(p.iloc[lag:].reset_index(drop=True)))
        else:
            return float(s.iloc[-lag:].reset_index(drop=True).corr(p.iloc[:lag].reset_index(drop=True)))

    @staticmethod
    def prepare_granger_df(
        df: pd.DataFrame,
        price_col: str = "PriceUSD",
        index_col: str = "mean",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Prepare bivariate series for Granger test: [btc_ret_1d, poly_change]."""
        out = df[[date_col, price_col, index_col]].copy()
        out = out.sort_values(date_col).reset_index(drop=True)
        out["btc_ret_1d"] = np.log(out[price_col] / out[price_col].shift(1))
        out["poly_change"] = out[index_col].pct_change()
        out = out.replace([np.inf, -np.inf], np.nan)
        return out[["btc_ret_1d", "poly_change"]].dropna().reset_index(drop=True)

    @staticmethod
    def _adf_check(series: pd.Series, label: str, alpha: float) -> Dict[str, float | str]:
        """Run ADF test and return one-line summary metadata."""
        clean = series.replace([np.inf, -np.inf], np.nan).dropna()
        adf_stat, p_val, _, _, _, _ = adfuller(clean)
        return {
            "label": label,
            "adf_stat": float(adf_stat),
            "p_value": float(p_val),
            "status": "stationary" if p_val < alpha else "NON-stationary",
        }

    @staticmethod
    def _extract_granger_pvalues(gc_result: dict) -> List[float]:
        """Extract min p-value per lag across 4 statsmodels Granger test statistics."""
        pvals: List[float] = []
        for lag in sorted(gc_result.keys()):
            ps = [
                gc_result[lag][0][test][1]
                for test in ("ssr_ftest", "ssr_chi2test", "lrtest", "params_ftest")
            ]
            pvals.append(float(min(ps)))
        return pvals

    @staticmethod
    def run_granger(
        bivariate_df: pd.DataFrame,
        label: str,
        max_lag: int = 30,
        alpha: float = 0.05,
        verbose: bool = False,
    ) -> Dict[str, object]:
        """
        Run bidirectional Granger causality and return p-values + stationarity summaries.

        Returns keys:
          - result_df (lag, p_poly->btc, p_btc->poly)
          - adf_rows  (list of stationarity check dictionaries)
          - sig_lags_poly_to_btc / sig_lags_btc_to_poly (list[int])
        """
        adf_rows = [
            PolymarketFeatures._adf_check(bivariate_df["btc_ret_1d"], "btc_ret_1d", alpha),
            PolymarketFeatures._adf_check(bivariate_df["poly_change"], "poly_change", alpha),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gc_poly_to_btc = grangercausalitytests(
                bivariate_df[["btc_ret_1d", "poly_change"]],
                maxlag=max_lag,
                verbose=verbose,
            )
            gc_btc_to_poly = grangercausalitytests(
                bivariate_df[["poly_change", "btc_ret_1d"]],
                maxlag=max_lag,
                verbose=verbose,
            )

        lags = list(range(1, max_lag + 1))
        p_poly_to_btc = PolymarketFeatures._extract_granger_pvalues(gc_poly_to_btc)
        p_btc_to_poly = PolymarketFeatures._extract_granger_pvalues(gc_btc_to_poly)

        result_df = pd.DataFrame({
            "lag": lags,
            "p_poly->btc": p_poly_to_btc,
            "p_btc->poly": p_btc_to_poly,
        })

        sig_lags_poly_to_btc = result_df.loc[result_df["p_poly->btc"] < alpha, "lag"].tolist()
        sig_lags_btc_to_poly = result_df.loc[result_df["p_btc->poly"] < alpha, "lag"].tolist()

        return {
            "label": label,
            "result_df": result_df,
            "adf_rows": adf_rows,
            "sig_lags_poly_to_btc": sig_lags_poly_to_btc,
            "sig_lags_btc_to_poly": sig_lags_btc_to_poly,
        }

    @staticmethod
    def run_granger_for_datasets(
        datasets: Dict[str, pd.DataFrame],
        price_col: str = "PriceUSD",
        index_col: str = "mean",
        date_col: str = "date",
        max_lag: int = 30,
        alpha: float = 0.05,
        verbose: bool = False,
    ) -> Dict[str, Dict[str, object]]:
        """Run Granger causality for multiple index datasets."""
        out: Dict[str, Dict[str, object]] = {}
        for name, df_src in datasets.items():
            biv = PolymarketFeatures.prepare_granger_df(
                df_src,
                price_col=price_col,
                index_col=index_col,
                date_col=date_col,
            )
            out[name] = PolymarketFeatures.run_granger(
                biv,
                label=name,
                max_lag=max_lag,
                alpha=alpha,
                verbose=verbose,
            )
        return out

    @staticmethod
    def build_granger_summary(
        granger_results: Dict[str, Dict[str, object]],
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Create a compact summary table of best lag and minimum p-value by direction."""
        rows = []
        for name, payload in granger_results.items():
            res = payload["result_df"]
            for col, direction in (("p_poly->btc", "PolyIndex → BTC"), ("p_btc->poly", "BTC → PolyIndex")):
                best_idx = int(res[col].idxmin())
                best_lag = int(res.loc[best_idx, "lag"])
                best_p = float(res.loc[best_idx, col])
                rows.append({
                    "Index": name,
                    "Direction": direction,
                    "Best Lag": best_lag,
                    "Min p": round(best_p, 4),
                    "Significant (alpha=0.05)": "YES ***" if best_p < alpha else "no",
                })
        return pd.DataFrame(rows)

    @staticmethod
    def run_executive_polymarket_granger(
        data_dir: Path,
        btc_price_df: pd.DataFrame,
        plots_dir: Path,
        max_lag: int = 30,
        alpha: float = 0.05,
        verbose: bool = False,
    ) -> Dict[str, object]:
        """
        Executive notebook helper:
        - load 3 Polymarket index parquet files,
        - run bidirectional Granger tests,
        - render/save p-value heatmaps,
        - build compact statistical + decision summary tables.
        """
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        def _join_index_with_btc(daily_agg: pd.DataFrame, source_index_col: str) -> pd.DataFrame:
            df_idx = daily_agg.copy()

            if "date" not in df_idx.columns:
                idx_name = df_idx.index.name if df_idx.index.name is not None else "index"
                df_idx = df_idx.reset_index().rename(columns={idx_name: "date"})
            elif isinstance(df_idx.index, pd.DatetimeIndex):
                df_idx = df_idx.reset_index(drop=True)

            df_idx = df_idx.loc[:, ~df_idx.columns.duplicated(keep="first")].copy()
            df_idx["date"] = pd.to_datetime(df_idx["date"], errors="coerce")
            df_idx = df_idx.dropna(subset=["date"])

            if source_index_col not in df_idx.columns:
                raise KeyError(
                    f"Expected index column '{source_index_col}' not found. Available: {list(df_idx.columns)}"
                )

            if source_index_col != "mean":
                df_idx = df_idx.rename(columns={source_index_col: "mean"})

            return df_idx.merge(btc_price_df, on="date", how="inner")

        def _build_pval_matrix(granger_results: Dict[str, Dict[str, object]], key: str) -> pd.DataFrame:
            frames = []
            for name, payload in granger_results.items():
                res = payload["result_df"]
                s = res[["lag", key]].copy().rename(columns={key: name})
                frames.append(s.set_index("lag"))
            return pd.concat(frames, axis=1)

        def _normalize_direction_label(label: str) -> str:
            return str(label).replace("→", "->").strip()

        def _usage_label(poly_to_btc: bool, btc_to_poly: bool) -> str:
            if poly_to_btc and btc_to_poly:
                return "Conditional secondary timing signal"
            if (not poly_to_btc) and btc_to_poly:
                return "Confirmation-only signal"
            if poly_to_btc and (not btc_to_poly):
                return "Early-warning candidate (needs validation)"
            return "Do not use for timing"

        df_crypto_idx = pd.read_parquet(Path(data_dir) / "polymarket_crypto_index_ts.parquet")
        df_trump_idx = pd.read_parquet(Path(data_dir) / "trump_index_ts_interp.parquet")
        df_usaffair_idx = pd.read_parquet(Path(data_dir) / "us_affairs_index_ts_interp.parquet")

        datasets = {
            "Crypto Index": _join_index_with_btc(df_crypto_idx, "crypto_poly_index"),
            "Trump Index": _join_index_with_btc(df_trump_idx, "trump_poly_index"),
            "US Affairs Index": _join_index_with_btc(df_usaffair_idx, "us_affairs_poly_index"),
        }

        granger_results = PolymarketFeatures.run_granger_for_datasets(
            datasets,
            price_col="PriceUSD",
            index_col="mean",
            date_col="date",
            max_lag=max_lag,
            alpha=alpha,
            verbose=verbose,
        )

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        for ax, key, direction in zip(
            axes,
            ["p_poly->btc", "p_btc->poly"],
            ["PolyIndex -> BTC Return", "BTC Return -> PolyIndex"],
        ):
            pval_mat = _build_pval_matrix(granger_results, key).T
            sns.heatmap(
                pval_mat,
                ax=ax,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn_r",
                vmin=0,
                vmax=1,
                linewidths=0.3,
                cbar_kws={"label": "p-value", "shrink": 0.85},
                annot_kws={"size": 6},
            )
            ax.axhline(y=0, color="black", linewidth=1.5)
            ax.set_title(
                f"Granger p-values: {direction}\\n(red cells = significant at alpha={alpha})",
                fontsize=11,
            )
            ax.set_xlabel("Lag (days)")
            ax.set_ylabel("Index")

        plt.suptitle(
            "Granger Causality Test — PolyIndex Changes vs BTC Returns\\n"
            "(max lag = 30 days | min p-value across 4 test statistics)",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        plt.savefig(
            plots_dir / "team09-eda-polymarket-granger-heatmap.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()

        summary_df = PolymarketFeatures.build_granger_summary(granger_results, alpha=alpha).copy()
        summary_df["Min p"] = summary_df["Min p"].astype(float).round(4)
        summary_df["Best Lag"] = summary_df["Best Lag"].astype(int)
        summary_df["Direction_norm"] = summary_df["Direction"].map(_normalize_direction_label)

        sig_col_candidates = [
            "Significant (alpha=0.05)",
            f"Significant (alpha={alpha})",
        ]
        sig_col = next((col for col in sig_col_candidates if col in summary_df.columns), None)
        if sig_col is None:
            raise KeyError(
                f"Significant column not found. Available columns: {list(summary_df.columns)}"
            )

        summary_df["Significant"] = summary_df[sig_col].astype(str).str.contains("YES")
        stat_view = summary_df[["Index", "Direction_norm", "Best Lag", "Min p", sig_col]].copy()
        stat_view = stat_view.rename(columns={"Direction_norm": "Direction", sig_col: f"Significant (alpha={alpha})"})

        def _lookup(index_name: str, direction: str, col: str):
            row = summary_df[
                (summary_df["Index"] == index_name) & (summary_df["Direction_norm"] == direction)
            ].iloc[0]
            return row[col]

        decision_rows = []
        for index_name in ["Crypto Index", "Trump Index", "US Affairs Index"]:
            p2b_sig = bool(_lookup(index_name, "PolyIndex -> BTC", "Significant"))
            b2p_sig = bool(_lookup(index_name, "BTC -> PolyIndex", "Significant"))
            decision_rows.append(
                {
                    "Index": index_name,
                    "Poly->BTC": f"{'YES' if p2b_sig else 'NO'} (lag {int(_lookup(index_name, 'PolyIndex -> BTC', 'Best Lag'))}, p={float(_lookup(index_name, 'PolyIndex -> BTC', 'Min p')):.4f})",
                    "BTC->Poly": f"{'YES' if b2p_sig else 'NO'} (lag {int(_lookup(index_name, 'BTC -> PolyIndex', 'Best Lag'))}, p={float(_lookup(index_name, 'BTC -> PolyIndex', 'Min p')):.4f})",
                    "Recommended Use": _usage_label(p2b_sig, b2p_sig),
                }
            )

        decision_view = pd.DataFrame(decision_rows)

        return {
            "datasets": datasets,
            "granger_results": granger_results,
            "summary_df": summary_df,
            "stat_view": stat_view,
            "decision_view": decision_view,
        }


# PolymarketPlots — reusable chart functions

class PolymarketPlots:
    """Reusable chart helpers shared across Crypto / Trump / US-Affairs sections."""

    @staticmethod
    def print_top_volume_questions(
        df: pl.DataFrame,
        n: int = 20,
        title: str = "Top Volume Market Questions",
    ) -> None:
        print(f"\n{title}:")
        print("=" * 80)
        for idx, row in enumerate(
            df.select(["question", "created_at", "category", "volume"])
              .sort("volume", descending=True)
              .head(n)
              .to_pandas()
              .itertuples(),
            1,
        ):
            q = f"{row.question} ({row.category})"
            print(f"{idx:02d}. [{row.created_at.strftime('%Y-%m-%d')}] {q:<125}{row.volume:>15,.1f}")

    @staticmethod
    def plot_category_distribution(
        df: pl.DataFrame,
        title: str = "Number of Markets by Category",
        save_path: Optional[Path] = None,
    ) -> None:
        ax = (
            df.group_by("category")
              .agg(pl.len().alias("count"))
              .sort("count", descending=True)
              .to_pandas()
              .plot(x="category", y="count", kind="barh", title=title, figsize=(15, 6))
        )
        ax.invert_yaxis()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_market_creation_hist(
        df: pl.DataFrame,
        title: str = "Market Creation Timeline",
        bins: int = 25,
        color: str = "#5dade2",
        save_path: Optional[Path] = None,
    ) -> None:
        df.to_pandas().hist(column="created_at", bins=bins, figsize=(10, 6), color=color)
        plt.suptitle(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_daily_volume_and_count(
        markets_df: pl.DataFrame,
        title: str,
        bar_color: str = "#5dade2",
        line_color: str = "#1f618d",
        save_path: Optional[Path] = None,
    ) -> None:
        """Dual-panel: daily volume (top) + daily market count (bottom)."""
        daily = (
            markets_df
            .with_columns(pl.col("created_at").cast(pl.Date).alias("date"))
            .group_by("date")
            .agg(
                pl.col("volume").sum().alias("daily_volume"),
                pl.col("market_id").count().alias("daily_market_count"),
            )
            .sort("date")
            .to_pandas()
        )
        daily["vol_roll30"] = daily["daily_volume"].rolling(30, min_periods=7, center=True).mean()
        daily["mkt_roll30"] = daily["daily_market_count"].rolling(30, min_periods=7, center=True).mean()

        fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

        ax = axes[0]
        ax.bar(daily["date"], daily["daily_volume"], color=bar_color, alpha=0.4, width=1.0, label="Daily Volume")
        ax.plot(daily["date"], daily["vol_roll30"], color=line_color, linewidth=2.0, label="30-day rolling avg")
        ax.set_ylabel("Total Volume (USD)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.25)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
        )

        ax2 = axes[1]
        ax2.bar(daily["date"], daily["daily_market_count"], color=bar_color, alpha=0.45, width=1.0, label="New markets / day")
        ax2.plot(daily["date"], daily["mkt_roll30"], color=line_color, linewidth=2.0, label="30-day rolling avg")
        ax2.set_ylabel("# Markets Created", fontsize=12)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.legend(fontsize=9, loc="upper left")
        ax2.grid(alpha=0.25)
        ax2.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

        print(f"  Total markets   : {markets_df.height:,}")
        print(f"  Date range      : {daily['date'].min()}  →  {daily['date'].max()}")
        print(f"  Total volume    : {daily['daily_volume'].sum():,.0f}")
        print(f"  Peak volume day : {daily.loc[daily['daily_volume'].idxmax(), 'date']}")
        print(f"  Peak count day  : {daily.loc[daily['daily_market_count'].idxmax(), 'date']}")

    @staticmethod
    def plot_std_token_price_scatter(
        df: pl.DataFrame | pd.DataFrame,
        title: str,
        base_color: str = "#aab7b8",
        high_color: str = "#e74c3c",
        stable_color: str = "#27ae60",
        save_path: Optional[Path] = None,
    ) -> None:
        """Scatter of std_token_price over market creation date (three tiers)."""
        _pdf = df.to_pandas() if isinstance(df, pl.DataFrame) else df.copy()
        _pdf["created_at"] = pd.to_datetime(_pdf["created_at"])

        _mean = _pdf["std_token_price"].mean()
        _std  = _pdf["std_token_price"].std()
        _hi   = min(_mean + _std, 1.0)
        _lo   = max(_mean - _std, 0.0)

        def _tier(v):
            return "high" if v > _hi else ("stable" if v < _lo else "normal")

        _pdf["tier"] = _pdf["std_token_price"].apply(_tier)
        _colors_map = {"high": high_color, "normal": base_color, "stable": stable_color}

        fig, ax = plt.subplots(figsize=(13, 5))
        for tier, color in _colors_map.items():
            sub = _pdf[_pdf["tier"] == tier]
            ax.scatter(sub["created_at"], sub["std_token_price"],
                       s=8, alpha=0.45, color=color, label=tier, rasterized=True)

        ax.axhline(_mean, color="#1a252f", linewidth=1.4, linestyle="--",
                   label=f"mean = {_mean:.3f}")
        ax.axhspan(_lo, _hi, alpha=0.06, color=stable_color,
                   label=f"μ±σ  [{_lo:.3f}, {_hi:.3f}]")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Market Created At", fontsize=12)
        ax.set_ylabel("std_token_price", fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.2)
        plt.xticks(rotation=30)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  std_token_price  mean={_mean:.4f}  std={_std:.4f}  μ+σ={_hi:.4f}")

    @staticmethod
    def plot_trade_count_insights(
        df: pl.DataFrame,
        title_prefix: str,
        bar_color: str = "#85c1e9",
        line_color: str = "#1a5276",
        save_path: Optional[Path] = None,
    ) -> None:
        """3-panel trade-count chart: daily mean trend / new markets / distribution."""
        _df_tc = (
            df
            .filter(pl.col("first_trade").is_not_null())
            .select(["created_at", "trade_count"])
            .to_pandas()
            .dropna(subset=["created_at", "trade_count"])
            .assign(date=lambda d: pd.to_datetime(d["created_at"]).dt.normalize())
        )
        _daily = (
            _df_tc.groupby("date")["trade_count"]
            .agg(daily_mean="mean", market_count="count")
            .reset_index()
            .sort_values("date")
        )
        _mu   = _df_tc["trade_count"].mean()
        _sig  = _df_tc["trade_count"].std()
        _hi_t = _mu + _sig
        _daily["rolling_30d"] = _daily["daily_mean"].rolling(30, min_periods=7, center=True).mean()

        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=False)

        ax = axes[0]
        ax.bar(_daily["date"], _daily["daily_mean"], color=bar_color, alpha=0.5, width=1.0, label="Daily mean trade_count")
        ax.plot(_daily["date"], _daily["rolling_30d"], color=line_color, linewidth=2, label="30-day rolling mean")
        ax.axhline(_mu, color="#e74c3c", linestyle="--", linewidth=1.2, label=f"Overall mean = {_mu:.1f}")
        ax.fill_between(_daily["date"], max(_mu - _sig, 0), _hi_t, alpha=0.08, color="#e74c3c")
        ax.annotate(f"μ = {_mu:.1f}", xy=(_daily["date"].max(), _mu),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color="#e74c3c")
        ax.set_ylabel("Avg trade_count per market", fontsize=11)
        ax.set_title("Daily Mean Trade Count per Market + 30-day Rolling Trend", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=30)

        ax2 = axes[1]
        _roll_mkt = _daily["market_count"].rolling(30, min_periods=7, center=True).mean()
        ax2.bar(_daily["date"], _daily["market_count"], color=bar_color, alpha=0.4, width=1.0, label="New markets / day")
        ax2.plot(_daily["date"], _roll_mkt, color=line_color, linewidth=2, label="30-day rolling avg")
        ax2.set_ylabel("# Markets created", fontsize=11)
        ax2.set_title(f"New {title_prefix} Markets Created per Day", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=9, loc="upper left")
        ax2.grid(alpha=0.25)
        ax2.tick_params(axis="x", rotation=30)

        ax3 = axes[2]
        _tc_vals = _df_tc["trade_count"].clip(upper=_df_tc["trade_count"].quantile(0.99))
        ax3.hist(_tc_vals, bins=60, color=bar_color, edgecolor="white", linewidth=0.3, alpha=0.8)
        ax3.axvline(_mu,  color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Mean = {_mu:.1f}")
        ax3.axvline(_hi_t, color="#e67e22", linestyle=":",  linewidth=1.2, label=f"Mean+1σ = {_hi_t:.1f}")
        ax3.axvline(float(_df_tc["trade_count"].median()), color="#27ae60", linestyle="--",
                    linewidth=1.2, label=f"Median = {_df_tc['trade_count'].median():.1f}")
        ax3.set_yscale("log")
        ax3.set_xlabel("trade_count (capped at 99th pct)", fontsize=11)
        ax3.set_ylabel("Frequency (log)", fontsize=11)
        ax3.set_title("Distribution of Trade Count per Market  (log scale)", fontsize=12, fontweight="bold")
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.25)

        plt.suptitle(f"Trade Count Insights — {title_prefix} Polymarkets",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

        _pct_hi = (_df_tc["trade_count"] > _hi_t).mean() * 100
        print(f"=== {title_prefix} trade_count summary ===")
        print(f"  Overall mean   : {_mu:>10.1f}")
        print(f"  Median         : {_df_tc['trade_count'].median():>10.1f}")
        print(f"  Std            : {_sig:>10.1f}")
        print(f"  Mean + 1σ      : {_hi_t:>10.1f}  ← high-activity threshold")
        print(f"  % markets > μ+σ: {_pct_hi:>9.1f}%")

    @staticmethod
    def plot_poly_index_distribution(
        index_series: pd.Series,
        index_label: str,
        title: str,
        hist_color: str = "#5dade2",
        cdf_color:  str = "#2e86c1",
        save_path: Optional[Path] = None,
    ) -> None:
        """Histogram-with-zone-coloring for any poly index."""
        _idx    = index_series.dropna()
        _mean   = _idx.mean()
        _median = _idx.median()
        _p25    = _idx.quantile(0.25)
        _p75    = _idx.quantile(0.75)
        _pct_lo = (_idx < 0.2).mean() * 100
        _pct_hi = (_idx > 0.6).mean() * 100

        fig, ax = plt.subplots(1, 1, figsize=(14, 5))

        n, bins, patches = ax.hist(_idx, bins=60, edgecolor="white", linewidth=0.3,
                                   color=hist_color, alpha=0.85)
        for patch, left in zip(patches, bins[:-1]):
            if left < 0.2:
                patch.set_facecolor("#27ae60")
            elif left > 0.6:
                patch.set_facecolor("#e74c3c")

        ax.axvline(_mean,   color="#1a252f", linestyle="--", linewidth=1.5, label=f"Mean   = {_mean:.3f}")
        ax.axvline(_median, color="#8e44ad", linestyle=":",  linewidth=1.5, label=f"Median = {_median:.3f}")
        ax.axvline(_p25,    color="#27ae60", linestyle="-",  linewidth=0.8, alpha=0.7, label=f"Q1     = {_p25:.3f}")
        ax.axvline(_p75,    color="#e74c3c", linestyle="-",  linewidth=0.8, alpha=0.7, label=f"Q3     = {_p75:.3f}")
        ax.text(0.10, ax.get_ylim()[1] * 0.92, f"Low\n({_pct_lo:.0f}%)", ha="center",
                fontsize=8, color="#27ae60", fontweight="bold")
        ax.text(0.70, ax.get_ylim()[1] * 0.92, f"High\n({_pct_hi:.0f}%)", ha="center",
                fontsize=8, color="#e74c3c", fontweight="bold")
        ax.set_xlabel(index_label, fontsize=12)
        ax.set_ylabel("Number of Markets", fontsize=12)
        ax.set_title(f"Distribution of {index_label}\n(green < 0.2 = low engagement,  red > 0.6 = high engagement)",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)

        plt.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

        print(f"  Mean   : {_mean:.4f}   Median : {_median:.4f}")
        print(f"  Q1     : {_p25:.4f}   Q3     : {_p75:.4f}")
        print(f"  Low engagement (< 0.2) : {_pct_lo:.1f}%  of markets")
        print(f"  High engagement (> 0.6): {_pct_hi:.1f}%  of markets")

    @staticmethod
    def plot_poly_index_trend(
        per_market_df: pd.DataFrame,
        index_col: str,
        title: str,
        band_color: str = "#2e86c1",
        line_color: str = "#1a5276",
        event_markers: Optional[List[Tuple[str, str, str]]] = None,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Dual-panel: daily min/mean/max band + 30-day rolling mean (top) +
        market count per day (bottom).

        Parameters
        ----------
        event_markers : list of (timestamp_str, label, color) for vertical lines.
        """
        daily = (
            per_market_df
            .dropna(subset=["created_at", index_col])
            .assign(date=lambda d: pd.to_datetime(d["created_at"]).dt.normalize())
            .groupby("date")[index_col]
            .agg(["min", "mean", "max", "count"])
            .reset_index()
            .rename(columns={"count": "n_markets"})
            .sort_values("date")
        )
        daily["rolling_mean"] = daily["mean"].rolling(30, min_periods=7, center=True).mean()
        _grand_mean = daily["mean"].mean()

        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})

        ax = axes[0]
        ax.fill_between(daily["date"], daily["min"], daily["max"],
                        alpha=0.12, color=band_color, label="Daily min–max range")
        ax.plot(daily["date"], daily["mean"],         color=band_color, linewidth=1.2, alpha=0.5, label="Daily mean")
        ax.plot(daily["date"], daily["rolling_mean"], color=line_color, linewidth=2.2, label="30-day rolling mean")
        ax.axhline(_grand_mean, color="#e74c3c", linestyle="--", linewidth=1.2,
                   label=f"Overall mean = {_grand_mean:.3f}")

        if event_markers:
            for ts_str, label, ev_color in event_markers:
                ts = pd.Timestamp(ts_str)
                ax.axvline(ts, color=ev_color, linestyle=":", linewidth=1.3, alpha=0.8)
                ax.annotate(label, xy=(ts, 0.72), xytext=(10, 0),
                            textcoords="offset points", fontsize=8, color=ev_color,
                            arrowprops=dict(arrowstyle="->", color=ev_color, lw=0.8))

        ax.set_ylabel(f"{index_col}  [0, 1]", fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.25)

        ax2 = axes[1]
        ax2.bar(daily["date"], daily["n_markets"], color=band_color, alpha=0.6, width=1.0, label="Markets / day")
        ax2.plot(daily["date"],
                 daily["n_markets"].rolling(30, min_periods=7, center=True).mean(),
                 color=line_color, linewidth=1.8, label="30-day avg")
        if event_markers:
            for ts_str, _, ev_color in event_markers:
                ax2.axvline(pd.Timestamp(ts_str), color=ev_color, linestyle=":", linewidth=1.3, alpha=0.7)
        ax2.set_ylabel("# Markets", fontsize=11)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.legend(fontsize=9, loc="upper left")
        ax2.grid(alpha=0.2)
        ax2.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

        print(f"  Overall mean  : {_grand_mean:.4f}")
        print(f"  Peak daily    : {daily['mean'].max():.4f}  on {daily.loc[daily['mean'].idxmax(), 'date'].date()}")

    @staticmethod
    def plot_poly_index_vs_btc(
        df_joined: pd.DataFrame,
        index_col: str,
        title: str,
        index_color: str = "steelblue",
        save_path_ts: Optional[Path] = None,
        save_path_scatter: Optional[Path] = None,
    ) -> Tuple[float, float]:
        """
        Dual-axis time-series (index vs BTC price) + scatter plot.
        Returns (pearson_r_mean, pearson_r_sum).
        """
        c_btc = "darkorange"

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel("Date")
        ax1.set_ylabel(f"{index_col} (mean)", color=index_color)
        ax1.plot(df_joined["date"], df_joined["mean"], color=index_color, lw=1.2, alpha=0.85, label=f"{index_col} (mean)")
        ax1.fill_between(df_joined["date"], df_joined["min"], df_joined["max"],
                         color=index_color, alpha=0.12, label=f"{index_col} [min, max]")
        ax1.tick_params(axis="y", labelcolor=index_color)

        ax2 = ax1.twinx()
        ax2.set_ylabel("BTC Price (USD)", color=c_btc)
        ax2.plot(df_joined["date"], df_joined["PriceUSD"], color=c_btc, lw=1.5, alpha=0.9, label="BTC PriceUSD")
        ax2.tick_params(axis="y", labelcolor=c_btc)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

        corr_mean  = float(df_joined["mean"].corr(df_joined["PriceUSD"]))
        corr_total = float(df_joined["sum"].corr(df_joined["PriceUSD"]))

        plt.title(
            f"{title}\nPearson r (mean vs price) = {corr_mean:.3f}  |  "
            f"r (sum vs price) = {corr_total:.3f}",
            fontsize=11,
        )
        fig.tight_layout()
        if save_path_ts:
            plt.savefig(save_path_ts, dpi=150, bbox_inches="tight")
        plt.show()

        fig2, ax3 = plt.subplots(figsize=(12, 6))
        ax3.scatter(df_joined["PriceUSD"], df_joined["mean"], alpha=0.4, s=12, color=index_color)
        ax3.set_xlabel("BTC Price (USD)")
        ax3.set_ylabel(f"{index_col} (mean)")
        ax3.set_title(f"{title}  Scatter  |  r = {corr_mean:.3f}")
        fig2.tight_layout()
        if save_path_scatter:
            plt.savefig(save_path_scatter, dpi=150, bbox_inches="tight")
        plt.show()

        print(f"\n  Pearson r (mean  vs BTC price): {corr_mean:.4f}")
        print(f"  Pearson r (total vs BTC price): {corr_total:.4f}")
        return corr_mean, corr_total

    @staticmethod
    def plot_rolling_correlation(
        roll_series: Dict[str, pd.Series],
        title: str = "30-Day Rolling Correlation with BTC Price",
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot multiple rolling-correlation series on one chart."""
        colors = ["steelblue", "crimson", "#1abc9c", "#f39c12", "#8e44ad"]
        fig, ax = plt.subplots(figsize=(12, 6))
        for (name, series), color in zip(roll_series.items(), colors):
            ax.plot(series.index, series, color=color, lw=1.2, label=name)
        ax.axhline(0,    color="black", lw=0.8, ls="--")
        ax.axhline( 0.5, color="grey",  lw=0.6, ls=":")
        ax.axhline(-0.5, color="grey",  lw=0.6, ls=":")
        ax.set_ylabel("Pearson r (30-day window)")
        ax.set_xlabel("Date")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_lag_correlation(
        lag_corr_dict: Dict[str, List[float]],
        lags: range,
        title: str = "Cross-Correlation at Different Lags vs BTC Price",
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot lagged cross-correlation for multiple index series."""
        colors = ["steelblue", "crimson", "#1abc9c", "#f39c12", "#8e44ad"]
        fig, ax = plt.subplots(figsize=(12, 5))
        for (name, values), color in zip(lag_corr_dict.items(), colors):
            ax.plot(list(lags), values, color=color, lw=1.5, label=name)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.axhline(0, color="grey",  lw=0.6, ls=":")
        ax.set_xlabel("Lag (days)  —  negative = index leads BTC; positive = BTC leads index")
        ax.set_ylabel("Pearson r")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
