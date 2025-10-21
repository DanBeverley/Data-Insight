import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import io
import base64
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ChartType(Enum):
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    CORRELATION_HEATMAP = "correlation_heatmap"
    BOX_PLOT = "box_plot"
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    DISTRIBUTION = "distribution"
    PAIR_PLOT = "pair_plot"
    JOINT_PLOT = "joint_plot"
    KDE_PLOT = "kde_plot"
    VIOLIN_PLOT = "violin_plot"
    SWARM_PLOT = "swarm_plot"
    HEXBIN = "hexbin"
    FEATURE_IMPORTANCE = "feature_importance"
    MISSING_DATA = "missing_data"
    DATA_SUMMARY = "data_summary"


@dataclass
class VisualizationRequest:
    chart_type: ChartType
    columns: Optional[List[str]] = None
    title: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class VisualizationResult:
    chart_base64: str
    chart_type: str
    description: str
    insights: List[str]
    interactive_html: Optional[str] = None


class IntelligentDataExplorer:
    def __init__(self):
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")
        self.figure_size = (12, 8)

    def analyze_data_characteristics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset to suggest relevant visualizations"""
        characteristics = {
            "shape": df.shape,
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
            "datetime_columns": df.select_dtypes(include=["datetime64"]).columns.tolist(),
            "missing_data": df.isnull().sum().to_dict(),
            "correlation_candidates": [],
            "suggested_visualizations": [],
        }

        numeric_cols = characteristics["numeric_columns"]
        categorical_cols = characteristics["categorical_columns"]

        if len(numeric_cols) >= 2:
            characteristics["correlation_candidates"] = numeric_cols
            characteristics["suggested_visualizations"].extend(["correlation_heatmap", "scatter_plots", "pair_plot"])

        if len(categorical_cols) > 0:
            characteristics["suggested_visualizations"].extend(["bar_charts", "count_plots"])

        if len(numeric_cols) > 0:
            characteristics["suggested_visualizations"].extend(["histograms", "box_plots", "distribution_plots"])

        if df.isnull().sum().sum() > 0:
            characteristics["suggested_visualizations"].append("missing_data_heatmap")

        return characteristics

    def get_data_head(self, df: pd.DataFrame, n: int = 10) -> Dict[str, Any]:
        """Get first n rows with metadata"""
        return {
            "data": df.head(n).to_dict("records"),
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "shape": df.shape,
            "index": df.head(n).index.tolist(),
        }

    def get_data_tail(self, df: pd.DataFrame, n: int = 10) -> Dict[str, Any]:
        """Get last n rows with metadata"""
        return {
            "data": df.tail(n).to_dict("records"),
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "shape": df.shape,
            "index": df.tail(n).index.tolist(),
        }

    def get_data_sample(self, df: pd.DataFrame, n: int = 10) -> Dict[str, Any]:
        """Get random sample with metadata"""
        sample_df = df.sample(n=min(n, len(df)), random_state=42)
        return {
            "data": sample_df.to_dict("records"),
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "shape": df.shape,
            "index": sample_df.index.tolist(),
        }

    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data information"""
        info_buffer = io.StringIO()
        df.info(buf=info_buffer)

        return {
            "info_text": info_buffer.getvalue(),
            "describe": df.describe().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).to_dict(),
            "unique_counts": {col: df[col].nunique() for col in df.columns},
        }

    def create_correlation_heatmap(
        self, df: pd.DataFrame, title: str = "Feature Correlation Heatmap"
    ) -> VisualizationResult:
        """Create correlation heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            raise ValueError("No numeric columns found for correlation analysis")

        correlation_matrix = numeric_df.corr()

        fig, ax = plt.subplots(figsize=self.figure_size)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )

        ax.set_title(title, fontsize=16, fontweight="bold")
        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()

        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_correlations.append(
                        f"{correlation_matrix.columns[i]} ↔ {correlation_matrix.columns[j]}: {corr_val:.3f}"
                    )

        insights = [
            f"Analyzed correlations between {len(numeric_df.columns)} numeric features",
            f"Found {len(strong_correlations)} strong correlations (|r| > 0.7)",
        ]

        if strong_correlations:
            insights.extend(strong_correlations[:5])

        return VisualizationResult(
            chart_base64=img_base64,
            chart_type="correlation_heatmap",
            description=f"Correlation heatmap showing relationships between {len(numeric_df.columns)} numeric features",
            insights=insights,
        )

    def create_kde_plot(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> VisualizationResult:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = columns or numeric_cols[:3]
        if not cols:
            raise ValueError("No numeric columns found for KDE plot")
        fig, ax = plt.subplots(figsize=self.figure_size)
        insights = []
        for col in cols:
            data = df[col].dropna()
            if not data.empty:
                sns.kdeplot(data, fill=True, alpha=0.3, label=col, ax=ax)
                insights.append(f"{col}: mean={data.mean():.2f}, std={data.std():.2f}")
        ax.legend()
        ax.set_title("Kernel Density Estimation", fontsize=16, fontweight="bold")
        ax.grid(True, alpha=0.3)
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        return VisualizationResult(
            chart_base64=img_base64,
            chart_type="kde_plot",
            description=f"KDE plots for {len(cols)} numeric features",
            insights=insights,
        )

    def create_violin_plot(self, df: pd.DataFrame, value_col: str, by_col: Optional[str] = None) -> VisualizationResult:
        if value_col not in df.columns:
            raise ValueError(f"Column {value_col} not found")
        fig, ax = plt.subplots(figsize=self.figure_size)
        if by_col and by_col in df.columns:
            sns.violinplot(x=by_col, y=value_col, data=df, ax=ax)
            ax.set_title(f"Violin plot of {value_col} by {by_col}", fontsize=16, fontweight="bold")
        else:
            sns.violinplot(y=df[value_col], ax=ax)
            ax.set_title(f"Violin plot of {value_col}", fontsize=16, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        return VisualizationResult(
            chart_base64=img_base64,
            chart_type="violin_plot",
            description=f"Violin plot for {value_col}{' grouped by ' + by_col if by_col else ''}",
            insights=[],
        )

    def create_swarm_plot(self, df: pd.DataFrame, value_col: str, by_col: Optional[str] = None) -> VisualizationResult:
        if value_col not in df.columns:
            raise ValueError(f"Column {value_col} not found")
        fig, ax = plt.subplots(figsize=self.figure_size)
        if by_col and by_col in df.columns:
            sns.swarmplot(x=by_col, y=value_col, data=df, ax=ax)
            ax.set_title(f"Swarm plot of {value_col} by {by_col}", fontsize=16, fontweight="bold")
        else:
            sns.swarmplot(y=df[value_col], ax=ax)
            ax.set_title(f"Swarm plot of {value_col}", fontsize=16, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        return VisualizationResult(
            chart_base64=img_base64,
            chart_type="swarm_plot",
            description=f"Swarm plot for {value_col}{' grouped by ' + by_col if by_col else ''}",
            insights=[],
        )

    def create_bar_chart(
        self, df: pd.DataFrame, value_col: Optional[str] = None, by_col: Optional[str] = None, agg: str = "mean"
    ) -> VisualizationResult:
        fig, ax = plt.subplots(figsize=self.figure_size)
        insights = []
        if by_col and by_col in df.columns:
            if not value_col:
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if not num_cols:
                    raise ValueError("No numeric columns available for aggregation")
                value_col = num_cols[0]
            grouped = getattr(df.groupby(by_col)[value_col], agg)()
            grouped.plot(kind="bar", ax=ax, color="teal", alpha=0.8)
            ax.set_title(f"{agg.title()} of {value_col} by {by_col}", fontsize=16, fontweight="bold")
            insights.append(f"{agg.title()} values across {len(grouped)} {by_col} categories")
        else:
            counts = df[value_col].value_counts() if value_col in df.columns else df.iloc[:, 0].value_counts()
            counts.plot(kind="bar", ax=ax, color="teal", alpha=0.8)
            ax.set_title(f"Counts of {value_col or df.columns[0]}", fontsize=16, fontweight="bold")
            insights.append(f"Top category: {counts.index[0]} ({counts.iloc[0]})")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        return VisualizationResult(
            chart_base64=img_base64, chart_type="bar_chart", description="Bar chart", insights=insights
        )

    def create_line_chart(
        self, df: pd.DataFrame, x_col: Optional[str] = None, y_col: Optional[str] = None
    ) -> VisualizationResult:
        if not x_col:
            dt_cols = df.select_dtypes(include=["datetime64", "datetime"]).columns.tolist()
            x_col = dt_cols[0] if dt_cols else df.columns[0]
        if not y_col:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            y_col = num_cols[0] if num_cols else df.columns[1]
        fig, ax = plt.subplots(figsize=self.figure_size)
        df_sorted = df.sort_values(by=x_col)
        ax.plot(df_sorted[x_col], df_sorted[y_col], color="steelblue")
        ax.set_title(f"{y_col} over {x_col}", fontsize=16, fontweight="bold")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        return VisualizationResult(
            chart_base64=img_base64,
            chart_type="line_chart",
            description=f"Line chart of {y_col} over {x_col}",
            insights=[],
        )

    def create_hexbin(self, df: pd.DataFrame, x_col: str, y_col: str) -> VisualizationResult:
        fig, ax = plt.subplots(figsize=self.figure_size)
        hb = ax.hexbin(df[x_col], df[y_col], gridsize=30, cmap="viridis", mincnt=1)
        plt.colorbar(hb, ax=ax)
        ax.set_title(f"Hexbin of {y_col} vs {x_col}", fontsize=16, fontweight="bold")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        return VisualizationResult(
            chart_base64=img_base64,
            chart_type="hexbin",
            description=f"Hexbin density of {y_col} vs {x_col}",
            insights=[],
        )

    def create_pair_plot(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> VisualizationResult:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = columns or num_cols[:5]
        if len(cols) < 2:
            raise ValueError("Need at least two numeric columns for pair plot")
        sns.set_theme(style="ticks")
        g = sns.pairplot(df[cols], corner=True)
        g.fig.suptitle("Pair Plot", y=1.02)
        img_buffer = io.BytesIO()
        g.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close("all")
        return VisualizationResult(
            chart_base64=img_base64,
            chart_type="pair_plot",
            description=f"Pair plot for {len(cols)} numeric features",
            insights=[],
        )

    def create_joint_plot(self, df: pd.DataFrame, x_col: str, y_col: str, kind: str = "scatter") -> VisualizationResult:
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError("Columns not found for joint plot")
        g = sns.jointplot(data=df, x=x_col, y=y_col, kind=kind)
        g.fig.suptitle(f"Joint plot ({kind}) of {y_col} vs {x_col}", y=1.02)
        img_buffer = io.BytesIO()
        g.fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close("all")
        return VisualizationResult(
            chart_base64=img_base64,
            chart_type="joint_plot",
            description=f"Joint plot ({kind}) of {y_col} vs {x_col}",
            insights=[],
        )

    def create_distribution_plots(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> VisualizationResult:
        """Create distribution plots for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if columns:
            numeric_cols = [col for col in columns if col in numeric_cols]

        if not numeric_cols:
            raise ValueError("No numeric columns found for distribution plots")

        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        insights = []

        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                ax = axes[i]

                data = df[col].dropna()

                ax.hist(data, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
                ax.axvline(data.mean(), color="red", linestyle="--", alpha=0.8, label=f"Mean: {data.mean():.2f}")
                ax.axvline(
                    data.median(), color="green", linestyle="--", alpha=0.8, label=f"Median: {data.median():.2f}"
                )

                ax.set_title(f"Distribution of {col}", fontweight="bold")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                ax.legend()
                ax.grid(True, alpha=0.3)

                skewness = data.skew()
                kurtosis = data.kurtosis()

                insights.append(f"{col}: Mean={data.mean():.2f}, Std={data.std():.2f}, Skew={skewness:.2f}")

        for j in range(len(numeric_cols), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()

        return VisualizationResult(
            chart_base64=img_base64,
            chart_type="distribution",
            description=f"Distribution plots for {len(numeric_cols)} numeric features",
            insights=insights,
        )

    def create_scatter_plot(
        self, df: pd.DataFrame, x_col: str, y_col: str, color_col: Optional[str] = None, title: Optional[str] = None
    ) -> VisualizationResult:
        """Create scatter plot between two variables"""
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Columns {x_col} or {y_col} not found in dataset")

        fig, ax = plt.subplots(figsize=self.figure_size)

        if color_col and color_col in df.columns:
            scatter = ax.scatter(df[x_col], df[y_col], c=df[color_col], alpha=0.6, cmap="viridis")
            plt.colorbar(scatter, label=color_col)
        else:
            ax.scatter(df[x_col], df[y_col], alpha=0.6, color="blue")

        ax.set_xlabel(x_col, fontweight="bold")
        ax.set_ylabel(y_col, fontweight="bold")
        ax.set_title(title or f"{y_col} vs {x_col}", fontsize=16, fontweight="bold")
        ax.grid(True, alpha=0.3)

        correlation = df[[x_col, y_col]].corr().iloc[0, 1]
        ax.text(
            0.05,
            0.95,
            f"Correlation: {correlation:.3f}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()

        insights = [
            f"Scatter plot shows relationship between {x_col} and {y_col}",
            f"Correlation coefficient: {correlation:.3f}",
            f"Relationship strength: {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.3 else 'Weak'}",
        ]

        if color_col:
            insights.append(f"Points colored by {color_col} values")

        return VisualizationResult(
            chart_base64=img_base64,
            chart_type="scatter",
            description=f"Scatter plot of {y_col} vs {x_col}",
            insights=insights,
        )

    def create_box_plots(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> VisualizationResult:
        """Create box plots for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if columns:
            numeric_cols = [col for col in columns if col in numeric_cols]

        if not numeric_cols:
            raise ValueError("No numeric columns found for box plots")

        fig, ax = plt.subplots(figsize=(max(10, len(numeric_cols) * 1.5), 8))

        box_data = [df[col].dropna() for col in numeric_cols]

        bp = ax.boxplot(box_data, labels=numeric_cols, patch_artist=True)

        colors = plt.cm.Set3(np.linspace(0, 1, len(numeric_cols)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title("Box Plots - Distribution and Outliers", fontsize=16, fontweight="bold")
        ax.set_ylabel("Values", fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()

        insights = []
        for col in numeric_cols:
            data = df[col].dropna()
            q1, q3 = data.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = data[(data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)]
            insights.append(f"{col}: {len(outliers)} outliers detected")

        return VisualizationResult(
            chart_base64=img_base64,
            chart_type="box_plot",
            description=f"Box plots showing distribution and outliers for {len(numeric_cols)} features",
            insights=insights,
        )

    def create_missing_data_heatmap(self, df: pd.DataFrame) -> VisualizationResult:
        """Create heatmap showing missing data patterns"""
        missing_data = df.isnull()

        if not missing_data.any().any():
            raise ValueError("No missing data found in the dataset")

        fig, ax = plt.subplots(figsize=self.figure_size)

        sns.heatmap(missing_data, yticklabels=False, cbar=True, cmap="viridis", ax=ax)
        ax.set_title("Missing Data Pattern", fontsize=16, fontweight="bold")
        ax.set_xlabel("Features", fontweight="bold")
        ax.set_ylabel("Samples", fontweight="bold")

        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()

        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100

        insights = [
            f"Total missing values: {missing_counts.sum()}",
            f"Columns with missing data: {(missing_counts > 0).sum()}",
        ]

        for col in missing_counts[missing_counts > 0].head(5).index:
            insights.append(f"{col}: {missing_counts[col]} missing ({missing_percentages[col]:.1f}%)")

        return VisualizationResult(
            chart_base64=img_base64,
            chart_type="missing_data",
            description="Heatmap showing missing data patterns across the dataset",
            insights=insights,
        )

    def interpret_user_request(self, user_message: str, df: pd.DataFrame) -> Optional[VisualizationRequest]:
        """Interpret user's natural language request for data visualization"""
        message = user_message.lower()

        if any(word in message for word in ["head", "first", "top", "beginning"]):
            return VisualizationRequest(ChartType.DATA_SUMMARY, parameters={"method": "head"})

        if any(word in message for word in ["tail", "last", "bottom", "end"]):
            return VisualizationRequest(ChartType.DATA_SUMMARY, parameters={"method": "tail"})

        if any(word in message for word in ["sample", "random", "few rows"]):
            return VisualizationRequest(ChartType.DATA_SUMMARY, parameters={"method": "sample"})

        # Correlation heatmap: trigger only on explicit correlation/heatmap terms
        if any(word in message for word in ["correlation", "corr", "heatmap"]):
            return VisualizationRequest(ChartType.CORRELATION_HEATMAP)

        if any(word in message for word in ["distribution", "histogram", "spread"]):
            return VisualizationRequest(ChartType.DISTRIBUTION)

        # Scatter plot: require explicit scatter or an "x vs y" phrase
        if ("scatter" in message) or (" vs " in message) or ("against" in message):
            return VisualizationRequest(ChartType.SCATTER)

        if any(word in message for word in ["box plot", "outlier", "quartile"]):
            return VisualizationRequest(ChartType.BOX_PLOT)

        if any(word in message for word in ["missing", "null", "na", "empty"]):
            return VisualizationRequest(ChartType.MISSING_DATA)

        if any(word in message for word in ["kde", "density"]):
            return VisualizationRequest(ChartType.KDE_PLOT)

        if any(word in message for word in ["violin"]):
            return VisualizationRequest(ChartType.VIOLIN_PLOT)

        if any(word in message for word in ["swarm"]):
            return VisualizationRequest(ChartType.SWARM_PLOT)

        if any(word in message for word in ["bar", "bar chart"]):
            return VisualizationRequest(ChartType.BAR_CHART)

        if any(word in message for word in ["line chart", "line plot", "time series", "trend over time"]):
            return VisualizationRequest(ChartType.LINE_CHART)

        if any(word in message for word in ["hexbin"]):
            return VisualizationRequest(ChartType.HEXBIN)

        if any(word in message for word in ["pair plot", "pairplot"]):
            return VisualizationRequest(ChartType.PAIR_PLOT)

        if any(word in message for word in ["joint plot", "jointplot"]):
            return VisualizationRequest(ChartType.JOINT_PLOT)

        # If the user mentions generic "relationship(s)" without specifying a chart,
        # do not force a heatmap here; let the chat layer ask clarifying questions.
        if "relationship" in message or "relationships" in message:
            return None

        return None
