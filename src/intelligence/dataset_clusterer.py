import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import logging

from .domain_detector import DomainDetector
from .profile_cache import ProfileCache

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    filename: str
    df: pd.DataFrame
    columns: Set[str]
    dtypes: Dict[str, str]
    row_count: int
    date_columns: List[str]
    numeric_columns: List[str]
    domain: str
    hash: str


@dataclass
class DatasetCluster:
    name: str
    datasets: List[str]
    relationship: str
    merged_df: Optional[pd.DataFrame] = None
    merge_strategy: str = "concat"


@dataclass
class ClusteringResult:
    registry: Dict[str, DatasetInfo]
    clusters: List[DatasetCluster]
    unified_dataframe: pd.DataFrame
    unified_context: str


class DatasetClusterer:
    def __init__(self):
        self.domain_detector = DomainDetector()
        self.profile_cache = ProfileCache()
        self.similarity_threshold = 0.5

    def process_datasets(self, datasets: Dict[str, pd.DataFrame]) -> ClusteringResult:
        registry = self._build_registry(datasets)
        clusters = self._cluster_datasets(registry)
        unified_df = self._merge_clusters(clusters, registry)
        context = self._build_context(registry, clusters, unified_df)

        return ClusteringResult(
            registry=registry, clusters=clusters, unified_dataframe=unified_df, unified_context=context
        )

    def _build_registry(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, DatasetInfo]:
        registry = {}

        for filename, df in datasets.items():
            date_cols = self._detect_date_columns(df)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            domain_matches = self.domain_detector.detect_domain(df)
            domain = domain_matches[0].domain if domain_matches else "general"

            dataset_hash = self.profile_cache.compute_dataset_hash(df)

            registry[filename] = DatasetInfo(
                filename=filename,
                df=df,
                columns=set(df.columns),
                dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
                row_count=len(df),
                date_columns=date_cols,
                numeric_columns=numeric_cols,
                domain=domain,
                hash=dataset_hash,
            )

            logger.info(f"[CLUSTERER] Registered {filename}: {len(df)} rows, domain={domain}")

        return registry

    def _detect_date_columns(self, df: pd.DataFrame) -> List[str]:
        date_cols = []
        for col in df.columns:
            if df[col].dtype == "datetime64[ns]":
                date_cols.append(col)
            elif df[col].dtype == "object":
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    try:
                        pd.to_datetime(sample, errors="raise")
                        date_cols.append(col)
                    except:
                        pass
            if col.upper() in ["DATE", "DATETIME", "TIME", "TIMESTAMP", "YEAR", "MONTH", "DAY"]:
                if col not in date_cols:
                    date_cols.append(col)
        return date_cols

    def _compute_schema_similarity(self, info1: DatasetInfo, info2: DatasetInfo) -> float:
        cols1 = info1.columns
        cols2 = info2.columns

        if not cols1 or not cols2:
            return 0.0

        intersection = len(cols1 & cols2)
        union = len(cols1 | cols2)

        jaccard = intersection / union if union > 0 else 0.0

        domain_bonus = 0.2 if info1.domain == info2.domain and info1.domain != "general" else 0.0

        shared_date = bool(set(info1.date_columns) & set(info2.date_columns))
        date_bonus = 0.15 if shared_date else 0.0

        return min(1.0, jaccard + domain_bonus + date_bonus)

    def _cluster_datasets(self, registry: Dict[str, DatasetInfo]) -> List[DatasetCluster]:
        if len(registry) == 1:
            filename = list(registry.keys())[0]
            return [
                DatasetCluster(
                    name=registry[filename].domain, datasets=[filename], relationship="single", merge_strategy="none"
                )
            ]

        filenames = list(registry.keys())
        clustered = set()
        clusters = []

        for i, name1 in enumerate(filenames):
            if name1 in clustered:
                continue

            info1 = registry[name1]
            cluster_members = [name1]
            clustered.add(name1)

            for name2 in filenames[i + 1 :]:
                if name2 in clustered:
                    continue

                info2 = registry[name2]
                similarity = self._compute_schema_similarity(info1, info2)

                if similarity >= self.similarity_threshold:
                    cluster_members.append(name2)
                    clustered.add(name2)
                    logger.info(f"[CLUSTERER] Grouped {name1} + {name2} (similarity={similarity:.2f})")

            if len(cluster_members) > 1:
                shared_cols = registry[cluster_members[0]].columns
                for m in cluster_members[1:]:
                    shared_cols = shared_cols & registry[m].columns

                if shared_cols:
                    merge_strategy = "merge"
                    relationship = "related"
                else:
                    merge_strategy = "concat"
                    relationship = "same_domain"
            else:
                merge_strategy = "none"
                relationship = "independent"

            clusters.append(
                DatasetCluster(
                    name=info1.domain,
                    datasets=cluster_members,
                    relationship=relationship,
                    merge_strategy=merge_strategy,
                )
            )

        logger.info(f"[CLUSTERER] Created {len(clusters)} clusters from {len(registry)} datasets")
        return clusters

    def _merge_clusters(self, clusters: List[DatasetCluster], registry: Dict[str, DatasetInfo]) -> pd.DataFrame:
        merged_dfs = []

        for cluster in clusters:
            if len(cluster.datasets) == 1:
                df = registry[cluster.datasets[0]].df.copy()
                df["_source_file"] = cluster.datasets[0]
                cluster.merged_df = df
                merged_dfs.append(df)
            else:
                dfs_to_merge = []
                for filename in cluster.datasets:
                    df = registry[filename].df.copy()
                    df["_source_file"] = filename
                    dfs_to_merge.append(df)

                if cluster.merge_strategy == "merge":
                    date_cols = set()
                    for filename in cluster.datasets:
                        date_cols.update(registry[filename].date_columns)

                    common_date = list(date_cols)[0] if date_cols else None

                    if common_date:
                        merged = dfs_to_merge[0]
                        for df in dfs_to_merge[1:]:
                            merged = pd.merge(
                                merged, df, on=common_date, how="outer", suffixes=("", f'_{df["_source_file"].iloc[0]}')
                            )
                        cluster.merged_df = merged
                        merged_dfs.append(merged)
                    else:
                        concat_df = pd.concat(dfs_to_merge, ignore_index=True)
                        cluster.merged_df = concat_df
                        merged_dfs.append(concat_df)
                else:
                    concat_df = pd.concat(dfs_to_merge, ignore_index=True)
                    cluster.merged_df = concat_df
                    merged_dfs.append(concat_df)

        if len(merged_dfs) == 1:
            return merged_dfs[0]

        unified = pd.concat(merged_dfs, ignore_index=True)
        return unified

    def _build_context(
        self, registry: Dict[str, DatasetInfo], clusters: List[DatasetCluster], unified_df: pd.DataFrame
    ) -> str:
        lines = []
        lines.append("MULTI-DATASET ANALYSIS CONTEXT")
        lines.append("=" * 50)
        lines.append("")
        lines.append(f"PROCESSED {len(registry)} DATASETS → {len(clusters)} CLUSTER(S)")
        lines.append("")

        for i, cluster in enumerate(clusters, 1):
            lines.append(f"CLUSTER {i}: {cluster.name.upper()}")
            lines.append(f"├── Sources: {', '.join(cluster.datasets)}")
            lines.append(f"├── Relationship: {cluster.relationship.upper()}")
            lines.append(f"├── Merge Strategy: {cluster.merge_strategy}")

            if cluster.merged_df is not None:
                lines.append(
                    f"├── Merged Shape: {cluster.merged_df.shape[0]} rows × {cluster.merged_df.shape[1]} columns"
                )
                cols = cluster.merged_df.columns.tolist()
                cols_display = cols[:10]
                if len(cols) > 10:
                    cols_display.append(f"... +{len(cols)-10} more")
                lines.append(f"└── Columns: {cols_display}")
            lines.append("")

        lines.append(f"UNIFIED DATAFRAME: {unified_df.shape[0]} rows × {unified_df.shape[1]} columns")
        lines.append(f"ALL COLUMNS: {unified_df.columns.tolist()}")
        lines.append("")
        lines.append("NOTE: _source_file column tracks which original file each row came from.")

        return "\n".join(lines)
