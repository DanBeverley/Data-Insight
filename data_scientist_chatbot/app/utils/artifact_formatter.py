from typing import List, Any


def _filename_to_display_name(filename: str) -> str:
    """Convert artifact filename to human-readable display name.

    Examples:
        correlation_heatmap.png -> Correlation Heatmap
        outlier_detection.html -> Outlier Detection
        numeric_distributions.html -> Numeric Distributions
        pairplot.png -> Pairplot
    """
    name = filename.rsplit(".", 1)[0]
    name = name.replace("_", " ").replace("-", " ")
    words = name.split()
    capitalized = " ".join(word.capitalize() for word in words)
    return capitalized


def format_artifact_context(artifacts: List[Any], execution_result: Any = None) -> str:
    """
    Formats the list of artifacts into a readable string for the Brain agent.
    Provides explicit display names derived from filenames to prevent hallucinated labels.

    Args:
        artifacts: List of artifact objects (or dicts).
        execution_result: Optional execution result object/dict.

    Returns:
        Formatted string describing the artifacts with filename-derived titles.
    """
    if not artifacts:
        return ""

    def get_attr(obj, attr, default=None):
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    viz_artifacts = [
        a
        for a in artifacts
        if getattr(a, "category", "").lower() == "visualization"
        or (isinstance(a, dict) and a.get("category") == "visualization")
    ]
    model_artifacts = [
        a
        for a in artifacts
        if getattr(a, "category", "").lower() == "model" or (isinstance(a, dict) and a.get("category") == "model")
    ]
    dataset_artifacts = [
        a
        for a in artifacts
        if getattr(a, "category", "").lower() == "dataset" or (isinstance(a, dict) and a.get("category") == "dataset")
    ]
    report_artifacts = [
        a
        for a in artifacts
        if getattr(a, "category", "").lower() == "report" or (isinstance(a, dict) and a.get("category") == "report")
    ]

    artifact_lines = ["**GENERATED ARTIFACTS** (use these exact titles in your response):"]

    if viz_artifacts:
        artifact_lines.append(f"\n📊 **Visualizations** ({len(viz_artifacts)}):")
        for artifact in viz_artifacts[:15]:
            filename = get_attr(artifact, "filename", "unknown.png")
            display_name = _filename_to_display_name(filename)
            url = (
                get_attr(artifact, "presigned_url")
                or get_attr(artifact, "cloud_url")
                or get_attr(artifact, "local_path")
                or f"/static/plots/{filename}"
            )
            artifact_lines.append(f"  • **{display_name}** → `{filename}` → {url}")
        if len(viz_artifacts) > 15:
            artifact_lines.append(f"  ... and {len(viz_artifacts) - 15} more")

    if report_artifacts:
        artifact_lines.append(f"\n📄 **Interactive Charts** ({len(report_artifacts)}):")
        for artifact in report_artifacts[:10]:
            filename = get_attr(artifact, "filename", "unknown.html")
            display_name = _filename_to_display_name(filename)
            url = (
                get_attr(artifact, "presigned_url")
                or get_attr(artifact, "cloud_url")
                or get_attr(artifact, "local_path")
                or f"/static/plots/{filename}"
            )
            artifact_lines.append(f"  • **{display_name}** → `{filename}` → {url}")
        if len(report_artifacts) > 10:
            artifact_lines.append(f"  ... and {len(report_artifacts) - 10} more")

    if model_artifacts:
        artifact_lines.append(f"\n💾 **Models** ({len(model_artifacts)}):")
        for artifact in model_artifacts[:5]:
            filename = get_attr(artifact, "filename", "unknown.pkl")
            display_name = _filename_to_display_name(filename)
            artifact_lines.append(f"  • **{display_name}** → `{filename}`")
        if len(model_artifacts) > 5:
            artifact_lines.append(f"  ... and {len(model_artifacts) - 5} more")

    if dataset_artifacts:
        artifact_lines.append(f"\n📁 **Datasets** ({len(dataset_artifacts)}):")
        for artifact in dataset_artifacts[:5]:
            filename = get_attr(artifact, "filename", "unknown.csv")
            display_name = _filename_to_display_name(filename)
            artifact_lines.append(f"  • **{display_name}** → `{filename}`")
        if len(dataset_artifacts) > 5:
            artifact_lines.append(f"  ... and {len(dataset_artifacts) - 5} more")

    artifact_lines.append(
        "\n**IMPORTANT**: When presenting these artifacts, use the exact display names above as section titles."
    )

    return "\n".join(artifact_lines)
