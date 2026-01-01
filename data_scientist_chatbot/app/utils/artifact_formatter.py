from typing import List, Any


def format_artifact_context(artifacts: List[Any], execution_result: Any = None) -> str:
    """
    Formats the list of artifacts into a readable string for the Brain agent.

    Args:
        artifacts: List of artifact objects (or dicts).
        execution_result: Optional execution result object/dict.

    Returns:
        Formatted string describing the artifacts.
    """
    if not artifacts:
        return ""

    # If execution failed, we might still have artifacts, but check success if provided
    if execution_result and hasattr(execution_result, "success") and not execution_result.success:
        # Depending on logic, we might still want to show what was generated before failure
        pass

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

    artifact_lines = []

    def get_attr(obj, attr, default=None):
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    if viz_artifacts:
        artifact_lines.append(f"📊 Visualizations ({len(viz_artifacts)}):")
        for artifact in viz_artifacts[:15]:
            filename = get_attr(artifact, "filename", "unknown.png")
            url = (
                get_attr(artifact, "presigned_url")
                or get_attr(artifact, "cloud_url")
                or get_attr(artifact, "local_path")
            )
            artifact_id = get_attr(artifact, "artifact_id") or filename

            artifact_lines.append(f"  • {filename} (ID: {artifact_id})")
            artifact_lines.append(f"    Path: {url}")
        if len(viz_artifacts) > 15:
            artifact_lines.append(f"  ... and {len(viz_artifacts) - 15} more")

    if model_artifacts:
        artifact_lines.append(f"\n💾 Models ({len(model_artifacts)}):")
        for artifact in model_artifacts[:5]:
            filename = get_attr(artifact, "filename", "unknown.pkl")
            url = (
                get_attr(artifact, "presigned_url")
                or get_attr(artifact, "cloud_url")
                or get_attr(artifact, "local_path")
            )
            artifact_id = get_attr(artifact, "artifact_id") or filename

            artifact_lines.append(f"  • {filename} (ID: {artifact_id})")
            artifact_lines.append(f"    Path: {url}")
        if len(model_artifacts) > 5:
            artifact_lines.append(f"  ... and {len(model_artifacts) - 5} more")

    if dataset_artifacts:
        artifact_lines.append(f"\n📁 Datasets ({len(dataset_artifacts)}):")
        for artifact in dataset_artifacts[:5]:
            filename = get_attr(artifact, "filename", "unknown.csv")
            url = (
                get_attr(artifact, "presigned_url")
                or get_attr(artifact, "cloud_url")
                or get_attr(artifact, "local_path")
            )
            artifact_id = get_attr(artifact, "artifact_id") or filename

            artifact_lines.append(f"  • {filename} (ID: {artifact_id})")
            artifact_lines.append(f"    Path: {url}")
        if len(dataset_artifacts) > 5:
            artifact_lines.append(f"  ... and {len(dataset_artifacts) - 5} more")

    return "\n".join(artifact_lines)
