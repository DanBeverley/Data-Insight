import json
import sys
from pathlib import Path
from typing import Dict, Any


class PerformanceRegressionDetector:
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold
        self.baseline_path = Path(__file__).parent / "baseline_metrics.json"
        self.baseline = self._load_baseline()

    def _load_baseline(self) -> Dict[str, Any]:
        if not self.baseline_path.exists():
            return {}

        with open(self.baseline_path) as f:
            return json.load(f)

    def check_regression(self, current_metrics: Dict[str, float]) -> bool:
        regressions = []

        for metric, baseline_value in self.baseline.items():
            if metric in ["version", "last_updated", "environment"]:
                continue

            if metric not in current_metrics:
                continue

            current_value = current_metrics[metric]

            if isinstance(baseline_value, (int, float)) and isinstance(current_value, (int, float)):
                change = (current_value - baseline_value) / baseline_value

                if change > self.threshold:
                    regressions.append({
                        "metric": metric,
                        "baseline": baseline_value,
                        "current": current_value,
                        "change_pct": change * 100
                    })

        if regressions:
            print("⚠️  Performance Regressions Detected:")
            for reg in regressions:
                print(f"  - {reg['metric']}: {reg['baseline']} → {reg['current']} (+{reg['change_pct']:.1f}%)")
            return False

        print("✓ No performance regressions detected")
        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python regression_detector.py <current_metrics.json>")
        sys.exit(1)

    current_metrics_path = Path(sys.argv[1])
    if not current_metrics_path.exists():
        print(f"Error: {current_metrics_path} not found")
        sys.exit(1)

    with open(current_metrics_path) as f:
        current_metrics = json.load(f)

    detector = PerformanceRegressionDetector(threshold=0.15)
    passed = detector.check_regression(current_metrics)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
