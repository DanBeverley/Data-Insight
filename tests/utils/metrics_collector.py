import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from statistics import mean, median


class TestMetricsCollector:
    def __init__(self, output_dir: str = "monitoring/metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": None,
            "categories": {},
            "failures": [],
            "flaky_tests": [],
            "summary": {}
        }

    def record_test(self, name: str, duration: float, status: str, category: str):
        if category not in self.metrics["categories"]:
            self.metrics["categories"][category] = {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "durations": []
            }

        cat = self.metrics["categories"][category]
        cat["total"] += 1
        cat["durations"].append(duration)

        if status == "passed":
            cat["passed"] += 1
        elif status == "failed":
            cat["failed"] += 1
            self.metrics["failures"].append({
                "name": name,
                "category": category,
                "duration": duration
            })
        elif status == "skipped":
            cat["skipped"] += 1

    def detect_flaky_tests(self, test_history: Dict[str, List[str]]):
        for test_name, outcomes in test_history.items():
            if len(outcomes) < 3:
                continue

            passed_count = outcomes.count("passed")
            failed_count = outcomes.count("failed")

            if passed_count > 0 and failed_count > 0:
                flakiness_score = min(passed_count, failed_count) / len(outcomes)

                if flakiness_score > 0.2:
                    self.metrics["flaky_tests"].append({
                        "name": test_name,
                        "score": flakiness_score,
                        "outcomes": outcomes
                    })

    def calculate_summary(self):
        total_tests = sum(cat["total"] for cat in self.metrics["categories"].values())
        total_passed = sum(cat["passed"] for cat in self.metrics["categories"].values())
        total_failed = sum(cat["failed"] for cat in self.metrics["categories"].values())

        all_durations = []
        for cat in self.metrics["categories"].values():
            all_durations.extend(cat["durations"])

        self.metrics["summary"] = {
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "pass_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            "total_duration": sum(all_durations),
            "avg_duration": mean(all_durations) if all_durations else 0,
            "median_duration": median(all_durations) if all_durations else 0,
            "p95_duration": self._percentile(all_durations, 0.95) if all_durations else 0,
            "p99_duration": self._percentile(all_durations, 0.99) if all_durations else 0
        }

    def _percentile(self, data: List[float], percentile: float) -> float:
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def save_metrics(self, run_id: str = None):
        if run_id:
            self.metrics["run_id"] = run_id

        self.calculate_summary()

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"test_run_{timestamp}.json"

        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        print(f"Metrics saved to {output_path}")
        return output_path

    def print_summary(self):
        self.calculate_summary()

        summary = self.metrics["summary"]
        print("\n" + "="*60)
        print("TEST EXECUTION SUMMARY")
        print("="*60)
        print(f"Total Tests:     {summary['total_tests']}")
        print(f"Passed:          {summary['passed']} ({summary['pass_rate']:.1f}%)")
        print(f"Failed:          {summary['failed']}")
        print(f"Total Duration:  {summary['total_duration']:.2f}s")
        print(f"Avg Duration:    {summary['avg_duration']:.3f}s")
        print(f"P95 Duration:    {summary['p95_duration']:.3f}s")
        print(f"P99 Duration:    {summary['p99_duration']:.3f}s")

        if self.metrics["failures"]:
            print(f"\nFailures: {len(self.metrics['failures'])}")
            for failure in self.metrics["failures"][:5]:
                print(f"  - {failure['name']} ({failure['category']})")

        if self.metrics["flaky_tests"]:
            print(f"\nFlaky Tests: {len(self.metrics['flaky_tests'])}")
            for flaky in self.metrics["flaky_tests"]:
                print(f"  - {flaky['name']} (score: {flaky['score']:.2f})")

        print("="*60 + "\n")

    def compare_with_baseline(self, baseline_path: str) -> Dict[str, Any]:
        if not Path(baseline_path).exists():
            return {"error": "Baseline not found"}

        with open(baseline_path) as f:
            baseline = json.load(f)

        self.calculate_summary()
        current = self.metrics["summary"]

        comparison = {
            "pass_rate_change": current["pass_rate"] - baseline["summary"]["pass_rate"],
            "duration_change": current["total_duration"] - baseline["summary"]["total_duration"],
            "avg_duration_change": current["avg_duration"] - baseline["summary"]["avg_duration"],
            "new_failures": [
                f for f in self.metrics["failures"]
                if not any(bf["name"] == f["name"] for bf in baseline.get("failures", []))
            ],
            "resolved_failures": [
                bf for bf in baseline.get("failures", [])
                if not any(f["name"] == bf["name"] for f in self.metrics["failures"])
            ]
        }

        return comparison


def pytest_configure(config):
    config._metrics_collector = TestMetricsCollector()


def pytest_runtest_logreport(report):
    if report.when == "call":
        collector = report.config._metrics_collector

        category = "unit"
        if "integration" in report.nodeid:
            category = "integration"
        elif "e2e" in report.nodeid:
            category = "e2e"
        elif "security" in report.nodeid:
            category = "security"
        elif "performance" in report.nodeid:
            category = "performance"

        status = "passed" if report.passed else "failed" if report.failed else "skipped"

        collector.record_test(
            name=report.nodeid,
            duration=report.duration,
            status=status,
            category=category
        )


def pytest_sessionfinish(session, exitstatus):
    collector = session.config._metrics_collector
    collector.print_summary()
    collector.save_metrics(run_id=f"ci_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
