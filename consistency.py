#!/usr/bin/env python3
"""
Consistency evaluation script for measuring engine benchmark variance.

This script runs the same submissions multiple times through the engine
and measures variance in the benchmark results (runtime and GFLOPS).

Usage:
    python consistency.py <modal_endpoint_url> [--runs N] [--seed S] [--problems PROBLEM1,PROBLEM2]
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
from tqdm import tqdm


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    avg_runtime_ms: Optional[float] = None
    testcase_runtimes_ms: Optional[Dict[int, float]] = None  # Dict mapping test_id to runtime_ms
    status: str = "UNKNOWN"
    error: Optional[str] = None


@dataclass
class SubmissionConfig:
    """Configuration for a test submission."""
    problem_slug: str
    problem_def: str
    solution_code: str
    language: str  # "cuda" or "triton"
    gpu: str = "T4"
    dtype: str = "float32"


@dataclass
class ConsistencyStats:
    """Statistics for consistency measurements."""
    results: List[BenchmarkResult] = field(default_factory=list)

    @property
    def runtime_values(self) -> List[float]:
        return [r.avg_runtime_ms for r in self.results if r.avg_runtime_ms is not None]
    
    @property
    def per_testcase_runtimes(self) -> Dict[int, List[float]]:
        """Aggregate all individual runtime measurements per testcase across all runs."""
        testcase_data: Dict[int, List[float]] = defaultdict(list)
        for result in self.results:
            if result.testcase_runtimes_ms:
                # result.testcase_runtimes_ms is a dict mapping test_id to runtime_ms
                for test_id, runtime_ms in result.testcase_runtimes_ms.items():
                    testcase_data[test_id].append(runtime_ms)
        return testcase_data

    def calculate_stats(self) -> Dict:
        """Calculate statistics for this submission."""
        runtime_vals = self.runtime_values
        testcase_data = self.per_testcase_runtimes

        stats = {
            "total_runs": len(self.results),
            "successful_runs": len(runtime_vals),
            "failed_runs": len(self.results) - len(runtime_vals),
        }

        def calc_stats(values: List[float], name: str) -> Optional[Dict]:
            if not values:
                return None
            stdev_val = statistics.stdev(values) if len(values) > 1 else 0.0
            mean_val = statistics.mean(values)
            return {
                "mean": mean_val,
                "median": statistics.median(values),
                "stdev": stdev_val,
                "min": min(values),
                "max": max(values),
                "cv": stdev_val / mean_val if mean_val > 0 and len(values) > 1 else 0.0,
                "values": values,
            }

        stats["runtime"] = calc_stats(runtime_vals, "runtime")
        
        # Calculate per-testcase statistics
        stats["per_testcase"] = {}
        for testcase_idx, runtimes in sorted(testcase_data.items()):
            testcase_stats = calc_stats(runtimes, f"testcase_{testcase_idx}")
            if testcase_stats:
                stats["per_testcase"][testcase_idx] = testcase_stats
        
        return stats


def load_submissions(problem_dir: Optional[Path] = None, submissions_dir: Optional[Path] = None) -> Dict[str, SubmissionConfig]:
    """Load test submissions from the submissions directory."""
    if submissions_dir is None:
        submissions_dir = Path(__file__).parent / "submissions"
    if problem_dir is None:
        problem_dir = Path(__file__).parent / "problems"

    if not problem_dir.exists() or not submissions_dir.exists():
        return {}

    submissions = {}
    problem_files = {}

    # Load problem definitions
    for problem_file in problem_dir.glob("*.py"):
        slug = problem_file.stem
        problem_files[slug] = problem_file.read_text()

    # Load CUDA solutions
    for solution_file in submissions_dir.glob("*.cu"):
        slug = solution_file.stem
        if slug in problem_files:
            submissions[f"{slug}_cuda"] = SubmissionConfig(
                problem_slug=slug,
                problem_def=problem_files[slug],
                solution_code=solution_file.read_text(),
                language="cuda",
            )

    # Load Triton solutions
    for solution_file in submissions_dir.glob("*.py"):
        slug = solution_file.stem
        if slug in problem_files:
            submissions[f"{slug}_triton"] = SubmissionConfig(
                problem_slug=slug,
                problem_def=problem_files[slug],
                solution_code=solution_file.read_text(),
                language="python",  # Triton uses "python" as language identifier
            )

    return submissions


async def run_benchmark(
    session: aiohttp.ClientSession,
    endpoint: str,
    submission: SubmissionConfig,
    gpu: str,
) -> BenchmarkResult:
    """Run a single benchmark through the modal endpoint."""
    url = f"{endpoint.rstrip('/')}/benchmark-{gpu}"
    payload = {
        "solution_code": submission.solution_code,
        "problem": submission.problem_slug,
        "problem_def": submission.problem_def,
        "dtype": submission.dtype,
        "language": submission.language,
    }

    result = BenchmarkResult(status="RUNNING")

    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                result.status = "ERROR"
                result.error = f"HTTP {response.status}: {await response.text()}"
                return result

            buffer = b""
            avg_runtime_ms = None
            testcase_runtimes_ms = None
            final_status = None
            error_message = None

            async for chunk in response.content.iter_chunked(8192):
                buffer += chunk
                while b"\n" in buffer:
                    line_bytes, buffer = buffer.split(b"\n", 1)
                    line_str = line_bytes.decode("utf-8", errors="ignore").strip()

                    if not line_str.startswith("data: "):
                        continue

                    try:
                        data_str = line_str[6:]
                        if not data_str:
                            continue
                        event_data = json.loads(data_str)
                        status = event_data.get("status", "")

                        if status == "BENCHMARKED":
                            avg_runtime_ms = event_data.get("avg_runtime_ms")
                            testcase_runtimes_ms = None
                            if "test_results" in event_data:
                                test_results = event_data.get("test_results", [])
                                testcase_runtimes_ms = {}
                                for test_result in test_results:
                                    test_id = test_result.get("test_id")
                                    runtime_ms = test_result.get("runtime_ms")
                                    if test_id is not None and runtime_ms is not None:
                                        testcase_runtimes_ms[test_id] = runtime_ms
                            final_status = "BENCHMARKED"
                        elif status in ["COMPILE_ERROR", "RUNTIME_ERROR", "ERROR", "WRONG_ANSWER"]:
                            final_status = status
                            # Capture all possible error details
                            error_parts = []
                            if event_data.get("message"):
                                error_parts.append(event_data["message"])
                            if event_data.get("details"):
                                error_parts.append(event_data["details"])
                            if event_data.get("error"):
                                error_parts.append(event_data["error"])
                            if event_data.get("traceback"):
                                error_parts.append(f"Traceback: {event_data['traceback']}")
                            error_message = "\n".join(error_parts) if error_parts else str(event_data)
                            break
                    except json.JSONDecodeError:
                        continue

            result.status = final_status or "UNKNOWN"
            result.avg_runtime_ms = avg_runtime_ms
            result.testcase_runtimes_ms = testcase_runtimes_ms
            result.error = error_message

    except Exception as e:
        result.status = "ERROR"
        result.error = str(e)

    return result


async def evaluate_consistency(
    endpoint: str,
    submissions: List[SubmissionConfig],
    num_runs: int = 10,
    seed: Optional[int] = None,
    gpu: str = "T4",
) -> Dict:
    """Evaluate consistency by running each submission multiple times."""
    if seed is not None:
        random.seed(seed)

    all_runs = [(s, i) for s in submissions for i in range(num_runs)]
    random.shuffle(all_runs)

    print(f"Running {len(submissions)} submissions × {num_runs} runs = {len(all_runs)} total benchmarks")
    if seed is not None:
        print(f"Seed: {seed}")
    
    print(f"\nRun order:")
    for idx, (submission, run_idx) in enumerate(all_runs, 1):
        key = f"{submission.problem_slug}_{submission.language}"
        print(f"  {idx}. {key} (run {run_idx + 1}/{num_runs})")
    print()

    results_by_submission: Dict[str, ConsistencyStats] = defaultdict(ConsistencyStats)

    async with aiohttp.ClientSession() as session:
        pbar = tqdm(total=len(all_runs), desc="Running benchmarks")
        
        for idx, (submission, run_idx) in enumerate(all_runs, 1):
            key = f"{submission.problem_slug}_{submission.language}"
            pbar.set_description(f"{key} (run {run_idx + 1}/{num_runs})")

            result = await run_benchmark(session, endpoint, submission, gpu)
            results_by_submission[key].results.append(result)

            if result.status == "BENCHMARKED":
                runtime_str = f"{result.avg_runtime_ms:.2f}ms" if result.avg_runtime_ms else "N/A"
                pbar.set_postfix(status="✓", runtime=runtime_str)
            else:
                error_msg = result.error or "Unknown error"
                error_preview = str(error_msg).split("\n")[0][:100]
                pbar.set_postfix(status=f"✗ {result.status}", error=error_preview[:50])

            pbar.update(1)
            await asyncio.sleep(0.5)
        
        pbar.close()

    # Calculate statistics
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    summary = {
        "endpoint": endpoint,
        "num_runs_per_submission": num_runs,
        "seed": seed,
        "gpu": gpu,
        "submissions": {},
    }

    for key, stats_obj in results_by_submission.items():
        stats = stats_obj.calculate_stats()
        summary["submissions"][key] = stats

        print(f"\n{key}:")
        print(f"  Runs: {stats['successful_runs']}/{stats['total_runs']} successful")
        
        if stats["runtime"]:
            rt = stats["runtime"]
            print(f"  Overall Runtime: {rt['mean']:.4f}ms ± {rt['stdev']:.4f}ms (CV: {rt['cv']:.2%}, range: [{rt['min']:.4f}, {rt['max']:.4f}])")
        
        # Print per-testcase statistics
        if stats.get("per_testcase"):
            print(f"  Per-Testcase Runtime:")
            for testcase_idx, tc_stats in sorted(stats["per_testcase"].items()):
                print(f"    Testcase {testcase_idx}: {tc_stats['mean']:.4f}ms ± {tc_stats['stdev']:.4f}ms (CV: {tc_stats['cv']:.2%}, range: [{tc_stats['min']:.4f}, {tc_stats['max']:.4f}])")
        
        failed_results = [r for r in stats_obj.results if r.status != "BENCHMARKED"]
        if failed_results:
            print(f"  Errors ({len(failed_results)}):")
            for failed in failed_results:
                error_msg = failed.error or "Unknown error"
                error_first_line = str(error_msg).split("\n")[0]
                print(f"    {failed.status}: {error_first_line[:150]}")

    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate consistency of engine benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python consistency.py https://your-modal-endpoint.modal.run
  python consistency.py https://your-modal-endpoint.modal.run --runs 20 --seed 42
  python consistency.py https://your-modal-endpoint.modal.run --problems relu_cuda,relu_triton
        """,
    )
    parser.add_argument("endpoint", help="Modal endpoint URL (e.g., https://xxx.modal.run)")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per submission (default: 10)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for ordering")
    parser.add_argument("--problems", type=str, default=None, help="Comma-separated list of problem keys")
    parser.add_argument("--gpu", type=str, default="T4", help="GPU type to use (default: T4)")
    parser.add_argument("--output", type=str, default=None, help="Output file for JSON results")

    args = parser.parse_args()

    # Load and filter submissions
    test_submissions = load_submissions()
    
    submissions = list(test_submissions.values())

    if args.problems:
        problem_keys = [k.strip() for k in args.problems.split(",")]
        submissions = [test_submissions[k] for k in problem_keys if k in test_submissions]
        if not submissions:
            print(f"Error: No valid problem keys found. Available: {list(test_submissions.keys())}")
            sys.exit(1)

    if not submissions:
        print("Error: No test submissions found. Add files to submissions/ directory.")
        sys.exit(1)

    # Run evaluation
    summary = asyncio.run(
        evaluate_consistency(
            endpoint=args.endpoint,
            submissions=submissions,
            num_runs=args.runs,
            seed=args.seed,
            gpu=args.gpu,
        )
    )

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
