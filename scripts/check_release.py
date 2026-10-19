
"""
Lesson 79: Release readiness checks.

执行环境检查、独立模块测试以及
完整测试发现。
"""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run_command(title, arguments):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        check=False,
    )

    return result.returncode == 0


def main():
    checks = [
        (
            "Environment",
            ["-m", "scripts.check_environment"],
        ),
        (
            "Semantic Evidence",
            [
                "-m", "unittest", "discover",
                "-s", "tests",
                "-p", "test_semantic_evidence.py",
                "-v",
            ],
        ),
        (
            "Semantic NLI",
            [
                "-m", "unittest", "discover",
                "-s", "tests",
                "-p", "test_semantic_nli.py",
                "-v",
            ],
        ),
        (
            "Model Provider",
            [
                "-m", "unittest", "discover",
                "-s", "tests",
                "-p", "test_model_provider.py",
                "-v",
            ],
        ),
        (
            "Information Gain",
            [
                "-m", "unittest", "discover",
                "-s", "tests",
                "-p", "test_information_gain.py",
                "-v",
            ],
        ),
        (
            "Full Test Suite",
            [
                "-m", "unittest", "discover",
                "-s", "tests",
                "-p", "test_*.py",
                "-v",
            ],
        ),
    ]

    failed = []

    for title, command in checks:
        if not run_command(title, command):
            failed.append(title)

    print("\n" + "=" * 60)

    if failed:
        print("RELEASE CHECK: FAILED")
        for name in failed:
            print("Failed:", name)
        return 1

    print("RELEASE CHECK: PASSED")
    print("Automated checks completed.")
    print(
        "Manual integration, security and "
        "installation checks are still required."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
