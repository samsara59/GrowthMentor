
"""
Lesson 80: Final release audit.

检查项目结构、隐私文件、Git状态。
不上传文件，不调用模型，不修改已有代码。
"""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "app/agent/graph_workflow.py",
    "app/agent/semantic_evidence.py",
    "app/agent/semantic_nli.py",
    "app/assessment/information_gain.py",
    "app/core/model_provider.py",
    "app/ui/streamlit_app.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
]

PRIVATE_PATHS = [
    "data",
    "books",
    ".env",
    ".streamlit/secrets.toml",
]


def git_files():
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        print("[WARNING] Git repository unavailable")
        return None

    return [
        name.decode("utf-8")
        for name in result.stdout.split(b"\0")
        if name
    ]


def check_structure():
    print("\n[1] Checking project structure")

    missing = []

    for name in REQUIRED:
        exists = (ROOT / name).is_file()

        print(
            "[OK]" if exists else "[MISSING]",
            name,
        )

        if not exists:
            missing.append(name)

    return len(missing) == 0


def check_private_files():
    print("\n[2] Checking Git tracked files")

    tracked = git_files()

    if tracked is None:
        return False

    unsafe = []

    for filename in tracked:
        path = filename.replace("\\", "/")
        lower = path.lower()

        for private in PRIVATE_PATHS:
            if (
                path == private
                or path.startswith(private + "/")
            ):
                unsafe.append(path)

        if lower.endswith((
            ".sqlite",
            ".sqlite3",
            ".db",
            ".pem",
            ".key",
        )):
            unsafe.append(path)

    unsafe = sorted(set(unsafe))

    if unsafe:
        for item in unsafe:
            print("[PRIVATE FILE]", item)
        return False

    print("[OK] No known private paths tracked")
    return True


def run_tests():
    print("\n[3] Running all unit tests")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
            "-v",
        ],
        cwd=ROOT,
        check=False,
    )

    return result.returncode == 0


def main():
    structure_ok = check_structure()
    privacy_ok = check_private_files()
    tests_ok = run_tests()

    print("\n" + "=" * 50)

    if structure_ok and privacy_ok and tests_ok:
        print("PRELIMINARY RELEASE AUDIT: PASSED")
        print(
            "Manual security and integration "
            "review still required."
        )
        return 0

    print("PRELIMINARY RELEASE AUDIT: FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
