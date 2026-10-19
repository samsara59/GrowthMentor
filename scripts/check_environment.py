
"""
Lesson 79: Environment checker.

检查项目文件、Python 版本和基础依赖。
不下载模型、不访问外部 API。
"""

from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "app/ui/streamlit_app.py",
    "app/agent/semantic_evidence.py",
    "app/agent/semantic_nli.py",
    "app/core/model_provider.py",
    "app/assessment/information_gain.py",
    "requirements.txt",
]

REQUIRED_MODULES = [
    "streamlit",
    "langgraph",
    "chromadb",
]


def check_files():
    print("\n[1] Checking project files")

    success = True

    for filename in REQUIRED_FILES:
        path = ROOT / filename
        exists = path.is_file()

        print(
            "[OK]" if exists else "[MISSING]",
            filename,
        )

        if not exists:
            success = False

    return success


def check_python():
    print("\n[2] Checking Python")

    version = sys.version_info

    print("Python:", sys.version.split()[0])
    print("Executable:", sys.executable)

    valid = (
        version.major == 3
        and version.minor == 11
    )

    if not valid:
        print("[WARNING] Recommended: Python 3.11")
    else:
        print("[OK] Python version")

    return valid


def check_dependencies():
    print("\n[3] Checking dependencies")

    success = True

    for module in REQUIRED_MODULES:
        exists = (
            importlib.util.find_spec(module)
            is not None
        )

        print(
            "[OK]" if exists else "[MISSING]",
            module,
        )

        if not exists:
            success = False

    return success


def main():
    results = [
        check_files(),
        check_python(),
        check_dependencies(),
    ]

    print("\n" + "=" * 40)

    if all(results):
        print("Environment check: PASSED")
        return 0

    print("Environment check: FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
