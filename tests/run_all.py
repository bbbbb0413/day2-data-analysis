"""pytest 없이 tests/의 모든 테스트를 실행한다.

자동화 환경에 pytest가 없을 수 있어 대체 경로를 남겨 둔다.
pytest가 설치돼 있으면 `.venv/bin/python -m pytest tests/ -v`를 쓰는 편이 낫다.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def main() -> int:
    """test_*.py를 모두 임포트해 test_ 함수를 실행한다."""
    failed: list[str] = []
    total = 0

    for path in sorted(HERE.glob("test_*.py")):
        module = importlib.import_module(path.stem)
        names = sorted(n for n in vars(module)
                       if n.startswith("test_") and callable(getattr(module, n)))
        if not names:
            continue
        print(f"\n[{path.name}]")
        for name in names:
            total += 1
            try:
                getattr(module, name)()
            except Exception as e:
                failed.append(f"{path.stem}::{name}")
                print(f"  FAIL  {name}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=2)
            else:
                print(f"  PASS  {name}")

    print(f"\n{total - len(failed)}/{total} 통과")
    for name in failed:
        print(f"  실패: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
