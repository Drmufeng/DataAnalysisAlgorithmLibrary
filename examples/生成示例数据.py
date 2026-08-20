"""生成不包含真实用户信息的数据概览示例文件。"""

from pathlib import Path

import pandas as pd


def main() -> None:
    """在 examples/data 下生成可重复使用的 Parquet 数据。"""

    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(
        {
            "学号": [1001, 1002, 1003, 1004, 1005],
            "性别": ["女", "男", "女", "男", "女"],
            "成绩": [86.5, 91.0, None, 77.5, 88.0],
            "提交时间": pd.to_datetime(
                ["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05"]
            ),
        }
    )
    data.to_parquet(output_dir / "学生成绩.parquet", index=False)


if __name__ == "__main__":
    main()
