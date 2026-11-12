"""任务三-1：RAG 问答系统人工评估脚本。

评估方式（1-5 分制，两个维度）：
    准确性 —— 生成答案是否与标准答案事实一致、有无错误或幻觉；
    完整性 —— 是否完整覆盖标准答案的要点，有无明显遗漏。

工作流程：
    1. 固定随机种子抽取 5 个测试样本（保证结果可复现）；
    2. 走完整 RAG 链路生成答案，并记录检索来源与相似度；
    3. 由生成模型按同一评分标准给出“LLM 辅助评分”（仅供参考）；
    4. 输出评估明细表 outputs/evaluation_report.csv（含人工评分空列），
       人工填写 human_accuracy / human_completeness 后，
       运行 ``python evaluate.py --summary`` 计算人工平均得分；
    5. 完整结果另存 outputs/evaluation_detail.json。

    python evaluate.py            # 生成 5 个样本并自动辅助评分
    python evaluate.py --summary  # 汇总人工填写后的平均得分
"""
from __future__ import annotations

import argparse
import json
import re
import random

import pandas as pd

from app import config
from app.data_loader import load_and_clean
from app.service import QAService

#: 评分标准（人工与 LLM 辅助评分共用）
SCORE_RUBRIC = """请从两个维度对【生成答案】按 1-5 分制打分：
- 准确性(accuracy)：与【标准答案】事实是否一致，有无错误/幻觉。
  5=完全一致；4=基本一致仅措辞不同；3=大体正确有小瑕疵；
  2=存在明显错误；1=答非所问或严重错误。
- 完整性(completeness)：是否覆盖标准答案要点。
  5=要点完整且有合理展开；4=覆盖主要要点；3=覆盖部分要点；
  2=遗漏较多；1=基本未回答。
只输出一行 JSON：{"accuracy": 分数, "completeness": 分数}"""

JUDGE_PROMPT_TEMPLATE = """【问题】
{question}

【标准答案】
{reference}

【生成答案】
{prediction}

{rubric}"""


def sample_testset(n: int = 5, seed: int = 42) -> pd.DataFrame:
    """固定随机种子从清洗后数据中抽取 n 个测试样本。"""
    dataframe = load_and_clean()
    return dataframe.sample(n=n, random_state=seed).reset_index(drop=True)


def _parse_scores(text: str) -> tuple[int | None, int | None]:
    """从评分模型输出中容错解析 accuracy / completeness 分数。"""
    match = re.search(
        r"\{[^{}]*accuracy[^{}]*completeness[^{}]*\}", text, re.S
    )
    if not match:
        return None, None
    block = match.group(0)
    acc = re.search(r"accuracy['\"]?\s*[:：]\s*([1-5])", block)
    comp = re.search(r"completeness['\"]?\s*[:：]\s*([1-5])", block)
    return (
        int(acc.group(1)) if acc else None,
        int(comp.group(1)) if comp else None,
    )


def auto_judge(service: QAService, question: str, reference: str,
               prediction: str) -> tuple[int | None, int | None, str]:
    """调用生成模型按评分标准对单条样本打分。"""
    messages = [
        {"role": "system", "content": "你是严格的医疗问答评分员，只输出 JSON。"},
        {"role": "user", "content": JUDGE_PROMPT_TEMPLATE.format(
            question=question, reference=reference,
            prediction=prediction, rubric=SCORE_RUBRIC)},
    ]
    raw = service.llm.generate(messages, max_new_tokens=64)
    accuracy, completeness = _parse_scores(raw)
    return accuracy, completeness, raw


def run_evaluation() -> None:
    """生成 5 个评估样本，执行 RAG 问答并完成辅助评分与落盘。"""
    config.ensure_dirs()
    service = QAService()
    testset = sample_testset(5)

    records: list[dict] = []
    print("=" * 70)
    print("人工评估：随机抽取 5 个测试样本（seed=42）")
    print("=" * 70)

    for i, row in testset.iterrows():
        question = row["question"]
        reference = row["answer"]
        print(f"\n[{i + 1}/5] 问题：{question}")

        result = service.answer(question, session_id=f"eval-{i}",
                                use_history=False, max_new_tokens=256)
        prediction = result["answer"]
        sources = ",".join(result["sources"])

        accuracy, completeness, raw_judge = auto_judge(
            service, question, reference, prediction
        )
        print(f"  检索来源：{sources}")
        print(f"  标准答案：{reference}")
        print(f"  生成答案：{prediction}")
        print(f"  LLM 辅助评分：准确性={accuracy} 完整性={completeness}")

        records.append({
            "id": i + 1,
            "disease": row["disease"],
            "question": question,
            "reference_answer": reference,
            "generated_answer": prediction,
            "sources": sources,
            "auto_accuracy": accuracy,
            "auto_completeness": completeness,
            "human_accuracy": "",   # 人工评分列，1-5 分
            "human_completeness": "",
            "judge_raw": raw_judge,
        })

    # 保存完整 JSON 明细
    detail_path = config.OUTPUT_DIR / "evaluation_detail.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # 保存人工评分 CSV（不含 judge_raw 调试字段）
    report = pd.DataFrame(records).drop(columns=["judge_raw"])
    report_path = config.OUTPUT_DIR / "evaluation_report.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    # 输出 LLM 辅助评分均分
    auto_acc = [r["auto_accuracy"] for r in records if r["auto_accuracy"]]
    auto_comp = [r["auto_completeness"] for r in records
                 if r["auto_completeness"]]
    print("\n" + "=" * 70)
    print("评估汇总（LLM 辅助评分，仅供参考）")
    print("=" * 70)
    if auto_acc and auto_comp:
        print(f"准确性平均得分：{sum(auto_acc) / len(auto_acc):.2f} / 5")
        print(f"完整性平均得分：{sum(auto_comp) / len(auto_comp):.2f} / 5")
        print(f"综合平均得分："
              f"{(sum(auto_acc) + sum(auto_comp)) / (len(auto_acc) + len(auto_comp)):.2f} / 5")
    print(f"\n评估明细：{detail_path}")
    print(f"人工评分表：{report_path}")
    print("请在 CSV 中填写 human_accuracy / human_completeness（1-5），")
    print("然后运行：python evaluate.py --summary 查看人工平均得分。")


def summarize_human_scores() -> None:
    """读取人工填写完成的评分 CSV，计算准确性 / 完整性平均得分。"""
    report_path = config.OUTPUT_DIR / "evaluation_report.csv"
    if not report_path.exists():
        raise FileNotFoundError(
            f"未找到评估报告 {report_path}，请先运行 python evaluate.py"
        )

    report = pd.read_csv(report_path, encoding="utf-8-sig")
    human_acc = pd.to_numeric(report["human_accuracy"], errors="coerce").dropna()
    human_comp = pd.to_numeric(
        report["human_completeness"], errors="coerce"
    ).dropna()

    if len(human_acc) == 0 or len(human_comp) == 0:
        print("尚未读取到有效的人工评分，请先在 "
              f"{report_path} 中填写 human_accuracy / human_completeness。")
        return

    print("=" * 70)
    print("人工评估汇总（1-5 分制）")
    print("=" * 70)
    print(f"已评分样本数：准确性 {len(human_acc)} 条，完整性 {len(human_comp)} 条")
    print(f"准确性平均得分：{human_acc.mean():.2f} / 5")
    print(f"完整性平均得分：{human_comp.mean():.2f} / 5")
    print(f"综合平均得分："
          f"{(human_acc.sum() + human_comp.sum()) / (len(human_acc) + len(human_comp)):.2f} / 5")


def main() -> None:
    parser = argparse.ArgumentParser(description="医疗 RAG 问答系统评估")
    parser.add_argument(
        "--summary", action="store_true",
        help="不重新生成，仅汇总 evaluation_report.csv 中的人工评分",
    )
    args = parser.parse_args()

    if args.summary:
        summarize_human_scores()
    else:
        run_evaluation()


if __name__ == "__main__":
    main()
