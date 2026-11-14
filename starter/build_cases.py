"""Build the 24-question agent benchmark (cases.json) from converted Day 6 questions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

UNANSWERABLE_CASES = [
    {
        "id": "u01",
        "split": "dev",
        "question": (
            "Theo Nghị định 13/2023/NĐ-CP, mức phạt tiền tối đa tính theo triệu đồng "
            "hoặc phần trăm doanh thu đối với hành vi mua, bán dữ liệu cá nhân trái phép là bao nhiêu?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "Đã rà soát toàn bộ 36 trang nội dung Nghị định 13/2023/NĐ-CP: Điều 3 khoản 4 nghiêm cấm "
            "mua bán dữ liệu cá nhân và Điều 4 (trang 5) chỉ nêu chung có thể bị xử lý kỷ luật, "
            "xử phạt vi phạm hành chính hoặc xử lý hình sự theo quy định; văn bản không quy định "
            "mức phạt tiền cụ thể hay tỷ lệ % doanh thu."
        ),
    },
    {
        "id": "u02",
        "split": "dev",
        "question": (
            "Mức lệ phí thẩm định Hồ sơ đánh giá tác động xử lý dữ liệu cá nhân "
            "và Hồ sơ chuyển dữ liệu ra nước ngoài phải nộp cho Bộ Công an là bao nhiêu?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "Đã kiểm tra toàn văn Điều 24, Điều 25 (trang 19-22) và Điều 31 (trang 25): Nghị định chỉ "
            "quy định việc gửi 01 bản chính hồ sơ theo Mẫu số 04/06 và nguồn kinh phí ngân sách hoặc "
            "tự bố trí, hoàn toàn không có điều khoản quy định mức phí/lệ phí thẩm định hồ sơ."
        ),
    },
    {
        "id": "u03",
        "split": "test",
        "question": (
            "Theo Điều 28 Nghị định 13/2023/NĐ-CP, nhân sự phụ trách bảo vệ dữ liệu cá nhân "
            "bắt buộc phải có chứng chỉ hành nghề hoặc bằng cấp tối thiểu nào?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "Đã kiểm tra Điều 28 (trang 23-24) và Điều 30 (trang 24-25): văn bản chỉ yêu cầu chỉ định "
            "bộ phận và nhân sự phụ trách bảo vệ dữ liệu cá nhân nhạy cảm và trao đổi thông tin với "
            "Cơ quan chuyên trách, không quy định yêu cầu chứng chỉ hành nghề hay bằng cấp tối thiểu."
        ),
    },
    {
        "id": "u04",
        "split": "test",
        "question": (
            "Hồ sơ đánh giá tác động xử lý dữ liệu cá nhân có thời hạn hiệu lực tối đa bao nhiêu năm "
            "trước khi tổ chức bắt buộc phải làm thủ tục gia hạn định kỳ?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "Đã kiểm tra toàn văn Điều 24 (trang 19-21) và Điều 43 (trang 29): Điều 24 khoản 6 chỉ "
            "yêu cầu cập nhật, bổ sung hồ sơ khi có sự thay đổi về nội dung đã gửi Bộ Công an, "
            "không có quy định thời hạn hiệu lực theo số năm hay thủ tục gia hạn định kỳ."
        ),
    },
]


def build_cases(questions_path: Path, outputs: list[Path]) -> list[dict]:
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    cases: list[dict] = []
    for q in questions:
        item = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer",
            "expected_answer": q["expected_answer"],
            "gold": q["gold"],
        }
        if "filter" in q and q["filter"] is not None:
            item["filter"] = q["filter"]
        cases.append(item)

    cases.extend(UNANSWERABLE_CASES)
    payload = json.dumps(cases, ensure_ascii=False, indent=2) + "\n"
    for out_path in outputs:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh bộ 24 câu cases.json (gồm 4 câu thiếu evidence)")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("starter/data/converted/questions.json"),
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        nargs="+",
        default=[
            Path("starter/data/converted/cases.json"),
            Path("data/converted/cases.json"),
        ],
    )
    args = parser.parse_args()
    cases = build_cases(args.questions, args.outputs)
    dev_ans = sum(1 for c in cases if c["split"] == "dev" and c["answerable"])
    dev_unans = sum(1 for c in cases if c["split"] == "dev" and not c["answerable"])
    test_ans = sum(1 for c in cases if c["split"] == "test" and c["answerable"])
    test_unans = sum(1 for c in cases if c["split"] == "test" and not c["answerable"])
    print(
        f"Đã tạo {len(cases)} cases: "
        f"dev={dev_ans}+{dev_unans} ({dev_ans + dev_unans}), "
        f"test={test_ans}+{test_unans} ({test_ans + test_unans})"
    )


if __name__ == "__main__":
    main()
