# Báo cáo Day 7 


## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Kế thừa toàn bộ dữ liệu từ bài thực hành Day 6: văn bản **Nghị định số 13/2023/NĐ-CP** ngày 17/4/2023 của Chính phủ về Bảo vệ dữ liệu cá nhân (`data/13-2023-ND-CP.pdf`), văn bản quy phạm pháp luật công khai.
  - Tổng cộng **37 trang đánh số** (trên **36 trang có nội dung**, trang 1–37) và **124 chunks** trích từ chiến lược C (`by_paragraph`, `max_size=700`) đã lưu tại `starter/data/day6_chunks.json` và chuyển đổi bằng [starter/convert_day6.py](../../starter/convert_day6.py) sang [data/converted/chunks.json](../../data/converted/chunks.json) và [data/converted/questions.json](../../data/converted/questions.json) (đồng bộ với `starter/data/converted/`).
  - Phân bố 124 chunks theo `metadata.chapter`: `Chương I` (20), `Chương II` (62), `Chương III` (11), `Chương IV` (2), `Phụ lục` (29).
- **Vector store, embedding model, phiên bản và số chiều:**
  - **Vector store:** PostgreSQL + extension `pgvector` trong container Docker Day 6 (`day6_student_starter-postgres-1`, image `pgvector/pgvector:0.8.7-pg16`, cổng `127.0.0.1:55433`, DB `day6_lab`), truy vấn trực tiếp bảng `rag_chunks` lọc theo `strategy = 'C'` (124 vector).
  - **Embedding model:** `google/embeddinggemma-2` (`sentence-transformers`, `torch.float32`), cắt chiều **256 dimensions** (`truncate_dim=256`, `normalize_embeddings=True`, prompt truy vấn `SearchQuery`), khớp 100% với index đã nạp ở Day 6 (xem [starter/retrievers.py](../../starter/retrievers.py)).
- **Agent model, cách chạy và phân chia dev/test:**
  - **LLM controller:** `openai/gpt-4o-mini` gọi qua **OpenRouter** (`temperature=0.0`, JSON object mode) bằng adapter [starter/llm_controller.py](../../starter/llm_controller.py) tích hợp với [starter/agentic.py](../../starter/agentic.py), chạy thật không mock qua [starter/run_agent_benchmark.py](../../starter/run_agent_benchmark.py) và [starter/run_test_evaluation.py](../../starter/run_test_evaluation.py).
  - **Phân chia benchmark (tổng 24 câu trong [data/converted/cases.json](../../data/converted/cases.json) và `starter/data/converted/cases.json`):**
    - `dev`: **17 câu** = 15 câu có bằng chứng (`q01`–`q15` từ Day 6) + 2 câu thiếu bằng chứng (`u01`, `u02`).
    - `test`: **7 câu** = 5 câu có bằng chứng (`q16`–`q20` từ Day 6) + 2 câu thiếu bằng chứng (`u03`, `u04`).
    - Toàn bộ bước thiết kế phương pháp, chọn trọng số rerank, tăng budget và tinh chỉnh prompt controller được khóa trên `dev` trước khi đánh giá độc lập trên `test`.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - 20 câu có bằng chứng (`q01`–`q20`) kế thừa `gold` (`source_id`, `page`, `evidence`) và `expected_answer` từ `starter/data/day6_questions.json` ([data/converted/questions.json](../../data/converted/questions.json)).
  - 4 câu thiếu bằng chứng được tạo trong [starter/build_cases.py](../../starter/build_cases.py) và rà soát toàn bộ 124 chunks (36 trang) để chắc chắn tài liệu không quy định:
    - `u01` (`dev`): mức phạt tiền tối đa (triệu đồng / % doanh thu) cho hành vi mua bán dữ liệu cá nhân trái phép (Điều 3 khoản 4 & Điều 4 chỉ nêu xử lý hành chính/kỷ luật/hình sự nói chung).
    - `u02` (`dev`): mức lệ phí thẩm định Hồ sơ ĐGTĐ xử lý dữ liệu và chuyển dữ liệu ra nước ngoài (Điều 24, 25, 31 không có biểu phí).
    - `u03` (`test`): chứng chỉ hành nghề hoặc bằng cấp tối thiểu bắt buộc của nhân sự phụ trách theo Điều 28 (Điều 28 & 30 không quy định bằng cấp/chứng chỉ).
    - `u04` (`test`): thời hạn hiệu lực tối đa theo năm và thủ tục gia hạn định kỳ của Hồ sơ ĐGTĐ (Điều 24 khoản 6 chỉ yêu cầu cập nhật khi thay đổi nội dung).
  - **Tiêu chí chấm thủ công (0 / 1 / 2):**
    - *Context score:* `0` = không có câu chứa bằng chứng cần thiết; `1` = có thông tin liên quan nhưng thiếu chi tiết/điều kiện chính; `2` = chứa đầy đủ bằng chứng để trả lời đúng và đủ.
    - *Agent grounded answer:* `0` = sai / không grounded / từ chối nhầm (`abstain` khi câu hỏi có đáp án); `1` = đúng ý chính nhưng thiếu điều kiện/ngoại lệ quan trọng; `2` = đúng, đầy đủ, mọi ý được đoạn trích `[chunk_id]` hậu thuẫn (`citation_support = true`). Với câu thiếu bằng chứng (`u01`–`u04`), `abstention_correct = true` chỉ khi `status = "abstain"`.

---

## Retrieval và thử nghiệm

Các phương pháp đã triển khai trong [starter/retrievers.py](../../starter/retrievers.py) và chạy tự động bằng [starter/run_retrieval_dev.py](../../starter/run_retrieval_dev.py) (chi phí dịch vụ retrieval cục bộ = `0 USD`):

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| `dev-tfidf` (Baseline) | `tfidf`, `top_k=3`, không filter, không rerank, `budget=800` | Baseline lexical chuẩn từ `baseline.py` | dev | 12/15 (80.0%) | 13/15 (86.7%) | 12/15 (80.0%) | median `0.525 ms`, tổng `18.495 ms`, `0 USD` |
| `dev-dense` (Phương pháp 2) | `dense_pgvector` (`google/embeddinggemma-2` 256d), `top_k=3`, suy luận `chapter` filter, không rerank, `budget=800` | Truy hồi ngữ nghĩa từ Day 6 vector store sẽ bắt các câu diễn đạt khác từ khóa gốc (`q01`, `q12`) | dev | 10/15 (66.7%) | **14/15 (93.3%)** | 10/15 (66.7%) | median `327.157 ms`, tổng `6383.941 ms` (load model `20.453 s`), `0 USD` |
| `dev-bm25` (Phương pháp 3) | `bm25` (Okapi BM25 tự cài đặt), `top_k=3`, suy luận `chapter` filter, không rerank, `budget=800` | Bổ sung bão hòa TF và chuẩn hóa độ dài tài liệu tốt hơn TF-IDF thuần | dev | 11/15 (73.3%) | **14/15 (93.3%)** | 11/15 (73.3%) | median `1.081 ms`, tổng `16.293 ms`, `0 USD` |
| `dev-hybrid-rrf` | `hybrid_rrf` (`dense + bm25`, `candidate_n=20`, `rrf_k=60`), `top_k=3`, filter, không rerank, `budget=800` | Kết hợp thứ hạng ngữ nghĩa và từ khóa bằng RRF để ổn định top-3 | dev | 10/15 (66.7%) | 13/15 (86.7%) | 10/15 (66.7%) | median `225.482 ms`, tổng `3656.554 ms`, `0 USD` |
| `dev-improve1-budget1800` (Cải thiện 1) | `hybrid_rrf`, `top_k=3`, filter, `budget=1800` (giữ nguyên retriever) | Giả thuyết: phần lớn lỗi `Context Hit` ở `dev-hybrid-rrf` do chunk dài ~600–700 ký tự nên `budget=800` cắt mất kết quả rank 2–3 | dev | 10/15 (66.7%) | 13/15 (86.7%) | **12/15 (80.0%)** | median `183.665 ms`, tổng `3200.196 ms`, `0 USD` |
| `dev-improve2-rerank-budget2200` (Cải thiện 2 — Chọn) | `hybrid_rerank` (`0.45 dense + 0.25 bm25 + 0.10 tfidf + 0.20 exact_match` trên 20 ứng viên sau chuẩn hóa Min-Max), `top_k=3`, suy luận `chapter` filter, `budget=2200` | Khắc phục suy giảm `Hit@1`/`Hit@3` của RRF bằng kết hợp điểm chuẩn hóa + thưởng cụm từ pháp lý/số liệu (`72 giờ`, `60 ngày`, bigram) và đủ budget chứa trọn top-3 | dev | **12/15 (80.0%)** | **14/15 (93.3%)** | **14/15 (93.3%)** | median `198.42 ms`, tổng `3314.654 ms`, `0 USD` |
| `test-selected-rerank-budget2200` (Cấu hình đã chọn) | `hybrid_rerank`, `top_k=3`, filter, `budget=2200` (khóa trước khi chạy `test`) | Đánh giá độc lập khả năng khái quát hóa so với baseline trên tập `test` chưa từng tinh chỉnh | test | **2/5 (40.0%)** | **4/5 (80.0%)** | **4/5 (80.0%)** | median `240.676 ms`, tổng `3131.823 ms` (load model `12.597 s`), `0 USD` |
| `test-tfidf` (Baseline) | `tfidf`, `top_k=3`, không filter, không rerank, `budget=800` | Đối chuẩn baseline trên tập `test` | test | 1/5 (20.0%) | 3/5 (60.0%) | 1/5 (20.0%) | median `0.469 ms`, tổng `2.576 ms`, `0 USD` |

*(Ghi chú độ dài context trung bình: trên `dev`, `budget=800` chỉ chứa trung bình `628.7–675.5` ký tự/câu (~1 chunk), lên `1376.8` ký tự ở `budget=1800` và `1724` ký tự ở `budget=2200`; trên `test`, tăng từ `720` lên `1937` ký tự. Nguồn số liệu: [results/dev_comparison.json](../../results/dev_comparison.json), [results/test_comparison.json](../../results/test_comparison.json).)*

### Chấm chất lượng Context thủ công (0 / 1 / 2) trên 5 câu `test` có bằng chứng

Chi tiết đánh giá lưu tại [results/test_context_audit.json](../../results/test_context_audit.json):

- **`test-tfidf` (`budget=800`):** `0/1/2 = 4 / 0 / 1`, điểm trung bình **0.40 / 2.00**.
- **`test-selected-rerank-budget2200` (`budget=2200`):** `0/1/2 = 1 / 0 / 4`, điểm trung bình **1.60 / 2.00**.

| Query ID | `test-tfidf` (`budget=800`) | Điểm | `test-selected-rerank-budget2200` (`budget=2200`) | Điểm | Lý do chấm |
|---|---|---|---|---|---|
| `q16` | `[p01:para:000]` (722 chars) | `0` | `[p01:para:000, p29:para:001, p22:para:001]` (2172 chars) | **`2`** | `budget=800` cắt mất rank-2 `p29:para:001` (hiệu lực 01/7/2023), chỉ còn `p01:para:000` (ngày ký 17/4/2023); `budget=2200` giữ đủ `p29:para:001`. |
| `q17` | `[p15:para:001]` (716 chars) | `2` | `[p15:para:001, p08:para:000, p14:para:001]` (2089 chars) | **`2`** | Cả hai đều đưa `p15:para:001` (72 giờ và ngoại lệ) lên rank 1 và vào context. |
| `q18` | `[p14:para:001]` (714 chars) | `0` | `[p14:para:001, p10:para:002, p16:para:003]` (2024 chars) | `0` | Cả hai đều thiếu `p16:para:001` (208 chars, bị ngắt trang tách khỏi tiêu đề `Điều 17` trên `p15:para:002`). |
| `q19` | `[p23:para:001]` (732 chars) | `0` | `[p01:para:000, p01:para:001, p02:para:002]` (1369 chars) | **`2`** | Baseline không filter và cắt rank-3 `p01:para:000` (chưa có Điều 1); `selected` suy luận filter `Chương I` + `budget=2200` giữ trọn `p01:para:001`. |
| `q20` | `[p15:para:001]` (716 chars) | `0` | `[p28:para:000, p27:para:002, p28:para:001]` (2031 chars) | **`2`** | Baseline cắt mất rank-3 `p28:para:000` (Điều 39 khoản 1); `selected` dùng filter `Chương III` + rerank đưa `p28:para:000` lên rank 1. |

### Hai lỗi tiêu biểu trên `dev`, nguyên nhân, cách thử và lý do chốt cấu hình

1. **Lỗi 1 — Context truncation ở `budget=800` dù retrieval đã trúng trong top-3 (`q11`, `q04`, `q08`, `q12`):**
   - *Hiện tượng:* Trong Day 6 Strategy C, mỗi chunk có độ dài trung bình ~500–700 ký tự. Khi đóng gói với header `[chunk_id | source | page]`, chunk rank 1 thường chiếm ~600–740 ký tự, làm tràn `budget=800` và cắt bỏ toàn bộ rank 2–3. Ví dụ: ở `q11` (hỏi thời hạn gửi hồ sơ chuyển dữ liệu ra nước ngoài), `dev-tfidf` và `dev-bm25` đều đưa được chunk đúng `nd-13-2023-nd-cp:p22:para:000` (60 ngày) vào top-3, nhưng chunk nhiễu `p22:para:001` (10 ngày bổ sung) chiếm rank 1 đã đẩy chunk đúng ra khỏi context. Tương tự trên `dev-dense` (`Hit@3=14/15` nhưng `CtxHit=10/15`) và `dev-hybrid-rrf` (`Hit@3=13/15` nhưng `CtxHit=10/15`).
   - *Cách thử & kết quả:* Cải thiện 1 (`dev-improve1-budget1800`) nâng `budget` lên `1800`, khôi phục được `q08` và `q12` (`CtxHit` tăng từ `10/15` lên `12/15`); Cải thiện 2 nâng tiếp lên `2200` để luôn chứa trọn tối thiểu 3 chunk (~1900–2100 ký tự gồm header), xóa hoàn toàn khoảng cách giữa `Hit@3` và `CtxHit` (`14/15` và `14/15` trên `dev`, `4/5` và `4/5` trên `test`).
2. **Lỗi 2 — Mất tín hiệu xếp hạng chi tiết khi trộn bằng RRF và nhầm lẫn giữa các đoạn cùng chủ đề (`q02`, `q12`):**
   - *Hiện tượng:* Trên `q12` ("Đơn vị nào là Cơ quan chuyên trách bảo vệ dữ liệu cá nhân?"), `dev-tfidf` xếp hạng nhầm `p24:para:000` (chỉ có tiêu đề Điều 29) và `p24:para:002` (Điều 30) lên trước `p24:para:001` (chứa tên "Cục An ninh mạng... - Bộ Công an"), dẫn tới `Hit@3=0`. Ngoài ra `dev-hybrid-rrf` dùng công thức hạng rời rạc `1/(60+rank)` đánh mất độ chênh lệch điểm cosine/lexical, làm trượt `q02` (`p04:para:000`) ra khỏi top-3 dù `dev-tfidf` và `dev-bm25` đều bắt trúng.
   - *Cách thử & lý do chọn:* Cải thiện 2 (`HybridRerankRetriever`) chuẩn hóa Min-Max điểm thô từ 3 kênh (`0.45 dense + 0.25 bm25 + 0.10 tfidf`) kết hợp `0.20` điểm khớp cụm từ pháp lý và bigram (`_phrase_and_legal_match_score`), giúp nâng `Hit@1` trở lại `12/15 (80.0%)`, giữ `Hit@3=14/15 (93.3%)` và `CtxHit=14/15 (93.3%)`. Điểm yếu còn lại ở `dev` chỉ là `q01` (câu phân loại chung quá ngắn trong `p02:para:000` đứng sau các chunk dài hơn chứa nhiều lần cụm "dữ liệu cá nhân"). Với `CtxHit` cao nhất (`93.3%`) và tổng độ trễ chấp nhận được (~200 ms/câu), `hybrid_rerank + budget=2200 + chapter filter` là cấu hình tối ưu để chốt trước khi chạy `test`.

---

## Agent và đánh giá

### Thiết kế Tool, điều kiện dừng, chọn context và xử lý lỗi

- **Tool cung cấp cho controller (`RealLLMController`):** 1 tool duy nhất là tìm kiếm tri thức trên corpus (tương ứng hành động `{"action": "search", "query": "..."}` được `agentic.run_agent` gọi qua `retriever.search(query, filters=filters, top_k=3)`).
- **Vòng lặp & điều kiện dừng:** `max_searches = 3` (tối đa 3 lượt tìm kiếm, tổng tối đa 4 bước gọi LLM). Vòng lặp dừng khi LLM trả về `"answer"` kèm ít nhất 1 `chunk_id` tồn tại trong `final_context_ids` và xuất hiện đúng định dạng `[chunk_id]` trong câu trả lời; hoặc `"abstain"` khi thiếu bằng chứng; hoặc bị hệ thống chặn thành `abstain` nếu vượt `max_searches` hay trích dẫn không hợp lệ (`invalid_or_missing_citations`).
- **Cách chọn context:** Theo [starter/agentic.py](../../starter/agentic.py), sau mỗi lượt `search`, kết quả được nối vào danh sách `candidates`, loại trùng và đóng gói theo `budget_chars` (`800` cho baseline, `2200` cho selected) bằng `pack_context` rồi gửi lịch sử cập nhật sang LLM. Với cấu hình `selected`, `filters` được suy luận hoặc lấy theo trường `filter` của `case` (ví dụ `{"chapter": "Chương I"}` hoặc `{"chapter": "Chương III"}`).
- **Cải tiến controller sau phân tích lỗi `dev` (trước -> sau):**
  - Ở bản chạy đầu (`agent-selected-dev-before`), lượt tìm đầu tiên để LLM tự viết lại query ngay từ đầu nên hay đánh rơi từ khóa gốc hoặc tách ngữ cảnh (`q10` hỏi thời hạn thông báo vi phạm bị abstain sớm sau 1 lượt tìm; `q12` trả lời mơ hồ dựa vào `p24:para:000`).
  - Bản cải tiến trong [starter/llm_controller.py](../../starter/llm_controller.py) bật hai cơ chế: (1) `use_original_first_query=True` — luôn chạy truy vấn đầu bằng đúng câu hỏi gốc (không tốn 1 lượt gọi LLM và giữ nguyên từ khóa pháp lý + tên Chương); (2) `require_reformulate_before_abstain=True` — nếu mới tìm 1 lượt và chưa đủ bằng chứng, nhắc LLM diễn giải lại từ đồng nghĩa / Điều khoản và tìm thêm ít nhất 1 lượt trước khi được quyền `abstain`.
- **Xử lý lỗi hệ thống:** Lỗi API/HTTP hoặc JSON không hợp lệ được ném ngoại lệ rõ ràng và ghi nhận `status="error"` (không có fallback trả lời giả); mọi lần chạy đều ghi `provider_errors = 0`.

### Tổng hợp chỉ số `agent_metrics.py` trên `dev` và `test`

| Cấu hình / split | Tool use | Số lượt gọi (`1/2/3`) | Grounded answer `0/1/2` và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm | Token / Chi phí / Tổng thời gian |
|---|---|---|---|---|---|---|---|---|
| `agent-baseline-dev` (`dev`, `tfidf`, `budget=800`) | 17/17 | `14 / 1 / 2` | `5 / 1 / 9` (`N=15`, mean **`1.27`**) | 10/12 (83.3%) | 12/12 (100%) | 2/2 (100%) | 3/15 (20.0%) | `25,644 in / 1,959 out` · `0.005022 USD` · `75.03 s` |
| `agent-selected-dev-before` (`dev`, `hybrid_rerank`, trước sửa controller) | 17/17 | `16 / 1 / 0` | `3 / 0 / 12` (`N=15`, mean **`1.60`**) | 12/14 (85.7%) | 14/14 (100%) | 2/2 (100%) | 1/15 (6.7%) | `29,611 in / 2,031 out` · `0.005660 USD` · `72.32 s` |
| `agent-selected-dev` (`dev`, `hybrid_rerank`, **sau sửa controller**) | 17/17 | `14 / 1 / 2` | `1 / 0 / 14` (`N=15`, mean **`1.87`**) | **14/14 (100%)** | 14/14 (100%) | 2/2 (100%) | 1/15 (6.7%) | `28,157 in / 1,694 out` · `0.005240 USD` · `56.39 s` |
| `agent-baseline-test` (`test`, `tfidf`, `budget=800`) | 7/7 | `2 / 1 / 4` | `4 / 0 / 1` (`N=5`, mean **`0.40`**) | 1/2 (50.0%) | 2/2 (100%) | 2/2 (100%) | 3/5 (60.0%) | `13,694 in / 738 out` · `0.002497 USD` · `27.02 s` |
| `agent-selected-test` (`test`, `hybrid_rerank`, `budget=2200`) | 7/7 | `4 / 1 / 2` | `1 / 0 / 4` (`N=5`, mean **`1.60`**) | **4/4 (100%)** | 4/4 (100%) | 2/2 (100%) | 1/5 (20.0%) | `15,899 in / 687 out` · `0.002797 USD` · `33.26 s` |

### Bảng trace từng câu (toàn bộ tập `test` của `agent-selected-test` đối chiếu `baseline`, cộng các ca `dev` tiêu biểu)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q16` (`test`) | Có evidence (Điều 43 hiệu lực thi hành) | `selected`: 1 lượt với câu gốc -> top-3 `[p01:para:000, p29:para:001, p22:para:001]` (2172 chars). *(Baseline: `budget=800` cắt mất `p29:para:001`, dẫn đến nhầm ngày ký 17/4/2023 = grade 0).* | `selected`: "...có hiệu lực thi hành từ ngày 01 tháng 7 năm 2023 `[nd-13-2023-nd-cp:p29:para:001]`." | **Grade 2**, `cit=True` (`baseline`: Grade 0, `cit=False`) |
| `q17` (`test`) | Có evidence (Điều 16 khoản 5 xóa dữ liệu) | `selected`: 1 lượt -> top-3 `[p15:para:001, p08:para:000, p14:para:001]`. | `selected`: "...trong 72 giờ sau khi có yêu cầu... trừ trường hợp pháp luật có quy định khác `[nd-13-2023-nd-cp:p15:para:001]`." | **Grade 2**, `cit=True` |
| `q18` (`test`) | Có evidence (Điều 17 khoản 4 nghĩa vụ hợp đồng) | `selected`: 3 lượt (gốc + 2 lượt rewrite `"xử lý dữ liệu cá nhân không cần sự đồng ý nghĩa vụ hợp đồng Nghị định 13/2023/NĐ-CP"`) -> đều không kéo được `p16:para:001` (vì bị ngắt trang tách khỏi tiêu đề `Điều 17` ở `p15:para:002`). | `selected`: "Không đủ bằng chứng để trả lời." (`abstain`, `cits=[]`). | **Grade 0** (Từ chối nhầm do lỗi biên giới chunk qua trang) |
| `q19` (`test`) | Có evidence (Chương I, Điều 1 phạm vi điều chỉnh) | `selected`: 1 lượt với filter `{"chapter": "Chương I"}` -> `[p01:para:000, p01:para:001, p02:para:002]`. *(Baseline không có filter và bị budget cắt nên bỏ sót `p01:para:001` -> abstain).* | `selected`: "Nghị định này quy định về bảo vệ dữ liệu cá nhân và trách nhiệm bảo vệ dữ liệu cá nhân của cơ quan, tổ chức, cá nhân có liên quan `[nd-13-2023-nd-cp:p01:para:001]`." | **Grade 2**, `cit=True` (`baseline`: Grade 0) |
| `q20` (`test`) | Có evidence (Chương III, Điều 39 khoản 1) | `selected`: 1 lượt với filter `{"chapter": "Chương III"}` -> `[p28:para:000, p27:para:002, p28:para:001]`. *(Baseline search 1 nạp `p15:para:001` chiếm 716/800 chars khiến search 2–3 dù thấy `p28:para:000` ở rank 1 vẫn bị `pack_context` loại -> abstain).* | `selected`: "...chỉ được tiếp nhận dữ liệu cá nhân sau khi có hợp đồng hoặc thỏa thuận về xử lý dữ liệu với Bên Kiểm soát dữ liệu cá nhân `[nd-13-2023-nd-cp:p28:para:000]`." | **Grade 2**, `cit=True` (`baseline`: Grade 0) |
| `u03` (`test`) | Thiếu evidence (chứng chỉ/bằng cấp Điều 28) | `selected`: 2 lượt (gốc + `"Điều kiện chứng chỉ hành nghề..."`) -> `[p23:para:001, p23:para:002, p01:para:000]` có đủ Điều 28 nhưng không nhắc chứng chỉ/bằng cấp. | `selected`: `abstain` ("Không đủ bằng chứng để trả lời.", `cits=[]`). | **Abstention đúng** (`True`) |
| `u04` (`test`) | Thiếu evidence (thời hạn gia hạn Hồ sơ ĐGTĐ) | `selected`: 3 lượt (gốc + 2 lượt rewrite về Điều 24) -> `[p20:para:002, p35:para:003, p19:para:001]` chỉ nói cập nhật khi thay đổi, không có thời hạn năm. | `selected`: `abstain` ("Không đủ bằng chứng để trả lời.", `cits=[]`). | **Abstention đúng** (`True`) |
| `q01` (`dev`) | Có evidence (phân loại dữ liệu cá nhân) | `selected`: 3 lượt không bắt được `p02:para:000` (nên chọn `abstain` thay vì đoán sai `p03:para:000` như bản `before`). | `selected`: `abstain` (`before` từng trả lời sai chỉ liệt kê dữ liệu nhạy cảm với `[p03:para:000]`). | **Grade 0** (giảm citation ảo từ `cit=False` sang `abstain`) |
| `q09` (`dev`) | Có evidence (trẻ em từ đủ 7 tuổi) | `selected`: 1 lượt -> `[p17:para:000, p16:para:003, p17:para:001]`. *(Baseline trả lời thiếu vế ngoại lệ Điều 17 -> Grade 1).* | `selected`: Nêu đủ đồng ý của trẻ em, cha/mẹ/người giám hộ và ngoại lệ Điều 17 `[nd-13-2023-nd-cp:p17:para:000]`. | **Grade 2**, `cit=True` (`baseline`: Grade 1) |
| `q10` (`dev`) | Có evidence (thông báo vi phạm 72 giờ) | `before`: 1 lượt rewrite ngắn làm hụt `p18:para:000` -> abstain sớm. `selected`: câu gốc ở lượt 1 trúng `p18:para:000` rank 1. | `selected`: "...chậm nhất 72 giờ sau khi xảy ra hành vi vi phạm `[nd-13-2023-nd-cp:p18:para:000]`." | **Sửa thành công: Grade 0 -> Grade 2**, `cit=True` |
| `q11` (`dev`) | Có evidence (60 ngày nộp Hồ sơ chuyển dữ liệu) | `selected`: `budget=2200` giữ được `p22:para:000` cùng `p22:para:001` (Baseline `budget=800` chỉ thấy `p22:para:001` nên nhầm sang 10 ngày bổ sung). | `selected`: "...trong thời gian 60 ngày kể từ ngày tiến hành xử lý dữ liệu cá nhân `[nd-13-2023-nd-cp:p22:para:000]`." | **Grade 2**, `cit=True` (`baseline`: Grade 0, `cit=False`) |
| `q12` (`dev`) | Có evidence (Cơ quan chuyên trách) | `selected`: câu gốc đưa đủ 3 đoạn trang 24 `[p24:para:000, p24:para:001, p24:para:002]` và model trích đúng `p24:para:001`. | `selected`: "...là Cục An ninh mạng và phòng, chống tội phạm sử dụng công nghệ cao - Bộ Công an `[nd-13-2023-nd-cp:p24:para:001]`." | **Sửa thành công: Grade 0 (`cit=False`) -> Grade 2 (`cit=True`)** |

### Đường dẫn artifact, lệnh tái lập, đánh đổi và giới hạn

- **Đường dẫn kết quả & nhãn đã đồng bộ vào `results/` (và `starter/results/`):**
  - Retrieval comparison: [results/dev_comparison.json](../../results/dev_comparison.json), [results/test_comparison.json](../../results/test_comparison.json), [results/test_context_audit.json](../../results/test_context_audit.json).
  - Agent traces, manual reviews & summaries: [results/agent-baseline-dev/](../../results/agent-baseline-dev/summary.json), [results/agent-selected-dev-before/](../../results/agent-selected-dev-before/summary.json), [results/agent-selected-dev/](../../results/agent-selected-dev/summary.json), [results/agent-baseline-test/](../../results/agent-baseline-test/summary.json), [results/agent-selected-test/](../../results/agent-selected-test/summary.json).
- **Lệnh chạy tái lập từ thư mục gốc repo:**
  ```powershell
  python starter/build_cases.py --questions starter/data/converted/questions.json --output starter/data/converted/cases.json
  python starter/run_retrieval_dev.py
  python starter/run_agent_benchmark.py --split dev --config both --max-searches 3
  python starter/run_test_evaluation.py --max-searches 3
  python starter/agent_metrics.py --cases starter/data/converted/cases.json --traces starter/results/agent-selected-test/traces.json --reviews starter/results/agent-selected-test/reviews.json --split test --output starter/results/agent-selected-test/summary.json
  python -m unittest discover -s tests -v
  ```
- **Kết quả trước/sau, đánh đổi và giới hạn kỹ thuật còn tồn tại:**
  1. **Hiệu quả rõ rệt trên cả `dev` và `test`:** Trên `dev`, cấu hình `selected` cải tiến nâng `Grounded Answer` mean từ `1.27` (baseline) -> `1.60` (trước sửa controller) -> **`1.87`**, đạt **`14/14 (100%)`** citation hợp lệ và giữ `100%` abstention đúng trên `u01`, `u02`. Trên `test` độc lập, `agent-selected-test` tăng điểm trung bình gấp 4 lần từ **`0.40`** (`1/5` đúng) lên **`1.60`** (`4/5` đúng, `4/4 = 100%` citation hợp lệ, `2/2` abstention đúng). Chi phí LLM rất thấp (`0.00524 USD` cho 17 câu `dev` và `0.002797 USD` cho 7 câu `test`), tổng thời gian ~33s–56s (~4.8s/câu do gọi API + embed `240 ms/truy vấn`).
  2. **Đánh đổi giữa `budget_chars` và ngữ cảnh nhiễu:** Tăng `budget` từ `800 -> 2200` tăng chi phí prompt token thêm ~10–16% (từ `13,694` lên `15,899` tokens trên `test`), nhưng gần như loại bỏ các lỗi cắt bằng chứng (`q11`, `q16`, `q19`, `q20`) và giúp model không bị đánh lừa bởi chunk đầu tiên sai (`p01:para:000` ở `q16`, `p22:para:001` ở `q11`).
  3. **Giới hạn 1 — Ghép ứng viên tuần tự của [starter/agentic.py](../../starter/agentic.py):** Hàm `run_agent` gọi `candidates.extend(results)` rồi chạy `pack_context(candidates, budget)`. Với `budget` nhỏ (như `800` ở baseline), chunk sai ở lượt 1 chiếm trọn budget khiến chunk đúng rank 1 ở lượt tìm lại 2–3 không bao giờ lọt vào context gửi cho LLM (quan sát trực tiếp tại `q20` `agent-baseline-test`). Cần ưu tiên hoặc thay thế bằng kết quả lượt tìm mới nhất theo điểm rerank nếu muốn hỗ trợ budget hẹp hơn.
  4. **Giới hạn 2 — Tách chunk ngang ranh giới trang (`q18`):** Chuyển trang 15->16 chia tách tiêu đề `Điều 17` + câu dẫn "không cần có sự đồng ý" (`p15:para:002`) khỏi khoản 4 (`p16:para:001`, chỉ dài 208 ký tự). Không một truy vấn nào kéo được `p16:para:001` vào top-3 vì thiếu ngữ cảnh tiêu đề Điều. Hướng khắc phục cho vòng kế tiếp là chunk theo cấu trúc Điều/Khoản xuyên trang hoặc gắn `article_heading` vào đầu mỗi chunk khi ingest.

---

## Thử thách nâng cao (nếu thực hiện)


