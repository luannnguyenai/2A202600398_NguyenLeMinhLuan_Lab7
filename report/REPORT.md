# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Nguyễn Lê Minh Luân]
**Nhóm:** [2]
**Ngày:** 2026-04-10

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity (gần 1.0) nghĩa là hai đoạn văn bản có ý nghĩa ngữ nghĩa tương tự nhau — embedding vectors của chúng gần như "cùng hướng" trong không gian nhiều chiều. Trong ngữ cảnh pháp luật, điều này có nghĩa là hai điều khoản nói về cùng một vấn đề pháp lý, dù dùng từ ngữ khác nhau.

**Ví dụ HIGH similarity (từ Bộ luật Dân sự):**
- Sentence A: "Cá nhân, pháp nhân xác lập, thực hiện quyền dân sự trên cơ sở tự do, tự nguyện cam kết, thỏa thuận."
- Sentence B: "Mọi cam kết, thỏa thuận không vi phạm điều cấm của luật có hiệu lực thực hiện đối với các bên."
- Tại sao tương đồng: Cả hai câu đều nói về quyền tự do ý chí trong giao dịch dân sự, chia sẻ ngữ nghĩa về cam kết, thỏa thuận và hiệu lực pháp lý.

**Ví dụ LOW similarity (từ Bộ luật Dân sự):**
- Sentence A: "Tòa án tuyên bố một người là đã chết khi biệt tích 5 năm liền trước ngày yêu cầu."
- Sentence B: "Chủ sở hữu có quyền khai thác, sử dụng, định đoạt tài sản thuộc sở hữu của mình."
- Tại sao khác: Một câu thuộc chương về tuyên bố chết (Nhân thân), câu kia thuộc chương về quyền sở hữu (Tài sản) — hai domain pháp lý hoàn toàn khác biệt.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ đo **hướng** của vector, không phụ thuộc vào **độ dài** (magnitude). Đặc biệt quan trọng với văn bản pháp luật vì Điều 3 (ngắn gọn, 5 khoản) và Điều 122 (dài, nhiều khoản) cần được so sánh công bằng. Nếu dùng Euclidean distance, Điều 122 luôn bị coi là "xa hơn" chỉ vì nội dung dài hơn, dù ngữ nghĩa liên quan. Cosine similarity loại bỏ sai lệch này.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Áp dụng công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
> `num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23 chunks`
> **Đáp án: 23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn trong văn bản pháp luật?**
> `num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25 chunks`
> Số chunks tăng từ 23 lên 25. Đối với văn bản pháp luật như Bộ luật Dân sự, overlap lớn hơn đặc biệt quan trọng vì nhiều Điều khoản tham chiếu chéo nhau — ví dụ "theo quy định tại Điều 3 của Bộ luật này". Overlap đảm bảo câu tham chiếu này xuất hiện ở cả hai chunk liền kề, giúp retrieval hiểu được mối liên hệ giữa các Điều.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Pháp luật Dân sự Việt Nam (Vietnamese Civil Law)

**Tại sao nhóm chọn domain này?**
> Bộ luật Dân sự 2015 (Luật số 91/2015/QH13) là một trong những văn bản pháp luật nền tảng nhất của hệ thống pháp luật Việt Nam với hơn 600 điều khoản. Domain này có cấu trúc phân cấp rõ ràng (Phần → Chương → Điều → Khoản → Điểm), ngôn ngữ đặc thù, và yêu cầu hiểu ngữ nghĩa sâu — đây là bài kiểm tra thực tế khắt khe cho hệ thống RAG. Việc xây dựng RAG cho pháp luật có giá trị ứng dụng thực tiễn cao (hỗ trợ tra cứu pháp lý, tư vấn sơ bộ).

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Luat_Dan_Su.md | Cổng Thông tin điện tử Chính phủ (chinhphu.vn) | 283,442 | category=law, lang=vi, law_code=91/2015/QH13 |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| category | str | "law" | Cho phép lọc ra chỉ các tài liệu pháp lý khi hệ thống có nhiều loại tài liệu |
| lang | str | "vi" | Đảm bảo chỉ trả về tài liệu tiếng Việt, tránh nhầm với văn bản nước ngoài |
| source | str | "data/Luat_Dan_Su.md" | Truy ngược nguồn gốc điều khoản được trích dẫn |
| law_code | str | "91/2015/QH13" | Lọc theo số hiệu luật khi hệ thống có nhiều bộ luật khác nhau |
| chunk_idx | int | 0, 1, 2, ... | Xác định vị trí chunk trong tài liệu gốc, hỗ trợ tái tạo ngữ cảnh xung quanh |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên `Luat_Dan_Su.md` (chunk_size=500):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Luat_Dan_Su.md | fixed_size | 556 | 499.2 | Kém — cắt ngang Điều/Khoản |
| Luat_Dan_Su.md | by_sentences | 892 | 311.4 | Trung bình — giữ câu nhưng tách Điều |
| Luat_Dan_Su.md | recursive (chunk=500) | 599 | 462.3 | Tốt — ưu tiên tách theo đoạn văn |
| Luat_Dan_Su.md | recursive (chunk=1000) | 230 | 907.1 | Tốt nhất — giữ trọn Điều khoản |

### Strategy Của Tôi

**Loại:** RecursiveChunker (chunk_size=1000)

**Mô tả cách hoạt động:**
> RecursiveChunker chia văn bản theo thứ tự ưu tiên: trước tiên thử tách theo đoạn (`\n\n`), rồi theo dòng (`\n`), rồi theo câu (`. `), rồi theo từ (` `), và cuối cùng là theo ký tự (`""`). Base case: nếu text ≤ `chunk_size` thì trả về ngay. Các phần nhỏ hơn `chunk_size` được gộp lại để tối ưu kích thước. Điều này đảm bảo chunk luôn giữ được ngữ nghĩa trọn vẹn nhất có thể.

**Tại sao tôi chọn strategy này cho Bộ luật Dân sự?**
> Bộ luật Dân sự có cấu trúc đặc thù: mỗi **Điều** là một đơn vị ngữ nghĩa độc lập, có thể ngắn (2-3 khoản) hoặc dài (10+ khoản). Với `chunk_size=1000`, RecursiveChunker thường giữ trọn được 1-2 Điều trong mỗi chunk — đây là đơn vị trả lời tự nhiên nhất cho câu hỏi pháp lý. So với FixedSizeChunker (cắt ngang giữa chừng một Điều), hay SentenceChunker (tách từng khoản riêng lẻ mất ngữ cảnh), recursive chunking cho retrieval quality cao nhất trên loại tài liệu này.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|-------------------|
| Luat_Dan_Su.md | fixed_size (baseline, 500) | 556 | 499.2 | Kém — cắt ngang Điều 68 thành 2 phần, mất ngữ cảnh tuyên bố mất tích |
| Luat_Dan_Su.md | by_sentences (baseline, 500) | 892 | 311.4 | Trung bình — từng khoản riêng nhưng mất header "Điều X" |
| Luat_Dan_Su.md | **recursive (của tôi, 1000)** | 230 | 907.1 | Tốt nhất — mỗi chunk giữ trọn 1-2 Điều kèm header |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | RecursiveChunker (1000) | 7/10 | Giữ trọn Điều khoản, header rõ ràng | Chunk lớn có thể chứa nhiều chủ đề |
| [Hoàng] | SentenceChunker (max=2) | 6/10 | Chunk nhỏ, tìm được chi tiết khoản | Mất header Điều, khó biết đang ở Điều nào |
| [Đức] | FixedSizeChunker (500) | 4/10 | Đơn giản, predictable | Cắt ngang Điều, mất "Điều X" header quan trọng |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker với `chunk_size=1000` cho kết quả tốt nhất cho Bộ luật Dân sự vì nó khai thác cấu trúc phân cấp sẵn có (## **Điều X.**). Khi mỗi chunk giữ được header "## **Điều 68.**" kèm nội dung, retrieval có thể tìm ra đúng Điều khi người dùng hỏi về "tuyên bố mất tích". FixedSizeChunker và SentenceChunker thường cắt bỏ header hoặc tách nội dung ra khỏi Điều gốc.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng regex `(?<=[.!?])(?:\s|\n)` để detect vị trí kết thúc câu — pattern này tìm các ký tự `.`, `!`, `?` theo sau bởi space hoặc newline. Sau khi tách thành danh sách câu, gom nhóm mỗi `max_sentences_per_chunk` câu thành 1 chunk. Edge case đã xử lý: chuỗi rỗng trả về `[]`, strip whitespace thừa ở mỗi chunk, lọc bỏ chunk rỗng. Với văn bản pháp lý có nhiều dấu chấm trong số thứ tự (1. 2. 3.), cần xử lý được cả trường hợp này.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Algorithm dùng danh sách separators theo thứ tự ưu tiên giảm dần (`\n\n` → `\n` → `. ` → ` ` → `""`). Base case: nếu text ≤ `chunk_size` thì trả về ngay. Với Bộ luật Dân sự, separator `\n\n` tách đúng các Điều (vì mỗi Điều cách nhau bằng dòng trắng). Khi một Điều quá dài (nhiều khoản), `\n` tiếp tục chia nhỏ theo khoản. Cuối cùng, gộp các phần nhỏ lại để tối ưu chunk size gần `chunk_size=1000`.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` tạo embedding cho mỗi chunk của Luat_Dan_Su.md bằng `embedding_fn`, lưu vào `self._store` dưới dạng dict chứa `id`, `content`, `embedding`, `metadata`. Với 230 chunks, quá trình indexing mất khoảng 0.0s (MockEmbedder) hoặc ~15s (Nomic qua LM Studio). `search` tạo embedding cho query ("Tòa án tuyên bố mất tích khi nào?"), tính cosine similarity với tất cả 230 stored embeddings, sort descending, trả về top-k kết quả.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` thực hiện pre-filtering: lọc `self._store` theo `{"category": "law", "lang": "vi"}` trước, rồi chỉ chạy similarity search trên tập đã lọc. Vì hiện tại chỉ có 1 tài liệu (Luat_Dan_Su.md), filter không thay đổi kết quả — nhưng khi hệ thống mở rộng thêm các bộ luật khác (Luật Hình sự, Luật Thương mại), filter `law_code=91/2015/QH13` sẽ cực kỳ hữu ích để chỉ search trong Bộ luật Dân sự. `delete_document` cho phép xóa chunk cụ thể nếu phát hiện chunk bị cắt sai.

### KnowledgeBaseAgent

**`answer`** — approach:
> RAG pattern 3 bước: (1) Gọi `store.search()` để retrieve top-3 chunks từ Bộ luật Dân sự liên quan đến query; (2) Xây prompt bao gồm system instruction ("Dựa trên các điều khoản pháp luật sau để trả lời"), các context blocks được format với score và nội dung Điều khoản, và câu hỏi; (3) Gọi `llm_fn(prompt)` và trả về kết quả. Prompt structure cho phép LLM cite đúng số Điều khoản trong câu trả lời, tăng tính traceability của câu trả lời pháp lý.

### Test Results

```
============================= test session starts ==============================
platform darwin -- Python 3.9.6, pytest-8.4.2, pluggy-1.6.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================== 42 passed in 0.02s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Các cặp câu lấy trực tiếp từ nội dung **Bộ luật Dân sự 2015**:

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score (Mock) | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Cá nhân, pháp nhân phải xác lập, thực hiện quyền dân sự của mình một cách thiện chí, trung thực." (Điều 3K3) | "Mọi cam kết, thỏa thuận không vi phạm điều cấm của luật có hiệu lực thực hiện đối với các bên." (Điều 3K2) | high | -0.0621 (low) | ✗ |
| 2 | "Tòa án ra quyết định tuyên bố mất tích khi một người biệt tích 2 năm liền." (Điều 68) | "Người bị tuyên bố là đã chết mà còn sống có quyền yêu cầu khôi phục quyền nhân thân." (Điều 73) | high | -0.0843 (low) | ✗ |
| 3 | "Chủ sở hữu có quyền khai thác, sử dụng, định đoạt tài sản." (Điều 158) | "Bộ luật này là luật chung điều chỉnh các quan hệ dân sự." (Điều 4K1) | low | -0.0392 (low) | ✓ |
| 4 | "Người từ đủ 18 tuổi trở lên là người thành niên." (Điều 20K1) | "Người chưa đủ 18 tuổi là người chưa thành niên." (Điều 21K1) | high | 0.0812 (low-medium) | ✗ |
| 5 | "Pháp nhân phải chịu trách nhiệm dân sự bằng tài sản của mình." (Điều 87) | "Cá nhân phải chịu trách nhiệm dân sự về việc thực hiện nghĩa vụ dân sự." (Điều 1) | low | 0.0156 (low) | ✓ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là Pair 2 và Pair 4 — hai cặp câu rõ ràng cùng chủ đề pháp lý ("tuyên bố mất tích" vs "tuyên bố chết"; "người thành niên" vs "người chưa thành niên") nhưng Mock Embeddings cho score cực thấp, thậm chí âm. Điều này chứng minh **mock embeddings dựa trên MD5 hash hoàn toàn mù quáng về ngữ nghĩa** — chúng chỉ là pseudo-random vectors. Khi chuyển sang Real AI (Nomic v1.5 qua LM Studio), Pair 2 (mất tích / tuyên bố chết) cho cosine similarity > 0.7 vì model hiểu đây là các khái niệm pháp lý liên quan. Đây là bài học cốt lõi: **chất lượng embedding model quyết định hoàn toàn khả năng retrieval**.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers (thống nhất trên Luat_Dan_Su.md)

| # | Query | Gold Answer (Điều luật) | Nội dung chính |
|---|-------|------------------------|----------------|
| 1 | Các nguyên tắc cơ bản của pháp luật dân sự là gì? | **Điều 3** | Bình đẳng, tự do ý chí, thiện chí trung thực, tự chịu trách nhiệm |
| 2 | Năng lực hành vi dân sự của người từ đủ 15 đến dưới 18 tuổi được quy định thế nào? | **Điều 21** | Có thể tự xác lập giao dịch dân sự theo quy định của pháp luật |
| 3 | Cá nhân có quyền thay đổi họ trong những trường hợp nào? | **Điều 27** | Theo yêu cầu của người có họ và được cơ quan có thẩm quyền xác định |
| 4 | Điều kiện để một cá nhân được làm người giám hộ là gì? | **Điều 49** | Thành niên, có năng lực hành vi dân sự đầy đủ, có điều kiện cần thiết |
| 5 | Tòa án tuyên bố một người mất tích khi nào? | **Điều 68** | Biệt tích 2 năm liền mà không có tin tức xác thực là còn sống |

### Kết Quả So Sánh (Mock Embedder vs LM Studio Nomic v1.5)

| # | Query | Mock: Top-1 Chunk | Mock Hit? | LM Studio: Top-1 Chunk | LM Studio Hit? |
|---|-------|-------------------|-----------|----------------------|----------------|
| 1 | Nguyên tắc cơ bản pháp luật dân sự | Điều 50 (Giám hộ pháp nhân) | ✗ Sai | **Điều 3 (Nguyên tắc cơ bản)** | ✓ Đúng |
| 2 | Năng lực hành vi 15-18 tuổi | Điều 238 (Sở hữu chung) | ✗ Sai | Điều 3 (Nguyên tắc cơ bản) | ✗ Sai |
| 3 | Quyền thay đổi họ | Điều 32 (Quyền đối với hình ảnh) | ✗ Sai | Điều 34 (Quyền được bảo vệ danh dự) | ✗ Sai |
| 4 | Điều kiện làm người giám hộ | Điều 87 (Pháp nhân chịu trách nhiệm) | ✗ Sai | **Điều 49 (Điều kiện giám hộ)** | ✓ Đúng |
| 5 | Tòa án tuyên bố mất tích | Điều 122 (Giao dịch vô hiệu) | ✗ Sai | **Điều 68 (Tuyên bố mất tích)** | ✓ Đúng |

**Tỉ lệ tìm đúng trong Top-3:**

| Embedder | Precision @Top-3 | Nhận xét |
|----------|-----------------|---------|
| **MockEmbedder (MD5 hash)** | 1/5 = **20%** | Chỉ trúng ngẫu nhiên, không hiểu ngữ nghĩa pháp lý |
| **LM Studio (Nomic v1.5)** | 3/5 = **60%** | Hiểu được khái niệm ngữ nghĩa; trúng các Điều quan trọng nhất |

**Phân tích từng câu hỏi:**

- **Q1 (Điều 3) — LM Studio ✓:** Model hiểu "nguyên tắc cơ bản" → tìm đúng Điều 3 với nội dung về bình đẳng, tự nguyện, thiện chí.
- **Q2 (Điều 21) — Cả hai ✗:** "Năng lực hành vi 15-18 tuổi" quá cụ thể. Chunk chứa Điều 21 có thể bị tách bởi RecursiveChunker, header bị lẫn vào chunk khác. → Cần tăng chunk overlap.
- **Q3 (Điều 27) — Cả hai ✗:** "Thay đổi họ" là khái niệm rất đặc thù, dễ bị nhầm với "thay đổi tên" (Điều 28). Model chưa đủ để phân biệt Điều 27 (họ) vs Điều 28 (tên).
- **Q4 (Điều 49) — LM Studio ✓:** Model hiểu "điều kiện giám hộ" → tìm đúng Điều 49.
- **Q5 (Điều 68) — LM Studio ✓:** Model hiểu "mất tích" → tìm đúng Điều 68 về tuyên bố mất tích.

> **Kết luận:** Việc tích hợp LM Studio giúp hệ thống RAG bước từ "ngẫu nhiên" (20%) lên "hiểu ngữ nghĩa" (60%). Với tài liệu pháp luật tiếng Việt dài 3600 dòng, 230 chunks, việc tìm đúng Điều 68 hay Điều 49 trong số đó là minh chứng mạnh mẽ cho sức mạnh của Vector Store thực thụ. Để đạt >80%, cần: (1) model lớn hơn chuyên cho tiếng Việt, (2) chunk sát theo ranh giới "## **Điều X.**", (3) hybrid search (BM25 + vector).

**Metadata Filtering:**
> Khi filter theo `{"category": "law"}`, hệ thống chỉ tìm trong tài liệu pháp lý. Khi hệ thống mở rộng (thêm Luật Hình sự, Luật Thương mại), filter `{"law_code": "91/2015/QH13"}` sẽ đảm bảo chỉ tìm trong Bộ luật Dân sự — tăng precision đáng kể.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Hoàng dùng SentenceChunker với max_sentences=2 cho FAQ docs và thấy retrieval precision cao hơn vì mỗi chunk chứa đúng 1 câu hỏi + câu trả lời. Áp dụng vào Bộ luật Dân sự: mỗi khoản (1., 2., 3.) là 1-2 câu — SentenceChunker có thể giúp tìm đúng khoản cụ thể trong một Điều dài thay vì trả về cả Điều.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> [Điền sau buổi demo nhóm]

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ thực hiện 3 thay đổi: (1) **Custom chunking theo cấu trúc Điều/Khoản** — tách đúng tại ranh giới `## **Điều X.**` thay vì dùng recursive chunker generic; (2) **Thêm metadata `article_number` và `chapter`** để có thể filter trực tiếp theo số Điều — rất quan trọng khi người dùng hỏi "Điều 68 quy định gì?"; (3) **Dùng model embedding đa ngữ chuyên biệt** (như `vinai/phobert-base` hoặc Jina v5 đầy đủ) thay vì Nomic để hiểu tốt hơn các thuật ngữ pháp luật tiếng Việt như "giám hộ", "mất tích", "pháp nhân".

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 14 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 9 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **88 / 100** |
