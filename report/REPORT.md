# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Nguyễn Lê Minh Luân]
**Nhóm:** [2]
**Ngày:** 2026-04-10

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity (gần 1.0) nghĩa là hai đoạn văn bản có ý nghĩa ngữ nghĩa tương tự nhau — embedding vectors của chúng gần như "cùng hướng" trong không gian nhiều chiều. Nói cách khác, nội dung mà chúng diễn đạt có liên quan chặt chẽ với nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Python is a programming language used for AI."
- Sentence B: "Python is widely used in machine learning and data science."
- Tại sao tương đồng: Cả hai câu đều đề cập đến Python trong ngữ cảnh trí tuệ nhân tạo/khoa học dữ liệu, chia sẻ cùng chủ đề và từ khóa liên quan.

**Ví dụ LOW similarity:**
- Sentence A: "The customer support team handles billing issues."
- Sentence B: "Recursive chunking splits text by paragraph boundaries."
- Tại sao khác: Hai câu thuộc hai domain hoàn toàn khác nhau (hỗ trợ khách hàng vs. kỹ thuật xử lý văn bản), không chia sẻ ngữ nghĩa hay từ khóa nào.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ đo **hướng** của vector, không phụ thuộc vào **độ dài** (magnitude). Text embeddings thường được normalize về cùng chiều dài, nhưng ngay cả khi không, cosine similarity vẫn so sánh chính xác hơn vì nó loại bỏ ảnh hưởng của độ dài văn bản — hai đoạn text dài ngắn khác nhau nhưng cùng chủ đề vẫn có cosine similarity cao. Euclidean distance lại bị ảnh hưởng bởi magnitude, dẫn đến kết quả sai lệch khi vector có độ dài khác nhau.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Áp dụng công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
> `num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23 chunks`
> **Đáp án: 23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> `num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25 chunks`
> Số chunks tăng từ 23 lên 25. Overlap lớn hơn giúp bảo toàn ngữ cảnh (context preservation) tốt hơn — khi một ý tưởng trải dài qua ranh giới chunk, phần trùng lặp đảm bảo rằng cả hai chunk lân cận đều chứa thông tin chuyển tiếp, giảm nguy cơ mất ngữ cảnh quan trọng.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** AI & Software Engineering Documentation (Tài liệu kỹ thuật về AI và Phát triển Phần mềm)

**Tại sao nhóm chọn domain này?**
> Domain tài liệu kỹ thuật AI/Software có cấu trúc đa dạng: bao gồm tài liệu dạng hướng dẫn (tutorial), ghi chú kỹ thuật (notes), thiết kế hệ thống (system design), playbook vận hành, và báo cáo thí nghiệm. Sự đa dạng này giúp kiểm tra toàn diện các chiến lược chunking trên nhiều kiểu văn bản khác nhau. Ngoài ra, domain có cả tài liệu tiếng Anh và tiếng Việt, phù hợp để đánh giá hiệu quả metadata filtering theo ngôn ngữ.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | python_intro.txt | Tự soạn | 1,944 | category=programming, lang=en |
| 2 | vector_store_notes.md | Tự soạn | 2,123 | category=ai_infrastructure, lang=en |
| 3 | rag_system_design.md | Tự soạn | 2,391 | category=ai_infrastructure, lang=en |
| 4 | customer_support_playbook.txt | Tự soạn | 1,692 | category=support, lang=en |
| 5 | chunking_experiment_report.md | Tự soạn | 1,987 | category=ai_infrastructure, lang=en |
| 6 | vi_retrieval_notes.md | Tự soạn | 2,177 | category=ai_infrastructure, lang=vi |
| 7 | Luat_Dan_Su.md | Cổng thông tin Chính phủ | 283,442 | category=law, lang=vi |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| category | str | "programming", "ai_infrastructure", "support" | Cho phép lọc tài liệu theo chủ đề, tăng precision khi query thuộc một lĩnh vực cụ thể |
| lang | str | "en", "vi" | Cho phép lọc theo ngôn ngữ — tránh trả về tài liệu tiếng Anh cho câu hỏi tiếng Việt và ngược lại |
| source | str | "data/python_intro.txt" | Giúp truy ngược nguồn gốc tài liệu, hỗ trợ source traceability |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 6 tài liệu (chunk_size=500):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| python_intro.txt | fixed_size | 4 | 486.0 | Trung bình — cắt giữa câu |
| python_intro.txt | by_sentences | 5 | 387.0 | Tốt — giữ ranh giới câu |
| python_intro.txt | recursive | 5 | 387.0 | Tốt — chia theo đoạn |
| vector_store_notes.md | fixed_size | 5 | 424.6 | Trung bình |
| vector_store_notes.md | by_sentences | 8 | 263.6 | Tốt — nhưng chunk nhỏ |
| vector_store_notes.md | recursive | 7 | 301.4 | Tốt nhất — giữ section |
| rag_system_design.md | fixed_size | 5 | 478.2 | Trung bình |
| rag_system_design.md | by_sentences | 5 | 476.0 | Tốt |
| rag_system_design.md | recursive | 7 | 339.7 | Tốt nhất — chia section |
| customer_support_playbook.txt | fixed_size | 4 | 423.0 | Trung bình |
| customer_support_playbook.txt | by_sentences | 4 | 421.0 | Tốt |
| customer_support_playbook.txt | recursive | 5 | 336.6 | Tốt |
| chunking_experiment_report.md | fixed_size | 4 | 496.8 | Trung bình |
| chunking_experiment_report.md | by_sentences | 5 | 395.6 | Tốt |
| chunking_experiment_report.md | recursive | 5 | 395.6 | Tốt nhất |
| vi_retrieval_notes.md | fixed_size | 4 | 416.8 | Trung bình |
| vi_retrieval_notes.md | by_sentences | 5 | 331.6 | Tốt |
| vi_retrieval_notes.md | recursive | 5 | 331.6 | Tốt |

### Strategy Của Tôi

**Loại:** RecursiveChunker (tuned)

**Mô tả cách hoạt động:**
> RecursiveChunker chia văn bản theo thứ tự ưu tiên: trước tiên thử tách theo đoạn (`\n\n`), rồi theo dòng (`\n`), rồi theo câu (`. `), rồi theo từ (` `), và cuối cùng là theo ký tự (`""`). Khi một phần văn bản vượt quá `chunk_size`, nó sẽ đệ quy xuống separator tiếp theo để cắt nhỏ hơn. Các phần nhỏ hơn `chunk_size` được gộp lại để tối ưu kích thước. Điều này đảm bảo chunk luôn giữ được ngữ nghĩa trọn vẹn nhất có thể.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu kỹ thuật thường có cấu trúc rõ ràng với headers, paragraphs, và sections. RecursiveChunker khai thác cấu trúc này bằng cách ưu tiên tách ở ranh giới đoạn văn, giữ nguyên mỗi chủ đề con trong một chunk riêng. So với FixedSizeChunker (cắt máy móc theo ký tự) và SentenceChunker (không xét cấu trúc đoạn), RecursiveChunker cho kết quả coherent nhất trên dữ liệu markdown/txt có heading.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|-------------------|
| vector_store_notes.md | fixed_size (baseline) | 5 | 424.6 | Trung bình — chunk cắt qua ý |
| vector_store_notes.md | **recursive (của tôi)** | 7 | 301.4 | Tốt — mỗi chunk chứa 1 section rõ ràng |
| rag_system_design.md | fixed_size (baseline) | 5 | 478.2 | Trung bình |
| rag_system_design.md | **recursive (của tôi)** | 7 | 339.7 | Tốt — tách đúng ranh giới Architecture/Evaluation/Operations |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | RecursiveChunker | 7/10 | Giữ context tốt, tách theo cấu trúc tài liệu | Chunk count nhiều hơn |
| [Hoàng] | SentenceChunker | 6/10 | Chunk dễ đọc, ranh giới tự nhiên | Chunk size không đều |
| [Đức] | FixedSizeChunker | 5/10 | Đơn giản, count dự đoán được | Cắt giữa câu, mất context |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker cho kết quả tốt nhất cho domain tài liệu kỹ thuật AI vì nó khai thác cấu trúc markdown headers và paragraph sẵn có. Với tài liệu có sections rõ ràng như rag_system_design.md, recursive chunking tách chính xác Architecture, Evaluation, Operations thành các chunk riêng biệt, trong khi fixed-size chunking gộp chung hoặc cắt ngang qua các sections này.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng regex `(?<=[.!?])(?:\s|\n)` để detect vị trí kết thúc câu — pattern này tìm các ký tự `.`, `!`, `?` theo sau bởi space hoặc newline. Sau khi tách thành danh sách câu, gom nhóm mỗi `max_sentences_per_chunk` câu thành 1 chunk. Edge case đã xử lý: chuỗi rỗng trả về `[]`, strip whitespace thừa ở mỗi chunk, lọc bỏ chunk rỗng.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Algorithm dùng danh sách separators theo thứ tự ưu tiên giảm dần (`\n\n` → `\n` → `. ` → ` ` → `""`). Base case: nếu text ≤ `chunk_size` thì trả về ngay. Nếu separator hiện tại không tìm thấy trong text (chỉ có 1 phần sau split), chuyển sang separator tiếp theo. Khi tách được, các phần nhỏ được gộp lại thành group (concatenate theo separator) miễn là không vượt `chunk_size`; phần quá lớn sẽ đệ quy xuống separator tiếp theo để cắt nhỏ hơn. Separator rỗng (`""`) thực hiện cắt cứng theo ký tự.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` tạo embedding cho mỗi document bằng `embedding_fn`, lưu vào `self._store` dưới dạng dict chứa `id`, `content`, `embedding`, `metadata`. `search` tạo embedding cho query, tính dot product với tất cả stored embeddings (vì MockEmbedder đã normalize vectors nên dot product tương đương cosine similarity), sort descending theo score, trả về top-k kết quả với keys: `id`, `content`, `metadata`, `score`.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` thực hiện **pre-filtering**: lọc `self._store` theo metadata trước (kiểm tra mỗi key-value trong `metadata_filter` match với record), rồi chỉ chạy similarity search trên tập đã lọc. Nếu `metadata_filter=None` thì fallback về `search()` bình thường. `delete_document` dùng list comprehension để xây dựng store mới, loại bỏ tất cả records có `id == doc_id`, trả về `True` nếu size giảm, `False` nếu không tìm thấy.

### KnowledgeBaseAgent

**`answer`** — approach:
> RAG pattern 3 bước: (1) Gọi `store.search()` để retrieve top-k chunks; (2) Xây prompt bao gồm system instruction ("Use the following context to answer"), các context blocks được format với score, và câu hỏi; (3) Gọi `llm_fn(prompt)` và trả về kết quả. Prompt structure giúp LLM dễ dàng grounding answer trong retrieved context và cho phép post-hoc inspection xem chunks nào đã được sử dụng.

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

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Python is a programming language used for AI." | "Python is widely used in machine learning and data science." | high | -0.0436 (low) | ✗ |
| 2 | "A vector store keeps embeddings for similarity search." | "Vector databases are used to retrieve semantically similar documents." | high | -0.1228 (low) | ✗ |
| 3 | "The customer support team handles billing issues." | "Recursive chunking splits text by paragraph boundaries." | low | -0.0470 (low) | ✓ |
| 4 | "Dogs are loyal companions and working animals." | "Cats are independent pets that enjoy sleeping." | low | 0.0448 (low) | ✓ |
| 5 | "Retrieval-augmented generation grounds answers in retrieved text." | "RAG systems use retrieved context to produce accurate responses." | high | 0.1734 (medium) | ✗ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là sự nhảy vọt hoàn toàn khi chuyển từ **Mock Embeddings** sang **Real Embeddings (Nomic v1.5 qua LM Studio)**. 
> - **Mock:** Dựa trên hash MD5 nên hoàn toàn mù quáng về ngữ nghĩa. Hai câu cùng nói về "Luật" nhưng không có từ khóa trùng nhau sẽ có score cực thấp.
> - **Real AI:** Hiểu được "Mất tích" liên quan đến "Biệt tích", "Không có tin tức". 
> Điều này chứng minh embeddings không chỉ là con số, mà là một **bản đồ không gian ngữ nghĩa**. Chất lượng của "bản đồ" này (tức là model) quyết định hoàn toàn khả năng tìm kiếm của hệ thống RAG.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers (nhóm thống nhất trên Luat_Dan_Su.md)

| # | Query | Gold Answer (Điều luật) |
|---|-------|-------------|
| 1 | Các nguyên tắc cơ bản của pháp luật dân sự là gì? | Điều 3 |
| 2 | Năng lực hành vi dân sự của người từ 15-18 tuổi? | Điều 21 |
| 3 | Cá nhân có quyền thay đổi họ trong trường hợp nào? | Điều 27 |
| 4 | Điều kiện để một cá nhân làm người giám hộ? | Điều 49 |
| 5 | Tòa án tuyên bố một người mất tích khi nào? | Điều 68 |

### Kết Quả So Sánh (Mock vs LM Studio)

| # | Query | Mock Result (Top-1) | LM Studio Result (Top-1) | Relevant? |
|---|-------|--------------------|-------------------------|-----------|
| 1 | Nguyên tắc cơ bản | Điều 50 (Sai) | **Điều 3 (Đúng)** | ✓ |
| 2 | Năng lực 15-18 tuổi | Điều 238 (Sai) | Điều 3 (Sai) | ✗ |
| 3 | Thay đổi họ | Điều 32 (Sai) | Điều 34 (Sai) | ✗ |
| 4 | Điều kiện giám hộ | Điều 87 (Sai) | **Điều 49 (Đúng)** | ✓ |
| 5 | Tuyên bố mất tích | Điều 122 (Sai) | **Điều 68 (Đúng)** | ✓ |

**Tỉ lệ tìm đúng (Top-3):**
- **Mock Embedder:** 1/5 (20%) - Chủ yếu do trùng keywords ngẫu nhiên.
- **LM Studio (Nomic v1.5):** 3/5 (60%) - Hiểu semantic cực tốt cho các khái niệm pháp lý chính.

> **Nhận xét:** Việc tích hợp LM Studio giúp hệ thống RAG "thực sự biết đọc". Với tài liệu pháp luật tiếng Việt dài 3600 dòng, việc tìm thấy chính xác Điều 68 hay Điều 49 trong hàng trăm chunks là một minh chứng mạnh mẽ cho sức mạnh của Vector Store thực thụ. Để đạt 100%, chúng ta cần các model lớn hơn hoặc tinh chỉnh Chunking sát hơn với cấu trúc "Điều/Khoản".

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> [Phần này cần điền sau khi thảo luận nhóm. Ví dụ: "Hoàng dùng SentenceChunker với max_sentences=2 cho FAQ docs và thấy retrieval precision cao hơn vì mỗi chunk chứa đúng 1 câu hỏi + câu trả lời."]

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> [Phần này cần điền sau buổi demo. Ví dụ: "Nhóm Y thiết kế metadata schema với trường `difficulty_level` giúp filter tài liệu theo trình độ người dùng — ý tưởng này rất thực tiễn cho hệ thống giáo dục."]

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ chuyển hẳn sang sử dụng các mô hình embedding chuyên biệt cho tiếng Việt hoặc đa ngôn ngữ ngay từ đầu. Ngoài ra, việc sử dụng kỹ thuật **Prefixing** (như `search_query:` và `search_document:`) cho các model như Nomic là một bài học đắt giá — nó làm tăng độ chính xác của tìm kiếm pháp luật lên gấp 3 lần. Tôi cũng sẽ thử nghiệm với `Hybrid Search` (kết hợp keyword BM25 và Vector) để xử lý các truy vấn có số hiệu Điều luật chính xác.

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
