# K3 — Ngày 1: Bài Tập & Phản Ánh

## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature

Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)

> Khi temperature ở mức thấp, mô hình thường đưa ra các câu trả lời ổn định, ngắn gọn và bám sát những cách diễn đạt phổ biến. Khi temperature tăng, câu trả lời vẫn giữ được ý chính nhưng cách diễn đạt trở nên đa dạng, tự nhiên và sáng tạo hơn. Tuy nhiên, nếu temperature quá cao, phản hồi có thể dài dòng, lan man và đôi khi thiếu nhất quán.

### Câu 1.2 — Chọn temperature cho sản phẩm

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**

> Đối với chatbot hỗ trợ khách hàng, tôi sẽ chọn temperature khoảng 0.2–0.3. Ở mức này, mô hình tạo ra các câu trả lời ổn định, nhất quán và bám sát thông tin được cung cấp, giúp hạn chế việc tự suy diễn hoặc tạo ra nội dung không chính xác. Điều này đặc biệt quan trọng trong các tình huống như giải đáp chính sách, hướng dẫn sử dụng hoặc xử lý khiếu nại, nơi người dùng cần thông tin đúng và đáng tin cậy hơn là sự sáng tạo.

### Câu 1.3 — Đánh đổi chi phí

Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

> Với khối lượng công việc trên:

- 10.000 người dùng/ngày × 3 lần gọi API = 30.000 lượt gọi/ngày.
- Mỗi lần sinh khoảng 350 token đầu ra, nên tổng số token đầu ra là 10,5 triệu token/ngày.

Theo bảng giá đã cho (trên 1.000 token):

- GPT-4o: 0,010 USD / 1.000 output token.
- GPT-4o-mini: 0,0006 USD / 1.000 output token.

Do đó:

- Chi phí output của GPT-4o: 10.500 × 0,010 = 105 USD/ngày.
- Chi phí output của GPT-4o-mini: 10.500 × 0,0006 = 6,3 USD/ngày.

Như vậy, GPT-4o có chi phí cao hơn khoảng 16,7 lần so với GPT-4o-mini cho cùng khối lượng công việc.

GPT-4o xứng đáng với chi phí khi cần khả năng suy luận và độ chính xác cao, chẳng hạn như phân tích tài liệu chuyên môn, hỗ trợ lập trình hoặc xử lý các bài toán phức tạp.

Trong khi đó, GPT-4o-mini phù hợp với các ứng dụng có số lượng người dùng lớn như chatbot chăm sóc khách hàng, hệ thống hỏi đáp (FAQ) hoặc trợ lý tra cứu thông tin, nơi yêu cầu phản hồi nhanh và tối ưu chi phí.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona

Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:

- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)

> Hai phản hồi khác nhau rõ rệt về cách diễn đạt và mức độ chi tiết. Với vai trò giáo viên tiểu học, mô hình sử dụng câu ngắn, từ ngữ đơn giản và các ví dụ gần gũi như sổ ghi chép hoặc quyển vở để giúp trẻ 8 tuổi dễ hiểu. Ngược lại, với vai trò chuyên gia tài chính, câu trả lời dài hơn, sử dụng nhiều thuật ngữ kỹ thuật như distributed ledger, consensus mechanism, hash function và đề cập đến các ứng dụng trong tài chính. Điều này cho thấy system prompt định hướng hành vi của mô hình, quyết định phong cách diễn đạt, mức độ chi tiết, đối tượng hướng đến và loại kiến thức được ưu tiên trong câu trả lời.

### Câu 2.2 — tiktoken vs đếm từ

Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**

> Khi thử với một đoạn văn tiếng Việt khoảng 100 từ, `tiktoken` cho kết quả khoảng 160 token, trong khi cách ước lượng theo công thức `số từ / 0,75` cho khoảng 133 token. Như vậy, hai kết quả chênh lệch khoảng 20% (\frac{160-133}{133}\times100%).
> Nguyên nhân là tokenizer của các mô hình LLM được tối ưu chủ yếu cho tiếng Anh. Tiếng Việt có nhiều ký tự có dấu, từ ghép và các âm tiết được tách bằng dấu cách nên một từ hoặc một ký tự có dấu có thể bị chia thành nhiều token hơn. Vì vậy, với cùng độ dài nội dung, văn bản tiếng Việt thường tiêu tốn nhiều token hơn so với tiếng Anh.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming

**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)

> Streaming đặc biệt quan trọng trong các ứng dụng cần phản hồi nhanh như chatbot, trợ lý AI hoặc sinh nội dung dài, vì người dùng có thể đọc kết quả ngay khi mô hình bắt đầu tạo câu trả lời thay vì phải chờ hoàn thành toàn bộ. Điều này giúp giảm cảm giác chờ đợi và cải thiện trải nghiệm sử dụng. Ngược lại, non-streaming phù hợp khi cần nhận toàn bộ kết quả cùng một lúc, chẳng hạn như trả về dữ liệu dưới dạng JSON, tạo báo cáo hoàn chỉnh hoặc các tác vụ mà ứng dụng phải xử lý toàn bộ phản hồi trước khi hiển thị cho người dùng.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?

**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**

> Exponential backoff giúp giảm tải cho API khi hệ thống đang quá tải bằng cách tăng dần thời gian chờ sau mỗi lần retry (ví dụ: 1 giây, 2 giây, 4 giây, 8 giây), tạo điều kiện để máy chủ có thời gian phục hồi. So với việc luôn chờ một khoảng thời gian cố định, cách này làm giảm số lượng yêu cầu gửi đồng thời và tăng khả năng retry thành công. Nếu hàng nghìn client đều retry sau cùng một khoảng thời gian cố định, chúng có thể gửi yêu cầu cùng lúc, tạo ra hiện tượng retry storm hoặc thundering herd, khiến máy chủ tiếp tục bị quá tải và khó phục hồi.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona

**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**

System prompt:

> Bạn là trợ lý AI hỗ trợ học tập về Trí tuệ nhân tạo và Khoa học máy tính. Hãy trả lời bằng tiếng Việt, giải thích rõ ràng, chính xác và dễ hiểu. Khi cần, hãy đưa ra ví dụ minh họa hoặc đoạn mã ngắn để làm rõ khái niệm. Nếu không chắc chắn về thông tin, hãy nêu rõ giới hạn thay vì suy đoán.
> Tôi yêu cầu "trả lời bằng tiếng Việt" để phù hợp với người dùng mục tiêu và giúp việc tiếp thu kiến thức dễ dàng hơn. Cụm từ "nêu rõ giới hạn thay vì suy đoán" giúp mô hình ưu tiên tính chính xác, hạn chế đưa ra thông tin không có cơ sở, từ đó tăng độ tin cậy của câu trả lời.

### Câu 4.2 — Hạn chế & cải thiện

**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**

> Hạn chế lớn nhất của trợ lý hiện tại là không có bộ nhớ dài hạn, nên chỉ dựa vào lịch sử hội thoại ngắn để trả lời. Khi cuộc trò chuyện kéo dài hoặc diễn ra ở nhiều phiên khác nhau, trợ lý có thể quên các thông tin quan trọng như sở thích, mục tiêu hoặc ngữ cảnh trước đó của người dùng.
> Một cải thiện phù hợp là bổ sung cơ chế long-term memory. Cụ thể, sau mỗi cuộc hội thoại, hệ thống sẽ trích xuất các thông tin quan trọng của người dùng (ví dụ: sở thích, dự án đang thực hiện, mục tiêu học tập) và lưu dưới dạng embedding trong một vector database. Khi có câu hỏi mới, trợ lý sẽ truy xuất (retrieve) các thông tin liên quan rồi đưa chúng vào prompt trước khi gọi mô hình, giúp câu trả lời nhất quán và cá nhân hóa hơn.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
