# Lab 21 — Evaluation Report

**Họ tên**: Đặng Minh Quang **MSSV**: 2A202601108 **Ngày**: 21/08/2026
**Tier**: `T4` **Base model**: `unsloth/Qwen3.5-4B` **GPU thực tế**: `Tesla T4, 14.6 GB khả dụng`

> Mọi con số dưới đây được chép từ các artefact trong `results/` của lượt đánh giá đầy đủ.

---

## 1. Setup

|                    |                                                                   |
| ------------------ | ----------------------------------------------------------------- |
| Dataset            | 250 ticket chăm sóc khách hàng tiếng Việt → JSON triage 4 trường  |
| Train / val        | 225 / 25 (seed 42)                                                |
| `max_length`       | 1024 theo tier T4 — p95 đo được là 98; mức gợi ý tối thiểu là 256 |
| `MASK_MODE`        | `assistant-only`                                                  |
| Epochs / max_steps | 2 epoch / 30 bước                                                 |

**Template có giữ khối `<think>` không?** Có. `template_check.json` xác nhận cả thẻ mở lẫn nội dung reasoning vẫn xuất hiện trong chuỗi sau khi render. Vì corpus mặc định chỉ chứa câu trả lời JSON, tôi giữ `MASK_MODE=assistant-only`; nếu chuyển sang dữ liệu có reasoning trace, kết quả kiểm tra này cho thấy template vẫn bảo toàn phần trace để có thể áp dụng chiến lược mask phù hợp.

---

## 2. Mask proof (NB1)

|                              |        |
| ---------------------------- | ------ |
| `supervised_fraction`        | 0.4149 |
| Câu trả lời nằm trong loss   | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn đầu của phần được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}
<|im_end|>
```

Kết quả giải mã ngược cho thấy chỉ phần trả lời của assistant tham gia tính loss, còn system prompt và ticket của người dùng đều đã được mask. Hai điều kiện cốt lõi vì thế cùng được thỏa mãn: câu trả lời nằm trong vùng giám sát và câu hỏi không bị mô hình học lại như một phần của đầu ra.

---

## 3. Ba baseline (NB2 — đo trước khi train)

| Run                         | target | regression | format | latency (ms) |
| --------------------------- | -----: | ---------: | -----: | -----------: |
| (a) base + naive prompt     | 0.0000 |     0.7578 | 0.0000 |       3508.9 |
| (b) base + optimized prompt | 0.7650 |     0.7578 | 1.0000 |       1117.9 |
| (c) LoRA fine-tune          | 0.9700 |     0.6111 | 1.0000 |       1481.3 |

**(b) có thật sự mạnh hơn (a) không?** Có. Chỉ riêng việc dùng prompt tối ưu đã nâng target từ 0 lên 0.765 và format từ 0 lên 1.0, trong khi regression giữ nguyên ở 0.7578. Tôi không chỉnh sửa `OPTIMIZED_PROMPT`, nhờ đó baseline (b) vẫn là một mốc so sánh được đóng băng và không bị làm yếu sau khi quan sát kết quả fine-tune. Latency của (b) cũng thấp hơn đáng kể so với (a), cho thấy việc mô tả rõ schema và miền giá trị không chỉ cải thiện độ chính xác mà còn hạn chế đầu ra lan man.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run         | vị trí      |             r |  trainable |   LR | train loss | **target** |      s | VRAM GB |
| ----------- | ----------- | ------------: | ---------: | ---: | ---------: | ---------: | -----: | ------: |
| `correct`   | text-linear |            16 | 32,464,896 | 1e-4 |     0.6253 |       0.97 |  969.1 |   12.01 |
| `attn_only` | q,v         | 283 (matched) | 32,456,704 | 1e-4 |     0.5368 |       0.97 |  856.1 |   12.02 |
| `wrong_lr`  | text-linear |            16 | 32,464,896 | 1e-5 |     1.5704 |       0.00 | 1004.3 |   12.01 |
| `qlora`     | text-linear |            16 | 32,464,896 | 1e-4 |     0.7058 |       0.94 | 1064.9 |    7.09 |

**4.1 — So sánh `attn_only` và `correct`.** Hai cấu hình hòa nhau trên tập target ở mức 0.97, dù `attn_only` có train loss thấp hơn, 0.5368 so với 0.6253. Vì vậy thứ tự theo loss không giống thứ tự theo thước đo tác vụ: loss gợi ý `attn_only` tốt hơn nhưng target chỉ cho kết quả hòa. Rank rất lớn 283 giúp `attn_only` khớp gần như chính xác ngân sách tham số của `correct`, nhưng không tạo ra lợi thế target; kết quả cho thấy không thể xem rank tách rời vị trí gắn adapter, và phải so sánh hai cấu hình dưới cùng ngân sách tham số.

**4.2 — Ảnh hưởng của learning rate.** `wrong_lr` chỉ giảm LR từ 1e-4 xuống 1e-5 nhưng loss cuối tăng từ 0.6253 lên 1.5704, đồng thời target sụp từ 0.97 xuống 0 và format cũng bằng 0. Với cùng 30 bước, LR kiểu full fine-tune quá nhỏ khiến LoRA chưa học đủ hành vi đầu ra. Nếu chỉ nhìn loss mà không biết LR, tôi có thể kết luận sai rằng dữ liệu khó, kiến trúc LoRA không phù hợp hoặc cần thêm epoch, trong khi nguyên nhân trực tiếp là kích thước bước cập nhật không phù hợp.

**4.3 — Đánh đổi của QLoRA.** QLoRA dùng 7.09 GB thay vì 12.01 GB, tiết kiệm 4.92 GB, tương đương khoảng 41.0% VRAM. Đổi lại target giảm 0.03, loss tăng 0.0805 và thời gian train tăng khoảng 95.8 giây so với cấu hình đúng. Số đo này ủng hộ khuyến nghị không chọn QLoRA làm mặc định cho Qwen3.5 khi T4 vẫn chứa được LoRA 16-bit: lợi ích bộ nhớ là rõ ràng nhưng chất lượng và tốc độ đều kém hơn. Tuy nhiên target 0.94 vẫn cao, nên QLoRA có thể là phương án thực dụng khi giới hạn VRAM nghiêm ngặt.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.147` · `valid_trace_rate = 0.00`

Fine-tune tạo ra cải thiện lớn trên nhiệm vụ đích: target tăng từ 0.765 của baseline prompt tốt lên 0.970, đồng thời giữ format JSON ở mức 1.0. Tuy nhiên model bị giảm năng lực chung: regression từ 0.7578 xuống 0.6111, tức giảm 0.1467, lớn hơn nhiều so với dung sai 0.02. Do đó verdict FAILED là đúng dù điểm target cao. Kết quả cho thấy tối ưu một corpus hẹp gồm 250 ticket có thể làm hành vi mới lấn át năng lực trả lời chung. `valid_trace_rate=0` không phải bằng chứng duy nhất về lỗi ở đây vì corpus mặc định không huấn luyện reasoning trace, nhưng vẫn cần được ghi nhận. Tôi không nới cổng đánh giá để biến kết quả thành PASSED; hướng sửa hợp lý là thêm khoảng 1–5% replay data đại diện cho năng lực chung, rồi huấn luyện và đo lại trên cùng tập regression đã đóng băng.

---

## 6. Phân tích định tính — gồm cả ca đúng và ca sai

`qualitative.json` chỉ lưu dự đoán fine-tune theo từng mẫu, không lưu đầu ra chi tiết của baseline (b). Vì vậy tôi để rõ cột (b) là “không lưu prediction-level” thay vì tái dựng hoặc suy đoán dữ liệu không có trong artefact. Trong phạm vi bảng dưới đây, “đúng” nghĩa là fine-tune khớp đủ 4/4 trường với nhãn thật; “sai” nghĩa là fine-tune chỉ khớp 3/4 trường. Do thiếu prediction-level của (b), bảng này không được dùng để khẳng định fine-tune thắng hay thua (b) trên từng mẫu.

| #   | Ticket (rút gọn)                                     | Nhãn đúng                                             | (b) prompt                 | (c) fine-tune                         | Nhận xét          |
| --- | ---------------------------------------------------- | ----------------------------------------------------- | -------------------------- | ------------------------------------- | ----------------- |
| 1   | Trả chuột không dây VN232232, gấp, shop hỗ trợ tốt   | `doi_tra, cao, chuột không dây, tich_cuc`             | Không lưu prediction-level | Khớp nhãn 4/4                         | ✅ Đúng hoàn toàn |
| 2   | Hoàn tiền ốp lưng VN812931, sớm, bực mình            | `hoan_tien, trung_binh, ốp lưng điện thoại, tieu_cuc` | Không lưu prediction-level | Khớp nhãn 4/4                         | ✅ Đúng hoàn toàn |
| 3   | Bình giữ nhiệt VN804124 chưa thấy tiền, khi nào tiện | `hoan_tien, thap, bình giữ nhiệt, tich_cuc`           | Không lưu prediction-level | Dự đoán `urgency=trung_binh`, đạt 3/4 | ❌ Sai urgency    |
| 4   | Nồi chiên DH249548 thiếu phụ kiện, khi nào tiện      | `san_pham_loi, thap, nồi chiên không dầu, trung_tinh` | Không lưu prediction-level | Dự đoán `urgency=trung_binh`, đạt 3/4 | ❌ Sai urgency    |
| 5   | Áo khoác VN613097 bị lỗi, khi nào tiện, cảm ơn shop  | `san_pham_loi, thap, áo khoác gió, tich_cuc`          | Không lưu prediction-level | Dự đoán `urgency=trung_binh`, đạt 3/4 | ❌ Sai urgency    |

Mẫu lỗi chung ở cả ba trường hợp là mô hình đều nâng `urgency` từ `thap` lên `trung_binh`, trong khi intent và product vẫn đúng. Có vẻ các tín hiệu giảm mức khẩn cấp như “khi nào tiện” chưa được học đủ ổn định, đặc biệt khi ticket đồng thời chứa một vấn đề cần xử lý như hoàn tiền hoặc sản phẩm lỗi. Kết quả này gợi ý tập train cần thêm các cách diễn đạt đa dạng cho nhóm urgency thấp và cần kiểm tra lại phân bố nhãn urgency trước lần huấn luyện tiếp theo.

---

## 7. Kết luận và điều tôi học được

**Kết luận.** Tôi chưa nên deploy trực tiếp adapter này. Model đạt 0.97 target và luôn trả đúng định dạng JSON, nên fine-tune đã học nhiệm vụ triage rất tốt; tuy nhiên regression giảm 0.147 là một mức suy giảm năng lực chung không thể chấp nhận so với dung sai 0.02. Nếu hệ thống chỉ làm triage trong một dịch vụ cô lập, adapter có thể được thử nghiệm giới hạn sau khi bổ sung giám sát, nhưng chưa đủ an toàn cho một trợ lý dùng chung. Thí nghiệm đối chứng cho thấy learning rate là đòn bẩy nhạy nhất trong lượt chạy này: chỉ giảm từ 1e-4 xuống 1e-5 đã làm target về 0. Vị trí adapter và rank phải được xem trong cùng ngân sách; `attn_only` rank 283 hòa target với `correct` nhưng loss thấp hơn, chứng minh train loss không thể thay thế đánh giá tác vụ. Mask đã hoạt động đúng nên không phải nguyên nhân của regression. Rủi ro chính nằm ở dữ liệu quá hẹp và thiếu replay data, khiến model chuyên môn hóa mạnh. Bước tiếp theo phải là thêm dữ liệu duy trì năng lực chung, cân bằng thêm các ca urgency thấp, rồi đánh giá lại bằng đúng cổng đã đóng băng.

**Ba điều tôi học được:**

1. Prompt tối ưu là baseline bắt buộc: riêng prompt đã nâng target từ 0 lên 0.765, nên chỉ so fine-tune với naive prompt sẽ phóng đại lợi ích.
2. Loss thấp không bảo đảm target cao hơn; `attn_only` có loss tốt hơn `correct` nhưng hai cấu hình hòa nhau ở 0.97.
3. Fine-tune có thể thắng nhiệm vụ đích mà vẫn không đạt điều kiện triển khai; mức tăng target +0.205 không bù được regression -0.147.

**Nếu có thêm 2 giờ, tôi sẽ thử:** thêm 1–5% replay data từ tập năng lực chung cùng các ca urgency thấp, giữ nguyên split/eval và cấu hình LoRA đúng, sau đó train lại một lượt và kiểm tra xem regression có nằm trong dung sai 0.02 mà không làm mất mức target 0.97 hay không.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link: chưa thực hiện
