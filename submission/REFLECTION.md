# Reflection — Lab 21

_Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài._

**1. Điều gì làm bạn ngạc nhiên nhất?**

Điều làm tôi ngạc nhiên nhất là hiệu quả của prompt engineering. Chỉ cần chuyển từ prompt đơn giản sang prompt mô tả rõ schema và các giá trị hợp lệ, target của base model đã tăng từ 0 lên 0.765, còn format tăng từ 0 lên 1.0. Fine-tune tiếp tục nâng target lên 0.97, nhưng regression lại giảm từ 0.7578 xuống 0.6111. Kết quả này cho thấy điểm cao trên nhiệm vụ đích chưa đủ để kết luận một model có thể triển khai an toàn.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Tôi mất nhiều thời gian nhất ở NB4 vì phải huấn luyện ba cấu hình đối chứng riêng. Phần này mất khoảng 51,6 phút trong tổng thời gian khoảng 76,8 phút của core pipeline. Ban đầu tôi nghĩ khâu chuẩn bị dữ liệu hoặc NB3 sẽ tốn thời gian nhất, nhưng thực tế phần lớn thời gian được dành cho việc chờ GPU và kiểm tra kết quả của từng cấu hình. Tôi cũng mất thêm thời gian xử lý sự khác biệt giữa Jupyter local và Google Colab.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Trước đây tôi nghĩ fine-tune có thể được xem là thành công nếu train loss giảm và điểm trên nhiệm vụ đích tăng. Sau lab này, tôi không còn tin rằng hai chỉ số đó là đủ. attn_only có train loss thấp hơn correct, nhưng cả hai cùng đạt target 0.97. Fine-tune cũng tăng target thêm 0.205 nhưng vẫn nhận verdict FAILED vì regression giảm 0.147. Vì vậy, model phải được đánh giá cả trên nhiệm vụ đích, năng lực ngoài miền, định dạng đầu ra và latency.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI assistant để đọc cấu trúc repo, giải thích mục tiêu của NB1–NB5, hỗ trợ thiết lập môi trường, phân tích log kiểm thử và đối chiếu báo cáo với các artefact. AI giúp tôi xác định một số vấn đề liên quan đến môi trường Python và cách chạy gatekeeper. Tuy nhiên, đôi lúc AI đưa ra đề xuất dựa trên giả định chưa đúng về môi trường đang chạy, chẳng hạn nhầm giữa Jupyter local và Colab hoặc đề nghị tạo lại môi trường khi nó vẫn đang được sử dụng. Vì vậy, tôi nhận ra rằng mọi đề xuất của AI đều cần được kiểm chứng bằng log, đường dẫn Python và kết quả chạy thực tế.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Tôi sẽ bắt đầu bằng việc thống nhất với khách hàng về tiêu chí thành công và đóng băng bộ đánh giá trước khi huấn luyện. Bộ đánh giá phải bao gồm nhiệm vụ đích, các ca ngoài miền để phát hiện regression, yêu cầu định dạng, latency và ngưỡng chấp nhận cho từng nhóm. Sau đó tôi sẽ đo base model với cả naive prompt và optimized prompt để xác định fine-tuning có thực sự cần thiết hay không. Chỉ sau khi có baseline đáng tin cậy, tôi mới xây dựng dữ liệu train và chuẩn bị thêm replay data để hạn chế catastrophic forgetting.
