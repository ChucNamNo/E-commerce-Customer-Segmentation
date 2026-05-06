# 🛒 E-commerce Customer Segmentation (Phân Nhóm Hành Vi Khách Hàng)

[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/datasets/umuttuygurr/e-commerce-customer-behavior-and-sales-analysis-tr)
*Ngày thực hiện: 12/03/2026 | Ngôn ngữ & Công cụ: Python, Machine Learning, K-Means Clustering, DBSCAN, PCA*

---

## 📌 1. Tổng quan dự án (Introduction)

**Mục tiêu:** 
Đánh giá hành vi mua sắm của khách hàng thương mại điện tử nhằm tối ưu hóa Giá trị vòng đời khách hàng (CLV) và chuyển dịch chiến lược từ tiếp thị đại trà sang cá nhân hóa trải nghiệm.

**Phạm vi nghiên cứu:**
* Phân tích nhân khẩu học khách hàng, chi tiết giao dịch và các chỉ số hành vi tương tác.
* Áp dụng và so sánh hai thuật toán phân cụm **K-means** và **DBSCAN** để nhận diện và phân khúc chân dung khách hàng.

---

## ⚙️ 2. Phương pháp thực hiện (Methodology)

### 🧹 2.1 Chuẩn bị & Tiền xử lý dữ liệu (Data Preparation)
*   **Làm sạch dữ liệu (24.049 bản ghi):** Xử lý giá trị khuyết thiếu bằng cách gán giá trị Trung vị (Median) cho các biến định lượng (Tuổi, Thời gian giao hàng, Điểm đánh giá) và Yếu vị (Mode) cho các biến định danh.
*   **Loại bỏ dữ liệu nhiễu:** Xóa 1.935 bản ghi trùng lặp để ngăn ngừa hiện tượng thiên lệch (bias).
*   **Xử lý ngoại lai (Outliers):** Áp dụng kỹ thuật Winsorization cho các biến biến động cao và biến đổi Logarit cho cột `Total_Amount` nhằm bảo toàn dữ liệu khách hàng VIP.
*   **Chuẩn hóa & Mã hóa:** Dùng Min-Max Scaling để chuẩn hóa dữ liệu và One-Hot Encoding cho các biến định danh.
*   **Giảm chiều dữ liệu (PCA):** Rút gọn từ 37 đặc trưng cốt lõi xuống còn 22 thành phần (bảo toàn 90% lượng thông tin).

### 🔬 2.2 Quy trình phân tích (Analysis Workflow)
1.  **EDA (Phân tích khám phá):** Tìm ra quy luật phân phối, điểm bất thường và mối tương quan giữa các biến số hành vi và nhân khẩu học.
2.  **Huấn luyện mô hình:** So sánh hiệu suất của hai mô hình học máy không giám sát: K-means (dựa trên khoảng cách) và DBSCAN (dựa trên mật độ).
3.  **Đánh giá mô hình:** Dựa trên hệ số Silhouette, thuật toán **K-means với số lượng cụm tối ưu k=4** được chọn làm mô hình tốt nhất.



![Biểu đồ phân phối theo độ tuổi của khách hàng](https://longchucute.id.vn/_next/image?url=https%3A%2F%2Fwww.notion.so%2Fimage%2Fattachment%253Aa1cbf585-cc68-493c-aba2-36e8544d4600%253Aimage.png%3Ftable%3Dblock%26id%3D357da7f3-5722-80b6-9508-fe5e71f955af%26cache%3Dv2&w=3840&q=75)
![Biểu đồ thể hiện 10 thành phố hàng đầu theo khách hàng](https://longchucute.id.vn/_next/image?url=https%3A%2F%2Fwww.notion.so%2Fimage%2Fattachment%253A009d0a2b-1c7c-41dc-bdc1-e9fc39845d4f%253Aimage.png%3Ftable%3Dblock%26id%3D357da7f3-5722-802a-8ae0-e29c71ac3925%26cache%3Dv2&w=2048&q=75)
![Biểu đồ phân phối theo giá trị đơn hàng](https://longchucute.id.vn/_next/image?url=https%3A%2F%2Fwww.notion.so%2Fimage%2Fattachment%253A1002c126-db6a-4412-8f68-c58227d5c324%253Aimage.png%3Ftable%3Dblock%26id%3D357da7f3-5722-80b4-b28d-d615c6a9c594%26cache%3Dv2&w=3840&q=75)
![Biểu đồ thể hiện tổng doanh thu theo danh mục sản phẩm](https://longchucute.id.vn/_next/image?url=https%3A%2F%2Fwww.notion.so%2Fimage%2Fattachment%253A4392457c-302d-490f-9179-387e1dbf2453%253Aimage.png%3Ftable%3Dblock%26id%3D357da7f3-5722-80b8-ac96-d3c922a37e2d%26cache%3Dv2&w=1920&q=75)
![Biểu đồ phân phối thời gian cho trang web](https://longchucute.id.vn/_next/image?url=https%3A%2F%2Fwww.notion.so%2Fimage%2Fattachment%253Ad3fabb31-e4b0-404d-a578-ab96fcf1c57a%253Aimage.png%3Ftable%3Dblock%26id%3D357da7f3-5722-8022-bc5c-f5a4cd529a02%26cache%3Dv2&w=2048&q=75)
![Biểu đồ phương sai tích lũy](https://longchucute.id.vn/_next/image?url=https%3A%2F%2Fwww.notion.so%2Fimage%2Fattachment%253Afee7c3e6-5d0f-4570-bb36-9b3248941804%253Aimage.png%3Ftable%3Dblock%26id%3D357da7f3-5722-8051-9428-d56f76b58cd1%26cache%3Dv2&w=3840&q=75)

---

## 📊 3. Kết quả & Insights (Findings & Insights)

### 📈 Xu hướng Nhân khẩu học & Doanh thu
*   **Giới tính & Vị trí:** Khách hàng nữ chiếm tỷ trọng cao hơn. Thành phố Istanbul áp đảo hoàn toàn về mức độ tập trung khách hàng.
*   **Doanh thu:** Danh mục **Electronics (Điện tử)** mang lại tổng doanh thu cao nhất, vượt xa các ngành hàng khác.
*   **Tương quan hành vi:** Không có mối tương quan rõ rệt giữa thời gian trên trang web (Session Duration) và mức chi tiêu. Tuy nhiên, Đơn giá sản phẩm (Unit Price) có tương quan dương rất mạnh với tổng giá trị đơn hàng.

![Biểu đồ nhiệt tương quan giữa các biến số](https://longchucute.id.vn/_next/image?url=https%3A%2F%2Fwww.notion.so%2Fimage%2Fattachment%253Aca04cc2e-9fbf-4fd6-aab1-66b4e5049b3f%253Aimage.png%3Ftable%3Dblock%26id%3D357da7f3-5722-80e1-816d-c0ba6603955a%26cache%3Dv2&w=3840&q=75)

### 👥 Phân cụm khách hàng (K-means Clustering)

Dữ liệu được phân thành 4 nhóm khách hàng riêng biệt:

| Cụm (Cluster) | Chân dung khách hàng | Đặc điểm hành vi | Tỷ trọng |
| :--- | :--- | :--- | :--- |
| **Cluster 0** | 🧑 Nam giới thông thường | Mua sắm dàn trải, số lượng mặt hàng ít, duy trì tương tác ổn định nhưng chỉ mua khi có nhu cầu cấp thiết. | 21.46% |
| **Cluster 1** | 🤵 Nam giới VIP / Tiềm năng | Quan tâm mạnh đến đồ công nghệ, thể thao, sách. Có các giao dịch đột biến với giá trị cực lớn (~37.852). | 27.37% |
| **Cluster 2** | 👩 Phụ nữ gia đình | Hành vi mua sắm đa dạng (Nhà cửa, Làm đẹp, Thực phẩm...). Tần suất ổn định, đóng vai trò quyết định chi tiêu sinh hoạt. | **28.72%** |
| **Cluster 3** | 💃 Nữ giới thời trang & Trải nghiệm| Tập trung vào thời trang và làm đẹp. Mua sắm mang tính cảm xúc, ngẫu hứng cao, dễ bị ảnh hưởng bởi xu hướng MXH. | 22.46% |

![Biểu đồ phân bố Danh mục sản phẩm theo cụm](https://longchucute.id.vn/_next/image?url=https%3A%2F%2Fwww.notion.so%2Fimage%2Fattachment%253A2d306ab1-fee7-4c14-9a94-47754ca745ad%253Aimage.png%3Ftable%3Dblock%26id%3D357da7f3-5722-803a-bd35-de68fc04fc2e%26cache%3Dv2&w=3840&q=75)

---

## 💡 4. Đề xuất chiến lược (Recommendations)

Các đề xuất được tinh chỉnh riêng biệt nhằm tối ưu hóa hoạt động của từng phòng ban:

### 🎯 Đội ngũ Marketing & Sales
*   **Khách hàng Nữ trải nghiệm (Cluster 3):** Đẩy mạnh Livestream, Influencer Marketing và Flash Sale để kích thích nhu cầu mua sắm thị giác và cảm xúc.
*   **Khách hàng Nam thông thường (Cluster 0):** Cung cấp mã giảm giá, miễn phí vận chuyển và chương trình tích điểm đổi quà để gia tăng sự trung thành và giảm tỷ lệ rời bỏ.

### 📦 Đội ngũ Product
*   **Phụ nữ gia đình (Cluster 2):** Xây dựng các gói "Combo gia đình" (VD: Mua sắm thực phẩm thiết yếu tặng kèm đồ gia dụng).
*   **Nam VIP (Cluster 1):** Giới thiệu sớm các dòng sản phẩm công nghệ mới ra mắt hoặc công cụ thể thao chuyên nghiệp (Exclusive early access).

### 💼 Đội ngũ Business Development
*   **Tối ưu Vận hành (Cluster 2):** Đơn giản hóa quy trình thanh toán, đẩy nhanh tốc độ giao hàng, thiết lập thông báo tự động nhắc mua lại với nhóm hàng tiêu dùng nhanh.
*   **Chăm sóc VIP (Cluster 1):** Thiết lập kênh chăm sóc khách hàng ưu tiên (Priority line), tặng các đặc quyền phi tài chính (vé mời sự kiện trải nghiệm).

### 💻 Đội ngũ Data Analytics & IT
*   **Mở rộng nghiên cứu:** Thực nghiệm thêm các thuật toán phân cụm khác (như Spectral Clustering) để liên tục tối ưu độ chính xác.
*   **Phân tích sâu (Deep dive):** Đánh giá mức độ hiệu quả của mã giảm giá và tính toán vòng đời mua sắm (Time between purchases) để thiết kế các chiến dịch retargeting chuẩn xác hơn.
