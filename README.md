﻿# Information-Retrieval
TỔNG ÔN INFORMATION RETRIEVAL
I. Vector space model
	1. Giới thiệu
- Vector space model là một mô hình đại số thế hiện thông tin văn bản như một vector, trong IR (Information Retrieval) các Query, Documents đều được vector hóa, các phần tử vector này thể hiện mức độ quan trọng của một từ và cả sự xuất hiện hay không xuất hiện của nó trong document.
 
- Trong biểu diễn không gian Euclid n chiều, mỗi chiều tương ứng với một từ trong tập hợp các từ (tổng hợp từ các document), query, document được biểu diễn là 1 điểm.
- Sự tương đồng giữa 2 văn bản: được định nghĩa là khoảng cách giữa các điểm, hoặc là góc giữa những vector trong không gian.
- Mỗi từ trong không gian trên sẽ có một trọng số tương ứng với phương pháp xếp hạng cụ thể, TF-IDF là một phương pháp phổ biến.
- TF-IDF giúp chuyển đổi thông tin dưới dạng văn bản thành một vector space model thông qua weight (trọng số).
- Điểm yếu: khi càng có nhiều term (từ) thì số lượng chiều càng lớn, vì thế trong 1 search engine có kho tài liệu lớn, số lượng chiều được chuyển đổi lên đến hàng chục triệu là điều không thể tránh khỏi.
	2. Thể hiện

 
 
 
- Biểu diễn dưới dạng vector như cách trên cũng là một dạng biểu diễn điển hình, tuy nhiên thực thế ta không thể dựa vào số lần xuất hiện của từ đó trong bộ từ điển là mấu chốt của vấn đề để xác định tầm quan trọng của nó được, có trường hợp 1 từ xuất hiện chỉ vỏn vẹn vài lần, tuy nhiên nó đóng vai trò hết sức quan trọng trong việc truy xuất. Vì thế ta cần một phương pháp đánh giá nào đó mang tính phân loại tốt hơn.
	3. Phương pháp tính trọng số tần xuất logarit
 
- Nếu từ đó không xuất hiện trong một tài liệu thì tft,d bằng 0. Vì log(0) = âm vô cùng nên ta cần +1.
- Điểm cho một cặp document-query được tính bằng tổng của các trọng số của term t trong cả document d và query q. Điểm sẽ bằng 0 nếu như query terms không xuất hiện trong document.
	4. Phương pháp tính trọng số nghịch đảo
- Tình huống đặt ra: 
+ Từ hiếm thì quan trọng hơn những từ có tần suất xuất hiện cao. Trong mỗi ngôn ngữ có những từ lặp đi lặp lại nhiều lần nhưng vô nghĩa (ví dụ trong tiếng Anh là a, the, to, of v.v), trong full-text search nó được gọi là stopwords.
+ Điểm bất cập của phương pháp term frequency trên là lúc những từ càng xuất hiện nhiều thì điểm càng cao, còn những từ hiếm thì điểm xếp hạng thấp hơn. Do đó chúng ta cần một phương pháp tốt hơn để đánh giá từ hiếm.
- Ý tưởng giải quyết bất cập: không có cách nào để tăng điểm của từ hiếm với lượng thông tin cung cấp chỉ là các văn bản ban đầu, vì thế ta sẽ tìm cách giảm trọng số của từ nào có document frequency cao bằng cách lấy tổng số tài liệu chia cho số tài liệu mà một từ xuất hiện.
 
- Lưu ý rằng, idf chỉ ảnh hưởng đến sự xếp hạng nếu query có 2 từ trở lên.
 
	5. Concept chung
- Như đã nói ở trên, câu truy vấn (query) cũng được “vector hóa”, chúng ta cần so sánh query đó với các tài liệu sẵn có. Như hình trên đã biểu thị hình thái các vector, việc chúng ta cần làm là tìm sự tương đồng của vector query của ta gần với tài liệu nào nhất rồi từ đó đưa ra xếp hạng các độ tương đồng đó.
- Như kiến thức ta đã học về toán học không gian, để so sánh khoảng cách giữa hai vector ta cần tính khoảng cách giữa chúng hoặc góc giữa chúng.
+ Đối với phương pháp tính khoảng cách: dễ thấy rằng, khoảng cách q và d có thể là rất lớn mặc dù chúng phân phối giống nhau.
 
+ Như vậy ta nhìn chung có vẻ phương pháp tính góc là khả quan hơn. Chúng ta sẽ đo góc từ 0 độ đến 360 độ ư ? không ! chúng ta sẽ dùng cosin để đánh giá.
- Dễ thấy rằng, góc giữa hai vector càng lớn thì cosin giữa chúng sẽ càng nhỏ, và điều ngược lại cũng xảy ra tương tự. Và khi góc 2 vector = 0 thì cosin sẽ bằng 1 (ý nói rằng 2 câu văn bản có thể bằng nhau hoặc chứa nhau), đây chính là tiền đề quan trọng mà ta cần tìm.
- Đồ thị hàm cosin:
 
- Lại một vấn đề nữa xảy ra: Tài liệu càng dài thì chẳng phải khả năng xuất hiện các từ trùng với câu truy vấn càng lớn hay sao? Vì thể chúng ta có một khái niệm mới gọi là “Chuẩn hóa Cosin”
	6. Chuẩn hóa Cosin
- Sự chuẩn hóa Cosin ra đời với mục đích làm giảm sự ảnh hưởng cúa các tài liệu dài so với tài liệu ngắn. Bằng cách chia từng phần tử của nó cho độ dài của nó, Dể tính độ dài của nó ta làm như sau (định mức L2 hay còn gọi là L2 norm)
 
 
 
Hay
 
=> Đây là thứ mà ta gọi là độ tường đồng Cosine (Cosine similarity), sự tương đồng Cosine chỉ đơn giản là tích vô hướng của 2 vector và chia cho tích độ dài.
