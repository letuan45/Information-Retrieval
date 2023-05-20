# BÀI TẬP CHUYÊN ĐỀ 2: Không gian Vector
# Sinh viên thực hiện:
# Nguyễn Viết Tín - N19DCCN171
# Huỳnh Tuấn Kiệt - N19DCCN079
# Lê Lâm Tuấn - N19DCCN177

import re
import numpy as np
import nltk
from nltk.corpus import stopwords
import os
import re
import math
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from scipy.sparse import dok_matrix
from scipy.spatial.distance import cosine

nltk.download('punkt')
nltk.download('stopwords')

# Tải stop words
stop_words = set(stopwords.words('english'))


# Loại bỏ stop words
def remove_stopwords(docs):
    filtered_docs = []
    for doc in docs:
        words = nltk.word_tokenize(doc)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        filtered_docs.append(' '.join(filtered_words))
    return filtered_docs


######### GIAI ĐOẠN 1: CHUẨN HÓA INPUT #########
# Mục tiêu: Xóa bỏ kí tự khoảng trắng thừa, kí tự đặc biệt, số, 
# Biến đổi lowercase
def read_file_and_modify(filename):
    with open('./npl/' + filename, 'r') as f:
        text = f.read()
    clean_text = ""
    for char in text:
        if char not in "\n":
            clean_text += char
    text = text.strip()  # loại bỏ khoảng trắng thừa ở đầu và cuối văn bản
    words = text.split()  # tách văn bản thành các từ và loại bỏ khoảng trắng thừa giữa 
    # các từ
    words = " ".join(words)
    clean_text = re.sub(r'\d+', '', words)  # loại bỏ số

    # loại bỏ khoảng trắng thừa ở đầu và cuối câu
    # Tách từng văn bản thành list
    clean_text = clean_text.split("/")
    for i in range(len(clean_text)):
        clean_text[i] = clean_text[i].strip().lower()
    return clean_text

######### GIAI ĐOẠN 2: CHUẨN HÓA DỮ LIỆU #########
docs = read_file_and_modify("doc-text")
queries = read_file_and_modify("query-text")
# phần tử cuối là rỗng, nên bỏ
queries = queries[:-1]
# Filter stop words
docs = remove_stopwords(docs)
queries = remove_stopwords(queries)

######### GIAI ĐOẠN 3: TF-IDF, MA TRẬN VÀ TRUY VẤN #########
# Tính TF của mỗi từ trong từng tài liệu
tf = []
for i, doc in enumerate(docs):
    tf.append({})
    for word in doc.split():
        if word in tf[i]:
            tf[i][word] += 1
        else:
            tf[i][word] = 1

# Tính IDF cho mỗi từ
idf = {}
for doc in docs:
    for word in set(doc.split()):
        if word in idf:
            idf[word] += 1
        else:
            idf[word] = 1
for word in idf:
    idf[word] = 1 + (math.log(len(docs) / idf[word]))

# Tính TF-IDF
tfidf = []
for i in range(len(docs)):
    tfidf.append({})
    for word in tf[i]:
        tfidf[i][word] = tf[i][word] * idf[word]

for query_idx, query in enumerate(queries):    
    # Tính TF-IDF cho câu query
    query_tfidf = {}
    for word in query.split():
        if word in query_tfidf:
            query_tfidf[word] += 1
        else:
            query_tfidf[word] = 1
    for word in query_tfidf:
        if word in idf:
            query_tfidf[word] *= idf[word]
        else:
            query_tfidf[word] = 0

    # Tính độ tương đồng cosine giữa câu query và từng tài liệu trong tập tài liệu
    similarity = []
    for i in range(len(docs)):
        dot_product = 0
        query_norm = 0
        doc_norm = 0
        for word in set(query.split()).union(set(docs[i].split())):
            dot_product += query_tfidf.get(word, 0) * tfidf[i].get(word, 0)
            query_norm += query_tfidf.get(word, 0) ** 2
            doc_norm += tfidf[i].get(word, 0) ** 2
        if query_norm == 0 or doc_norm == 0:
            similarity.append(0)
        else:
            similarity.append(dot_product / (math.sqrt(query_norm) * math.sqrt(doc_norm)))

    #In ra văn bản có độ tương đồng cao nhất với câu truy vấn
    print(f"Query: {query_idx + 1}")
    print(f"Answer: {similarity.index(max(similarity)) + 1}")
    print(f"Similarities: {max(similarity)}")
    print("=" * 50)

# LƯU TRỮ TF-IDF BẰNG CÁC HÀM BUILD SẴN
# Khởi tạo TfidfVectorizer và tính toán ma trận TF-IDF cho các văn bản trong tệp
print("---CHẠY VỚI CÁC HÀM BUILD SẴN TỪ THƯ VIỆN sklearn---")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs)

for query_idx, query in enumerate(queries):
    # Chuyển đổi câu truy vấn thành vector TF-IDF
    query_vector = vectorizer.transform([query])

    # Tính toán độ tương đồng giữa câu truy vấn và các văn bản trong tập dữ liệu
    cos_similarities = cosine_similarity(query_vector, tfidf_matrix)

    # Lấy chỉ mục của các văn bản có độ tương đồng cao nhất với câu truy vấn
    most_similar_index = cos_similarities.argmax()

    # In ra văn bản có độ tương đồng cao nhất với câu truy vấn
    print(f"Query: {query_idx+1}")
    print(f"Answer: {most_similar_index+1}")
    print(f"Similarities: {cos_similarities.max()}")
    print("=" * 50)