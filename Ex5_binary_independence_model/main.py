# BÀI TẬP CHUYÊN ĐỀ 5
# Sinh viên thực hiện:
# Nguyễn Viết Tín - N19DCCN171
# Huỳnh Tuấn Kiệt - N19DCCN079
# Lê Lâm Tuấn - N19DCCN177

import math
import re
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Tải stop words
stop_words = set(stopwords.words('english'))

# Loại bỏ stop words
def remove_stopwords(documents):
    filtered_documents = []
    for doc in documents:
        words = nltk.word_tokenize(doc)
        filtered_words = [
            word for word in words if word.lower() not in stop_words]
        filtered_documents.append(' '.join(filtered_words))
    return filtered_documents

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

######### GIAI ĐOẠN 3: Binary Independence Model ############
# docs = [
#     "breakthrough for schizophrenia",
#     "new schizophrenia drug",
#     "new approach for treatment of schizophrenia",
#     "new hopes for schizophrenia patients"
# ]

# query = "schizophrenia approach"

def create_inverted_index(docs):
    '''
    Khởi tạo Inverted Index từ các docs
    Inverted Index sẽ có dạng {'từ khóa': [tài liệu 1, tài liệu 2, ...]}
    '''
    inverted_index = {}
    for i in range(0, len(docs), 1):
        words = docs[i].lower().split()
        for word in words:
            if word not in inverted_index:
                inverted_index[word] = []
            if (i+1) not in inverted_index[word]:
                inverted_index[word].append(i+1)
    return inverted_index

def calculate_rsv(docs, weights):
    '''
    Tính RSV  
    '''
    # Tính RSV cho từng tài liệu
    RSV = []  # List này chứa các RSV của các document
    for doc_index in range(0, len(docs)):
        sub_rsv = 0
        for weight_item in weights:
            term = list(weight_item.keys())[0]  # lấy ra key là term
            if term in docs[doc_index]:
                sub_rsv += weight_item[term]
        RSV.append(sub_rsv)
    return RSV

def preweight(query, inverted_index, docs):
    '''
    Tính RSV ban đầu bằng công thức idf
    '''
    query_terms = query.split()
    query_terms_present = [] #Các từ trong query và có trong tập tài liệu
    for term in query_terms:
        if term in inverted_index:
            query_terms_present.append(term)
    query_terms_present.sort()

    weights = [] #List các weight
    N = len(docs)
    for term in query_terms_present:
        term_df = len(inverted_index[term])
        c_t = math.log10((N-term_df+0.5) / (term_df+0.5));
        weights.append({term: c_t})
    
    #Tính RSV cho từng tài liệu
    RSV = calculate_rsv(docs, weights)
    
    return RSV, weights

def compare_two_list(list1, list2):
    '''
    So sánh 2 list
    '''
    if len(list1) != len(list2):
        return False
    else:
        for i in range(len(list1)):
            if list1[i] != list2[i]:
                return False
        return True

def get_rel_docs(RSV, n_relvelant=1):
    '''
    Lấy các tài liệu revelant trả về document id
    '''
    sorted_indexes = sorted(
        enumerate(RSV), key=lambda x: x[1], reverse=True)
    
    top_n_indexes = [index+1 for index, _ in sorted_indexes[:n_relvelant]]
    return top_n_indexes

def recompute_weights(RSV, docs, inverted_index, weights, n_relvelant=1):
    '''
    Tính lại RSV dựa trên n tài liệu relvelant
    pi = (si + 0.5) / (S + 1)
    ri = (ni - si + 0.5) / (N - S + 1)
    '''
    new_weights = []
    S = n_relvelant
    N = len(docs)
    rel_docs = get_rel_docs(RSV, n_relvelant)
    for weight_item in weights:
        si = 0
        term = list(weight_item.keys())[0]  # lấy ra key là term
        posting = inverted_index[term]

        for rel_id in rel_docs:
            if(rel_id in posting):
                si += 1
        pi = (si+0.5) / (S+1)
        ni = len(posting) #df
        ri = (ni - si + 0.5) / (N-S+1)

        c_t = math.log10((pi * (1-ri)) / (ri * (1-pi)))
        new_weights.append({term: c_t})
    new_RSV = calculate_rsv(docs, new_weights)
    return new_RSV


n_rels = 2
for query_index in range(0, len(queries)):
    query = queries[query_index]
    inverted_index = create_inverted_index(docs)
    RSV, weights = preweight(query, inverted_index, docs)
    new_RSV = recompute_weights(RSV, docs, inverted_index, weights, n_rels)
    rel_docs = get_rel_docs(RSV, n_rels)
    new_rel_docs = get_rel_docs(new_RSV, n_rels)

    while not compare_two_list(rel_docs, new_rel_docs):
        RSV = new_RSV
        new_RSV = recompute_weights(RSV, docs, inverted_index, weights, n_rels)
        rel_docs = get_rel_docs(RSV, n_rels)
        new_rel_docs = get_rel_docs(new_RSV, n_rels)

    print("Query", query_index+1, ": ", new_rel_docs)

