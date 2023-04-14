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

# Tìm 1 item trong list 2 chiều, trả về số cột


def find_indices_of_value(lst, value):
    indices = []
    for i in range(len(lst)):
        if lst[i] == value:
            indices.append(i+1)  # index + 1 là STT của tài liệu
    return indices

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
#     "the quick brown fox jumps over the lazy dog",
#     "the lazy dog is sleeping",
#     "the quick fox is quick brown",
#     "the lazy dog jumps over the fox"
# ]

# query = "quick brown fox"

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


def get_df(inverted_index, term):
    return len(inverted_index[term])


def intersection_d_q(doc, query):
    return list(set(doc.split()) & set(query.split()))


def preweight(inverted_index):
    '''
    Tính toán trọng số BIM ban đầu dựa trên công thức:
    c(t) = log((N-df + 0.5)/(df+0.5))
    '''
    N = len(inverted_index)
    weight_i_index = []
    for term in inverted_index:
        term_df = get_df(inverted_index, term)
        c_t = math.log((N - term_df + 0.5) / (term_df + 0.5))
        weight_i_index.append({term: inverted_index[term], "c": c_t})

    return weight_i_index


def find_term(weighten_inverted_index, term):
    '''
    Tìm từ trong inverted index, trả về index của term đó,
    trả về -1 nếu không tìm được
    '''
    for index in range(len(weighten_inverted_index)):
        item_term = list(weighten_inverted_index[index].keys())[0]
        if (item_term == term):
            return index
    return -1


def query_BIM(weighten_inverted_index, query, docs):
    '''
    Tính toán RSV là tổng các weight term trong querry
    '''
    docs_computed_RSV = []
    for doc_index in range(len(docs)):
        positive_terms = intersection_d_q(docs[doc_index], query)
        if (len(positive_terms) == 0):
            docs_computed_RSV.append({"doc_id": doc_index+1, "rsv": 0})
        else:
            rsv = 0.0
            for term in positive_terms:
                term_index = find_term(weighten_inverted_index, term)
                rsv += weighten_inverted_index[term_index]['c']
            docs_computed_RSV.append({"doc_id": doc_index+1, "rsv": rsv})
    return docs_computed_RSV


def sort_by_RSV(evaluated_list):
    return sorted(evaluated_list, key=lambda x: -x['rsv'])


def revelant_feedback_weight(revelants, query, weighten_inverted_index, inverted_index, docs):
    '''
    Tính lại weight dựa trên feedback
    input là tập tài liệu liên quan, query, inverted_index đã tính weight
    '''
    revelant_docs = []
    # List các tài liệu liên quan
    for revelant_id in revelants:
        revelant_docs.append(docs[revelant_id-1])

    N = len(weighten_inverted_index)
    N_ref = len(revelants)

    for term in query.split():
        v_i = 0.0
        for doc in revelant_docs:
            if term in doc:
                v_i += 1
        pi = (v_i + 0.5) / (N_ref + 1)
        ui = (get_df(inverted_index, term) - v_i + 0.5) / (N - N_ref + 1)
        weight = math.log((1-ui)/ui) + math.log(pi/(1-pi))

        idx = find_term(weighten_inverted_index, term)
        weighten_inverted_index[idx]['c'] = weight


inverted_index = create_inverted_index(docs)
weighten_inverted_index = preweight(inverted_index)

results = []
for query_idx in range(len(queries)-1):
    query = queries[query_idx]
    evaluated_list = query_BIM(weighten_inverted_index, query, docs)
    sorted_evaluated_list = sort_by_RSV(evaluated_list)
    results.append({"query_id": query_idx + 1, "doc_list": sorted_evaluated_list[:5]})

print(results)
