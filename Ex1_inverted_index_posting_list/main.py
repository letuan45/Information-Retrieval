# BÀI TẬP CHUYÊN ĐỀ 1
# Sinh viên thực hiện:
# Nguyễn Viết Tín - N19DCCN171
# Huỳnh Tuấn Kiệt - N19DCCN079
# Lê Lâm Tuấn - N19DCCN177

import re
import numpy as np
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

def find_indices_of_value(lst, value):
    '''
    Tìm 1 item trong list 2 chiều, trả về số cột
    '''
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

######## GIAI ĐOẠN 3: MA TRẬN ĐÁNH DẤU #########
def get_sorted_vocabulary(docs):
    '''
    Sắp xếp lại các từ có trong từ điển
    '''
    vocabulary = set()
    for doc in docs:
        words = doc.split()
        vocabulary.update(words)
    return sorted(list(vocabulary))


def buld_incidence_matrix(sorted_vocabulary, docs):
    '''
    Xây dựng ma trận đánh dấu
    '''
    inc_matrix = np.zeros((len(sorted_vocabulary), len(docs)))
    for j, doc in enumerate(docs):
        words = doc.split()
        for i, word in enumerate(sorted_vocabulary):
            if word in words:
                inc_matrix[i, j] = 1
    return inc_matrix

def query_with_inc_matrix(queries, sorted_vocabulary, inc_matrix):
    '''
    Truy vấn với ma trận đánh dấu
    '''
    for j, query in enumerate(queries):
        res = None
        for i, word in enumerate(sorted_vocabulary):
            if word in query.split():
                # lấy phần tử ma trận đánh dấu tương ứng để AND
                if (res is None):
                    res = inc_matrix[i]
                else:
                    res = [a and b for a, b in zip(res, inc_matrix[i])]
        res = np.array(res)
        result = find_indices_of_value(res, 1)
        print("Query", j+1, ": ", result)

# sorted_vocabulary = get_sorted_vocabulary(docs)
# inc_matrix = buld_incidence_matrix(sorted_vocabulary, docs)
# query_with_inc_matrix(queries, sorted_vocabulary, inc_matrix)

######### GIAI ĐOẠN 4: CHỈ MỤC NGƯỢC ############
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

def intersection(posting_list_1, posting_list_2):
    '''
    Giao 2 posting list (list)
    '''
    result = []
    p1, p2 = 0,0
    posting_size_1 = len(posting_list_1)
    posting_size_2 = len(posting_list_2)
    while(p1 < posting_size_1 and p2 < posting_size_2):
        if posting_list_1[p1] == posting_list_2[p2]:
            result.append(posting_list_1[p1])
            p1 += 1
            p2 += 1
        elif posting_list_1[p1] < posting_list_2[p2]:
            p1 += 1
        else:
            p2 += 1
    return result

def has_skip(skip_step, list_length):
    def checker(pointer):
        return (skip_step + pointer) < list_length
    return checker

def intersection_with_skip_pointer(posting_list_1, posting_list_2, step = 1):
    '''
    Giao 2 posting list (list) với skip pointer
    '''
    result = []
    p1, p2 = 0, 0
    posting_size_1 = len(posting_list_1)
    posting_size_2 = len(posting_list_2)

    has_skip_l1 = has_skip(step, posting_size_1)
    has_skip_l2 = has_skip(step, posting_size_2)

    while p1 < posting_size_1 and p2 < posting_size_2:
        if posting_list_1[p1] == posting_list_2[p2]:
            result.append(posting_list_1[p1])
            p1 += 1
            p2 += 1
        elif posting_list_1[p1] < posting_list_2[p2]:
            if has_skip_l1(p1) and posting_list_1[p1 + step] <= posting_list_2[p2]:
                while has_skip_l1(p1) and posting_list_1[p1 + step] <= posting_list_2[p2]:
                    p1 += step
            else: p1 += 1
        else: 
            if has_skip_l2(p2) and posting_list_2[p2 + step] <= posting_list_1[p1]:
                while has_skip_l2(p2) and posting_list_2[p2 + step] <= posting_list_1[p1]:
                    p2 += step
            else: p2 += 1
    return result


def query_with_inv_index(queries, inverted_index):
    '''
    Truy vấn với chỉ mục ngược
    '''
    for query_index in range(0, len(queries)):
        query = queries[query_index]
        query_terms = query.split()
        result = []
        choosen_term = ""
        #Chọn ra posting của từ và từ đầu tiên trong query
        for term in query_terms:
            if(term in inverted_index):
                result = inverted_index[term]
                choosen_term = term
                break
        query_terms.remove(choosen_term)
        for term in query_terms:
            posting_2 = []
            if term in inverted_index:
                posting_2 = inverted_index[term]
            if(len(posting_2) > 0):
                result = intersection(result, posting_2)
                #uncomment đoạn sau để chạy intersect có bước nhảy
                #result = intersection_with_skip_pointer(result, posting_2, 3)
        print(query_index+1, result)

def query_with_inv_index_optimize(queries, inverted_index):
    '''
    Truy vấn với chỉ mục ngược - tối ưu
    '''
    for query_index in range(0, len(queries)):
        query = queries[query_index]
        query_terms = query.split()
        result = []
        choosen_term = ""
        query_posting = []

        #Chọn ra inverted index tương ứng query words
        for term in query_terms:
            if (term in inverted_index):
                item = {}
                item[term] = inverted_index[term]
                query_posting.append(item)

        #Sort theo df   
        sorted_posting = sorted(
            query_posting, key=lambda x: len(list(x.values())[0]))
            
        sorted_query_terms = [list(posting_list.keys())[0]
                        for posting_list in sorted_posting]
        choosen_term = sorted_query_terms[0] #Từ đầu tiên
        result = inverted_index[choosen_term] #Posting đầu tiên
            
        rest = sorted_query_terms[1:] #Phần còn lại
        for term in rest:
            posting_2 = inverted_index[term]
            result = intersection(result, posting_2)
            # uncomment đoạn sau để chạy intersect có bước nhảy
            # result = intersection_with_skip_pointer(result, posting_2, 3)
        print(query_index+1, result)

inverted_index = create_inverted_index(docs)
sorted_vocabulary = get_sorted_vocabulary(docs)
#query_with_inv_index(queries, inverted_index)
query_with_inv_index_optimize(queries, inverted_index)