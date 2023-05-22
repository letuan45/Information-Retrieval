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
        filtered_words = [word for word in words if word.lower() not in stop_words]
        filtered_documents.append(' '.join(filtered_words))
    return filtered_documents

#Tìm 1 item trong list 2 chiều, trả về số cột
def find_indices_of_value(lst, value):
    indices = []
    for i in range(len(lst)):
        if lst[i] == value:
            indices.append(i+1) #index + 1 là STT của tài liệu
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
    #các từ
    words = " ".join(words)
    clean_text = re.sub(r'\d+', '', words) #loại bỏ số

    #loại bỏ khoảng trắng thừa ở đầu và cuối câu
    #Tách từng văn bản thành list
    clean_text = clean_text.split("/")
    for i in range(len(clean_text)):
        clean_text[i] = clean_text[i].strip().lower()
    return clean_text

######### GIAI ĐOẠN 2: CHUẨN HÓA DỮ LIỆU #########
docs = read_file_and_modify("doc-text")
queries = read_file_and_modify("query-text") 
#phần tử cuối là rỗng, nên bỏ
queries = queries[:-1]
# Filter stop words
docs = remove_stopwords(docs)
queries = remove_stopwords(queries)

######## GIAI ĐOẠN 3: MA TRẬN ĐÁNH DẤU #########
#Sắp xếp lại các từ có trong từ điển
vocabulary = set()
for doc in docs:
    words = doc.split()
    vocabulary.update(words)
vocabulary = sorted(list(vocabulary))

# Ma trận đánh dấu
inc_matrix = np.zeros((len(vocabulary), len(docs)))
for j, doc in enumerate(docs):
    words = doc.split()
    for i, word in enumerate(vocabulary):
        if word in words:
            inc_matrix[i, j] = 1

# In ma trận đánh dấu
#print(inc_matrix)

# Truy vấn
for j, query in enumerate(queries):
    res = None
    for i, word in enumerate(vocabulary):
        if word in query.split():
            #lấy phần tử ma trận đánh dấu tương ứng để AND
            if(res is None):
                res = inc_matrix[i]
            else:
                res = [a and b for a, b in zip(res, inc_matrix[i])]
    res = np.array(res)
    result = find_indices_of_value(res, 1)
    #print("Query", j+1, ": ", result)

######### GIAI ĐOẠN 4: CHỈ MỤC NGƯỢC ############
# Khởi tạo Inverted Index
inverted_index = {}
for i in range(0, len(docs), 1):
    #Tách từ thành list
    words = docs[i].lower().split()
    for word in words:
        #Nếu từ không nằm trong chỉ mục, convert sang set để tránh tình
        #trạng bị trùng phần từ sau khi add
        if word not in inverted_index:
            inverted_index[word] = set()
        inverted_index[word].add(i+1)

# Inverted Index sẽ có dạng {'từ khóa': {tài liệu 1, tài liệu 2, ...}}
print(inverted_index["measurement"])

#Giao với bước nhảy (skip pointer)
def intersection_has_skip(set1, set2, hasSkip, steps):
    res = set()
    lst1 = sorted(set1)
    lst2 = sorted(set2)
    i = 0
    j = 0
    
    while i < len(lst1) and j < len(lst2):
        if lst1[i] == lst2[j]:
            res.add(lst1[i])
            i += steps[0] if hasSkip(i) else 1
            j += steps[1] if hasSkip(j) else 1
        elif lst1[i] < lst2[j]:
            i += steps[0] if hasSkip(i) else 1
        else:
            j += steps[1] if hasSkip(j) else 1
            
    return res

# Truy vấn
# Truy vấn Inverted Index với bước nhảy step
hasSkip = lambda i: i % 2 == 0 #lamda function xác định skip với giá trị chẵn

def hasSkipWithThirdIndex(i):
    return (i+1) % 3 == 0

def hasSkipCustom(i): #skip qua index chẵn và chia hết cho 3,           
    return i % 2 == i % 3 #tương tự với không chi hết

steps = (2, 1) #Nhảy với step = 2 với list đầu và step = 1 với list thứ hai
for query_index in range(0, len(queries)):
    keywords = queries[query_index].split()
    result = None
    for keyword in keywords:
        #Nếu từ khóa trong query không nằm trong chỉ mục, chuyển hóa về set
        #và dừng vòng lặp
        if keyword not in inverted_index:
            result = set()
            break
        # Gán giá trị đầu cho kết quả sau khi tìm ta từ khóa
        if result is None:
            result = inverted_index[keyword]
        # Lấy giao giữa kết quả hiện tại và chỉ mục ngược đang xét
        else:
            # Bỏ comment dòng sau nếu chạy với bước nhảy
            #result = intersection_has_skip(result, inverted_index[keyword], hasSkip, steps)
            result = result.intersection(inverted_index[keyword])
    
    #print(query_index+1, result);


######### GIAI ĐOẠN 5: CHỈ MỤC NGƯỢC TỐI ƯU############
for query_index in range(0, len(queries)):
    query = queries[query_index]
    terms = query.split()
    # Lấy danh sách thẻ định vị tương ứng với query
    posting_lists = [inverted_index[term] for term in terms if term in inverted_index]

    # Sort danh sách với tần suất xuất hiện df(t)
    posting_lists.sort(key=len)

    # Gán kết quả cho thẻ định vị đầu tiên (thẻ có ít chỉ mục nhất)
    results = posting_lists[0]

    # Giao với phần còn lại
    for postings_list in posting_lists[1:]:
        new_results = set()
        for doc_id in results:
            if doc_id in postings_list:
                new_results.add(doc_id)
        results = new_results
    #print(query_index+1,results);