# BÀI TẬP CHUYÊN ĐỀ 3: MÃ HÓA DANH SÁCH THẺ ĐỊNH VỊ
# Sinh viên thực hiện:
# Nguyễn Viết Tín - N19DCCN171
# Huỳnh Tuấn Kiệt - N19DCCN079
# Lê Lâm Tuấn - N19DCCN177

import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import struct

import bitarray
import json
import time
import pickle

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


# Tìm 1 item trong list 2 chiều, trả về số cột
def find_indices_of_value(lst, value):
    indices = []
    for i in range(len(lst)):
        if lst[i] == value:
            indices.append(i + 1)  # index + 1 là STT của tài liệu
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

#Sắp xếp lại các từ có trong từ điển
vocabulary = set()
for doc in docs:
    words = doc.split()
    vocabulary.update(words)
vocabulary = sorted(list(vocabulary))

vocab_str = "".join(vocabulary)

# print(vocab_str)


# Ma trận đánh dấu
inc_matrix = np.zeros((len(vocabulary), len(docs)))
for j, doc in enumerate(docs):
    words = doc.split()
    for i, word in enumerate(vocabulary):
        if word in words:
            inc_matrix[i, j] = 1


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
    # print("Query", j+1, ": ", result)

######### GIAI ĐOẠN 3: CHỈ MỤC NGƯỢC ############
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
# print(inverted_index)
#

######### GIAI ĐOẠN 4: CÀI ĐẶT THUẬT TOÁN ############

#Danh sách khoảng cách
distance_list={}
for key in inverted_index:
    new_values = []
    prev = None
    for val in sorted(inverted_index[key]):
        if prev is None:
            new_values.append(val)
        else:
            new_values.append(val - prev)
        prev = val
    distance_list[key] = new_values
# print(distance_list)

# Mã hóa vb
def encode_vbyte(num):
    encoded = bytearray()
    while num >= 128:
        encoded.append((num & 127) | 128)
        num >>= 7
    encoded.append(num)
    return bytes(encoded)


#Giải mã vb
def decode_vbyte(bytes):
    """Decodes a byte sequence using variable byte code."""
    num = 0
    shift = 0
    for byte in bytes:
        num |= (byte & 0x7F) << shift
        shift += 7
        if not (byte & 0x80):
            break
    return num

#Mã hóa Gamma
def gamma_encode(num):
    # Get the binary representation of the number
    binary = bin(num)[2:]

    # Encode the length of the binary representation in unary code
    length = len(binary)
    unary_code = '0' * (length - 1) + '1'

    # Combine the unary code and binary representation
    gamma_code = unary_code + binary

    return gamma_code

#Giải mã gamma
def gamma_decode(gamma_code):
    # Find the length of the unary code
    length = gamma_code.index('1') + 1

    # Extract the binary representation
    binary = gamma_code[length:]

    # Convert binary to decimal
    num = int(binary, 2)

    return num

#Mã hóa VB danh sách thẻ định vị
VB_EC_Posting_list = {}
for key in distance_list:
    new_values = []
    for val in distance_list[key]:
        new_values.append(encode_vbyte(val))
    VB_EC_Posting_list[key] = new_values
print("\Mã hóa VB\n",VB_EC_Posting_list)

#Giải mã VB danh sách thẻ định vị
VB_DC_Posting_list = {}
for key in VB_EC_Posting_list:
    new_values = []
    for val in VB_EC_Posting_list[key]:
        new_values.append(decode_vbyte(val))
    VB_DC_Posting_list[key] = new_values
print("\nGiải mã VB\n",VB_DC_Posting_list)

#Mã hóa VB danh sách thẻ định vị
GM_EC_Posting_list = {}
for key in distance_list:
    new_values = []
    for val in distance_list[key]:
        new_values.append(gamma_encode(val))
    GM_EC_Posting_list[key] = new_values
print("\nMã hóa Gamma\n",GM_EC_Posting_list)

#Giải mã VB danh sách thẻ định vị
GM_DC_Posting_list = {}
for key in GM_EC_Posting_list:
    new_values = []
    for val in GM_EC_Posting_list[key]:
        new_values.append(gamma_decode(val))
    GM_DC_Posting_list[key] = new_values
print("\nGiải mã VB\n",GM_DC_Posting_list)