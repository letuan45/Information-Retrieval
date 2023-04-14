# BÀI TẬP CHUYÊN ĐỀ 3: NÉN BỘ TỪ ĐIỂN
# Sinh viên thực hiện:
# Nguyễn Viết Tín - N19DCCN171
# Huỳnh Tuấn Kiệt - N19DCCN079
# Lê Lâm Tuấn - N19DCCN177

import re
import numpy as np
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
import sys
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
# Filter stop words
docs = remove_stopwords(docs)

def save_to_file(file_name, item):
    with open(file_name, 'wb') as f:
        pickle.dump(item, f)

######## GIAI ĐOẠN 3: XÂY DỰNG INVERTED INDEX #########
# initialize inverted index dictionary
inverted_index = {}

# Iterate over each document and tokenize the words
for i, doc in enumerate(docs):
    words = doc.lower().split()
    unique_words = set(words)
    # Iterate over each unique word in the document and add the document index to the posting list
    for word in unique_words:
        if word not in inverted_index:
            # If the word is not in the inverted index, add it with an empty posting list and a document frequency of 0
            inverted_index[word] = {'df': 0, 'posting': []}
        inverted_index[word]['df'] += 1
        inverted_index[word]['posting'].append(i)

######## GIAI ĐOẠN 4: NÉN BỘ TỪ ĐIỂN: FIXED-WIDTH ENTRIES #########
# Định dạng từ điển: "term": {df: number, postings: [list]}

#Tính toán dung lượng
print("Mảng phần tử kích thước tĩnh")
total_size = 0
for key, value in inverted_index.items(): 
    term_size = sys.getsizeof(key)
    df_size = sys.getsizeof(value['df'])
    posting_pointer_size = sys.getsizeof(id(value['posting']))
    total_size = total_size + (term_size + df_size + posting_pointer_size)

print("+ Kích thước: ",total_size / 1048576, "MB")
save_to_file('compress1.pickle', inverted_index)

######## GIAI ĐOẠN 5: NÉN BỘ TỪ ĐIỂN: LONG STRING #########
print("Chuỗi ký tự dài")
# Sort theo alphabetic
sorted_inverted_index = {term: inverted_index[term] for term in sorted(inverted_index)}
sorted_terms_string = ''.join(sorted(inverted_index.keys()))

# Định nghĩa term pointer
term_pointers = []
position = 0
for term in sorted(inverted_index.keys()):
    term_pointers.append(position)
    position += len(term)

# Chuyển thành list, bỏ key và chỉ nhận value
sorted_inverted_index = [{key: value} for key, value in sorted_inverted_index.items()]

# Thay đổi bộ từ điển, thay đổi key là vị trí của kí tự bắt đầu
for i in range(len(sorted_inverted_index)):
    key = term_pointers[i]
    sorted_inverted_index[i] = {key: sorted_inverted_index[i][list(sorted_inverted_index[i].keys())[0]]}

save_to_file('compress2_string.pickle', sorted_terms_string)
save_to_file('compress2_dict.pickle', sorted_inverted_index)
last_item_term_pointer = term_pointers[len(term_pointers)-1]

# Giải mã
# for i in range(len(sorted_inverted_index)):
#     item = sorted_inverted_index[i]
#     position = list(item.keys())[0]
#     next_position = len(sorted_terms_string)
#     if(position != last_item_term_pointer):
#         next_position = term_pointers[i+1]
#     key = sorted_terms_string[position:next_position]
#     sorted_inverted_index[i] = {key: sorted_inverted_index[i][list(sorted_inverted_index[i].keys())[0]]}

######## GIAI ĐOẠN 6: NÉN BỘ TỪ ĐIỂN: BLOCKED STORAGE #########
k=4 #Cài đặt k ban đầu
print("Blocked storage")
# Sort theo alphabetic
sorted_inverted_index = {term: inverted_index[term] for term in sorted(inverted_index)}

# Tạo long string, list các vị trí
long_string = ''
for term in sorted(inverted_index.keys()):
    term_length = str(len(term))
    append_string = (term_length + term)
    long_string += append_string

# Tạo term_pointer song hành là vị trí
terms = sorted(inverted_index.keys())
term_pointers = []
post_in = []
position = 0
count = 0
for term in terms:
    if(count == 0 or count % k == 0):
        term_pointers.append(position)
        post_in.append(count)
    position += len(term)+1
    count += 1

# Vì inverted index ban đầu có key là term nên khi loại bỏ sẽ không cùng định dạng
# Vì vậy cần thay đổi inverted index
inverted_index_t = []
for (key, value) in sorted_inverted_index.items():
    inverted_index_t.append(value)

for i in range(len(inverted_index_t)):
    if (i in post_in):
        ptr_value = term_pointers[post_in.index(i)]
        inverted_index_t[i]['term_ptr'] = ptr_value

save_to_file('compress3_string.pickle', long_string)
save_to_file('compress3_dict.pickle', inverted_index_t)


######## GIAI ĐOẠN 6: NÉN BỘ TỪ ĐIỂN: BLOCKING & FRONT CODING #########
# Định nghĩa K
k = 4
sorted_inverted_index = {term: inverted_index[term] for term in sorted(inverted_index)}

terms = list(sorted_inverted_index.keys())

def every(lambdaFn, list):
  for item in list:
    if not lambdaFn(item):
      return False
  return True
  
#   return general_part
def is_same_general_part(general_part):
    def is_same(str):
      general_len = len(general_part)
      str_len = len(str)
      i = 0
      j = 0
      while i < general_len and j < str_len:
          if general_part[i] != str[j]:
              return False
          i += 1
          j += 1
      return True
    return is_same

def get_common_prefix(list_term):
  first_term = list_term[0]
  remain_terms = list_term[1:]
  iter = 0
  general_part = first_term
  while general_part:
    is_same = is_same_general_part(general_part)
    if every(is_same, remain_terms):
      return general_part

    iter += 1
    general_part = first_term[:-iter]
  
  return general_part

def get_minus_string(str1, str2):
    freq = {}
    for char in str2:
        if char in freq:
            freq[char] += 1
        else:
            freq[char] = 1
    result = ''
    for char in str1:
        if char in freq and freq[char] > 0:
            freq[char] -= 1
        else:
            result += char
    if len(result) == 0:
        return ""
    else:
        return result
# Khởi tạo term pointer
term_pointers = [0]
long_string_front_coding = ""
position = 0
for i in range(0, len(terms), k):
    end =  k
    if(i+k > len(terms) - 1): end = len(terms) - 1
    local_tems_list = terms[i:i+end]
    long_term_front_coding = ''
    common_prefix = get_common_prefix(local_tems_list)
    rest_string = ""
    if(common_prefix != ''):
        len_general_part = len(common_prefix)
        diff_term_slice_list = list(map(lambda term: term[len_general_part:], local_tems_list))
        len_fist_term = str(len(local_tems_list[0]))
        long_term_front_coding += len_fist_term + common_prefix + '*'
        for term in local_tems_list:
            remain_char = get_minus_string(term, long_term_front_coding)
            if(rest_string == ""):
                rest_string += remain_char
            else:
                rest_string += str(len(remain_char)) + "<>" + remain_char
        long_term_front_coding += rest_string
        # Thêm value cho term pointer
        position += len(long_term_front_coding)
        term_pointers.append(position)
    else: term_pointers.append(None)
    long_string_front_coding += long_term_front_coding

# Vì inverted index ban đầu có key là term nên khi loại bỏ sẽ không cùng định dạng
# Vì vậy cần thay đổi inverted index
inverted_index_t = []
for (key, value) in sorted_inverted_index.items():
    inverted_index_t.append(value)

# Khởi tạo vị trí của term pointer
term_pointers_pos = 0
for i in range(0, len(inverted_index_t), 4):
    inverted_index_t[i]['term_ptr'] = term_pointers[term_pointers_pos]
    term_pointers_pos += 1

save_to_file('compress4_string.pickle', long_string_front_coding)
save_to_file('compress4_dict.pickle', inverted_index_t)