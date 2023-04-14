# BÀI TẬP CHUYÊN ĐỀ 3: NÉN BỘ TỪ ĐIỂN - ĐỌC FILE
# Sinh viên thực hiện:
# Nguyễn Viết Tín - N19DCCN171
# Huỳnh Tuấn Kiệt - N19DCCN079
# Lê Lâm Tuấn - N19DCCN177

import time
import pickle

def read_from_file(file_name):
    inverted_index = {}
    with open(file_name, 'rb') as f:
        inverted_index = pickle.load(f)
    return inverted_index

def messure_exc_time(file_name, print_string):
    start_time = time.time()  # Lấy thời điểm hiện tại
    inverted_index = read_from_file(file_name)
    end_time = time.time()  # Lấy thời điểm hiện tại
    time_taken = end_time - start_time # Thời gian sau khi chạy
    print(f"{print_string} {time_taken}s")

#ĐO THỜI GIAN ĐỌC
# messure_exc_time("compress1.pickle", "-> Thời gian đọc của PP nén mảng") #0.028s đến 0.04s
#messure_exc_time("compress2_dict.pickle", "-> Thời gian đọc của PP nén chuỗi dài") #0.019s đến 0.03s
# messure_exc_time("compress3_dict.pickle", "-> Thời gian đọc của PP nén khối") #0.016s đến 0.03s
# messure_exc_time("compress4_dict.pickle", "-> Thời gian đọc của PP nén khối và front coding") #0.015s đến 0.018s