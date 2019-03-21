import torch
import pyadamcuda as ac
import numpy as np
import json

def load_json(path):
    if path.endswith(".json"):
        with open(path) as json_data:
            #print path
            d = json.load(json_data)
            json_data.close()
            return d
    return 0

import numpy as np
from scipy.sparse import csr_matrix
import itertools
def init_sparse(data):

    sparse_matrix = None
    row_arr = None
    col_arr = None
    data_arr = None
    shape = None
    i=0
    for item in data:
        if i==0:
            shape = (item[0], item[1])
            row_arr = np.zeros(shape=(len(data)-1), dtype=np.float32)
            col_arr = np.zeros(shape=(len(data)-1), dtype=np.float32)
            data_arr = np.zeros(shape=(len(data)-1), dtype=np.float32)
        else:
            row_arr[i-1] = item[0]
            col_arr[i-1] = item[1]
            data_arr[i-1] = item[2]
            pass

        i+=1

    return csr_matrix( (data_arr,(row_arr,col_arr)), shape, dtype=np.float32), row_arr, col_arr, data_arr, shape

def init_mat(data):
    rows = len(data)
    cols = len(data[0])
    internal_data = []
    for row in data:
        for item in row:
            internal_data.append(item)
    return np.array(internal_data, dtype=np.float32).reshape((rows, cols))

def init_mat_int(data):
    rows = len(data)
    cols = len(data[0])
    internal_data = []
    for row in data:
        for item in row:
            internal_data.append(item)
    return np.array(internal_data, dtype=np.int32).reshape((rows, cols))

def init_vec(data):
    return np.array(data, dtype=np.float32).reshape((len(data),1))
