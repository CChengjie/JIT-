from utils import *

class Dataset(Dataset):
    def __init__(self, data_path, word2idx, max_seq_len, data_regularization=False):
        self.dafa_regularization = data_regularization
        self.word2idx = word2idx
        # define max length
        self.max_seq_len = max_seq_len
        # directory of data dataset
        self.data_path = data_path
        # define special symbols
        self.pad_index = 0
        self.unk_index = 1
        self.cls_index = 2
        self.sep_index = 3
        self.mask_index = 4

        # load data  [test]
        with open(data_path, "r", encoding="utf-8") as f:
            # 将数据集全部加载到内存
            self.lines = [eval(line) for line in tqdm.tqdm(f, desc="Loading Dataset")]
            # 打乱顺序
