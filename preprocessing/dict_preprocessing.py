from utils import *
import  nltk
nltk.download('punkt')
# code 和 msg 目录
code_and_msg_dir ="./code_and_msg"
# 制作词典（这里有疑惑，词典是为了做做tokenize，那么我应该是需要大量的数据来制作词典才对，而不是单一的这个项目。）
with open(code_and_msg_dir + "/" + "scrcpy1.0_train_code.txt","r",encoding="utf-8") as f:
    all_code = f.read()
with open(code_and_msg_dir + "/" + "scrcpy1.0_test_code.txt","r",encoding="utf-8") as f:
    all_code += f.read()
with open(code_and_msg_dir + "/" + "scrcpy1.0_train_msg.txt","r",encoding="utf-8") as f:
    all_msg = f.read()
with open(code_and_msg_dir + "/" + "scrcpy1.0_test_msg.txt","r",encoding="utf-8") as f:
    all_msg += f.read()
all_content = all_code+all_msg
print("长度为： ",len(all_content))

tokens = nltk.word_tokenize(all_content)
#去重
tokens = sorted(set(tokens))
# 记录频率，可以把频率的低的舍去，也可以不舍去。（这里选择不舍去）
def get_word2tf(content_list):
    # word2tf是记录频率的dic
    word2tf = {}
    for text in content_list:
        str = text.lower()
        word2tf = update_dic(str, word2tf)
    return word2tf
def update_dic(str,word2tf):
    if word2tf.get(str) is None:
        word2tf[str] = 1
    else:
        word2tf[str] += 1
    return  word2tf

word2tf = get_word2tf(tokens)

print("token数： ",len(word2tf)) #7341

# 我们要训练BERT, 所以我们会有一些特殊的token, 例如#CLS#, #PAD#(用来补足长度)等等,
# 所以我们留出前20个token做备用, 实际字的token从序号20开始


# word2idx是我们将要制作的字典
word2idx = {}
# 定义一些特殊token
pad_index = 0 # 用来补长度和空白
unk_index = 1 # 用来表达未知的字, 如果字典里查不到
cls_index = 2 #CLS#
sep_index = 3 #SEP#
mask_index = 4 # 用来做Masked LM所做的遮罩
num_index = 5 # (可选) 用来替换语句里的所有数字, 例如把 "23.9" 直接替换成 #num# # 由于数字很多，我们只把数字归为一个token
word2idx["#PAD#"] = pad_index
word2idx["#UNK#"] = unk_index
word2idx["#SEP#"] = sep_index
word2idx["#CLS#"] = cls_index
word2idx["#MASK#"] = mask_index
word2idx["#NUM#"] = num_index

idx = 20
for char, v in word2tf.items():
    word2idx[char] = idx
    idx += 1

print("index数： ",idx)  #7361

with open('scrcpy1.0_word2idx.json', 'w+', encoding='utf-8') as f:
    f.write(json.dumps(word2idx, ensure_ascii=False))

