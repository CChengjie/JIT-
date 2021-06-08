"""Pytorch BERT model"""
from utils import *

def gelu(x):
    """ gelu 激活 不同于OpenAI GPT's gelu
    """
    return x*0.5*(1.0 + torch.erf(x/math.sqrt(2.0)))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}

class BertConfig(object):
    """ configuration of a BertModel"""
    def __init__(self):
        self.model_name = 'bert'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 2
        self.bert_path = './Model'
        self.hidden_size = 768  #embedding dim
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.batch_size = 16
        self.num_epochs = 5
"""使用 mean max pool 的方式"""
"""
    沿着sequence length 的维度分别求均值和max，拼成一条向量，映射成一个值再激活
"""
class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path) #Bert预训练模型
        for param in self.bert.parameters():
            param.requires_grad = True    #requeires_grad=False准确率下降，速度变快
        self.cls = nn.Linear(3*config.hidden_size,config.hidden_size) #nn.Linear(input dim,output dim)
        self.final_cls = nn.Linear(config.hidden_size,1)# 用来把pooled_output也就是对应#CLS#的那一条向量映射为2分类 ，output dim=1 ，embeddingsize=1，为一个数，
        self.activation = nn.Sigmoid() #激活函数，表示模型的输出推断，值介于0和1之间

    def compute_loss(self, predictions, labels):
        # 将预测和标记的维度展平，防止出现维度不一致
        predictions = predictions.view(-1)
        labels = labels.float().view(-1)
        epsilon = 1e-8
        # 交叉熵
        loss = \
            - labels * torch.log(predictions + epsilon) - \
            (torch.tensor(1.0) - labels) * torch.log(torch.tensor(1.0) - predictions + epsilon)
        # 求均值， 并返回可以反转的loss
        # loss为一个实数
        loss = torch.mean(loss)
        return loss
    def forward(self, bef_input, aft_input, msg_input, labels=None):
        bef_input_ids, bef_input_mask, bef_input_types = bef_input[0], bef_input[1], bef_input[2]
        aft_input_ids, aft_input_mask, aft_input_types = aft_input[0], aft_input[1], aft_input[2]
        msg_input_ids, msg_input_mask, msg_input_types = msg_input[0], msg_input[1], msg_input[2]
        """
        encoded_layers, pooled_output = model(input_ids, attention_mask, token_types)  ->  
        源码中class BertModel: 
        def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True)
        
        """
        bef_encoded_layers,  = self.bert(input_ids = bef_input_ids,
                                  attention_mask = bef_input_mask,
                                  token_type_ids = bef_input_types,
                                  output_all_encoded_layers = True)
        aft_encoded_layers,  = self.bert(input_ids=aft_input_ids,
                                  attention_mask=aft_input_mask,
                                  token_type_ids=aft_input_types,
                                  output_all_encoded_layers = True)
        msg_encoded_layers,  = self.bert(input_ids=msg_input_ids,
                                  attention_mask=msg_input_mask,
                                  token_type_ids=msg_input_types,
                                  output_all_encoded_layers = True)
        # sequence_output = encoded_layers[-1]
        # pooled_output = self.pooler(sequence_output)   pooled_output为隐藏层中#CLS#对应的token的一条向量
        # 在forward函数中，如果output_all_encoded_layers=True，那么encoded_layer就是12层transformer的结果，
        # 否则只返回最后一层transformer的结果，pooled_output的输出就是最后一层Transformer经过self.pooler层后得到的结果。
        bef_sequence_output = bef_encoded_layers[2]
        aft_sequence_output = aft_encoded_layers[2]
        msg_sequence_output = msg_encoded_layers[2]
        #对各行求均值
        bef_avg_pooled = bef_sequence_output.mean(1)
        aft_avg_pooled = aft_sequence_output.mean(1)
        msg_avg_pooled = msg_sequence_output.mean(1)
        #求最大值
        bef_max_pooled = torch.max(bef_sequence_output, dim=1)
        aft_max_pooled = torch.max(bef_sequence_output, dim=1)
        msg_max_pooled = torch.max(bef_sequence_output, dim=1)
        #拼成一条向量
        bef_pooled = torch.cat((bef_avg_pooled, bef_max_pooled[0]), dim=1)
        aft_pooled = torch.cat((aft_avg_pooled, aft_max_pooled[0]), dim=1)
        msg_pooled = torch.cat((msg_avg_pooled, msg_max_pooled[0]), dim=1)

        features = torch.cat((bef_pooled, aft_pooled, msg_pooled), dim=1)
        features = self.cls(features)
        features = self.final_cls(features)
        #用sigmod函数做激活，返回0-1值
        predictions = self.activation(features)

        if labels is not None:
            #计算loss
            loss = self.compute_loss(predictions, labels)
            return  predictions, loss
        else :
            return predictions

