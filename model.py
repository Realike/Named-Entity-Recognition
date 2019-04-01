import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)

    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        # 初始化embedding, shape(vocab_size, embedding_dim)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # hiden_dim // 2 cause num_directions = 2
        # nn.LSTM(input_features=embedding_dim, output_features=hidden_dim//2 (bidirectional))
        # 总体matrix shape乘法: matrix * (nn.LSTM) * matrix
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the  start tag and we never transfer from the stop tag
        # 此处只考虑了name->start和stop->name转移可能性为0(命名体识别BIO足够)，可能也有其他考虑性(未添加)，比如词性标注名词后接名词
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        # print(self.transitions)

        self.hidden = self.init_hidden()

    # 初始化h_0, c_0
    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        # 初始化设所有除START_TAG=0外为-1000，为了将第一个数据取log exp的数据只有start标签起作用
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # print('init:',init_alphas)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.


        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                # feat[next_tag] to torch type and expand
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # print(emit_score)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # print(trans_score)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))  # alphas_t append的类型是tensor([0.3473], grad_fn=<ViewBackward>)
            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    # 返回状态矩阵probability
    def _get_lstm_features(self, sentence):
        # 初始化采用init_hidden()为bi-lstm前向后向h_0, c_0
        # bi-lstm 前向(或后向)last_hidden shape(num_layers*num_directions=2, batch=1, hidden=hidden_dim // 2)
        # 每次forward都初始化hidden，hidden只影响当前序列，对全局model weights不影响
        self.hidden = self.init_hidden()
        # sentence tensor([...])
        # word_embeds 构成lstm输入格式 --> shape(input_seq=len(sentence), batch=1, hidden=EMBEDDING_DIM)
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        # lstm_out shape(input_seq=len(sentence), batch=1, hidden=hidden_dim)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # to shape(len(sentence), hidden_dim)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        # lstm_feats shape(len(sentence), tagset_size)(matrix表示第i个词转到tags id的概率)  未softmax(此模型只需求score，不需softmax都可)
        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats

    # 返回当前序列对应的score
    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        # 将tags id前加入START_TAG, len(tags) + = 1
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            # tags[i=0]为STRAT_TAG，self.transitions[tags[i + 1], tags[i]]: 第i个tags id到第i+1个tags id转移概率
            # feat[tags[i + 1]] 表示当前某个序列id
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        # 加上tags末尾到STOP_TAG转移概率
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        # 初始化设所有除START_TAG=0外为-1000，为了将第一个数据取log exp的数据只有start标签起作用
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        # print('terminal_var:', terminal_var)
        # print('backpointers:', backpointers)
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path. 回溯
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    # 求loss，计算得分loss=-(log(exp(s) / exp`(s)))=log(exp`(s)) - log(exp(s))=forward_score - gold_score
    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Parameter sentence 表示 sentence id序列
        # Get the emission scores from the BiLSTM，传入sentence从已经训练好的模型得出状态矩阵probability
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
