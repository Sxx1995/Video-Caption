import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from builtins import range
import random
import math

class LanguageModel(nn.Module):

    def __init__(self, frame_size, projected_size, hidden_size,
                 num_frames, num_words, vocab):
        '''
        frame_size: The feature size of each video frame, usually 4096 (The penultimate fc layer of VGG)
        projected_size: Projected dimension of all features
        hidden_size: LSTM hiddens size
        num_frames: Number of (frames of video features), default is 60
        num_words: The (sequence length/number of words) of text feature, default is 30
        '''
        super(LanguageModel, self).__init__()

        self.frame_size = frame_size
        self.hidden_size = hidden_size
        self.num_frames = num_frames
        self.num_words = num_words
        self.projected_size = projected_size
        self.vocab = vocab
        self.vocab_size = len(vocab)

        # frame_embed is used to embed the frame feature to low-dim space
        self.vs_frame_embed = nn.Linear(frame_size, projected_size)
        self.vs_frame_drop = nn.Dropout(p=0.8)
        self.vf_frame_embed = nn.Linear(frame_size, projected_size)
        self.vf_frame_drop = nn.Dropout(p=0.8)
        self.frame_embed = nn.Linear(projected_size * 2, projected_size)
        self.frame_drop = nn.Dropout(p=0.8)

        # attend_layer is used for temporal attention
        self.attend_layer = AttentionLayer(hidden_size, projected_size)

        # word_embed is used to embed the text features to low-dim space
        self.word_embed = nn.Embedding(self.vocab_size, projected_size)
        self.word_drop = nn.Dropout(p=0.8)

        # lstm is used as decoder
        self.lstm_cell = nn.LSTMCell(projected_size, hidden_size)
        self.lstm_drop = nn.Dropout(p=0.8)
        # inith is used to initialize the hidden state of LSTM
        self.inith = nn.Sequential(
            nn.Linear(projected_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        # initc is used to initialize the cell of LSTM
        self.initc = nn.Sequential(
            nn.Linear(projected_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        # linear is used to map back the output of LSTM to vocab space
        self.linear = nn.Linear(hidden_size, self.vocab_size)

        self._init_weights()

    def _init_weights(self):
        variance = math.sqrt(2.0 / (self.frame_size + self.projected_size))
        self.vs_frame_embed.weight.data.normal_(0.0, variance)
        self.vs_frame_embed.bias.data.zero_()
        self.vf_frame_embed.weight.data.normal_(0.0, variance)
        self.vf_frame_embed.bias.data.zero_()
        self.word_embed.weight.data.uniform_(-1.73, 1.73)
        self.linear.weight.data.uniform_(-0.08, 0.08)
        self.linear.bias.data.zero_()

    def _init_lstm_state(self, v):
        mean_v = torch.mean(v, 1).squeeze(1)
        lstm_hidden = F.tanh(self.inith(mean_v))
        lstm_cell = F.tanh(self.initc(mean_v))
        return lstm_hidden, lstm_cell

    def forward(self, video_feats, captions, teacher_forcing_ratio=0.5):
        '''
        Input the video frame features and GT caption, return the generated caption
        Not teacher-forceing mode (input the GT caption to LSTM)
        Input the previous generated results (as the current input)
        UPDATED: in the end, we combine it with teacher forcing node to ease the convergence
        '''
        batch_size = len(video_feats)
        # Check if the current caption is None or not, if None, then is inference mode
        infer = True if captions is None else False

        # Encoding process!
        # vs is the feature of saliency region of each frame
        vs = video_feats[:, :, :self.frame_size].contiguous()
        vs = vs.view(-1, self.frame_size)
        vs = self.vs_frame_embed(vs)
        vs = self.vs_frame_drop(vs)
        vs_ = vs.view(batch_size, self.num_frames, -1)
        # vf if the full feature of each frame
        #vf = video_feats[:, :, self.frame_size:].contiguous()
        vf = video_feats[:, :, :self.frame_size].contiguous()
        vf = vf.view(-1, self.frame_size)
        vf = self.vf_frame_embed(vf)
        vf = self.vf_frame_drop(vf)
        # vf_ = vf_.view(batch_size, self.num_frames, -1)
        # vr is the residual between full frame feature and saliency region feature
        vr = vf - vs
        # v is the concatenated feature of vs, and vr
        v = torch.cat([vs, vr], 1)
        v = self.frame_embed(v)
        v = v.view(batch_size, self.num_frames, -1)

        # Initialize the hidden state of LSTM
        lstm_hidden, lstm_cell = self._init_lstm_state(v)

        # Decoding process, and start predict words
        outputs = []
        attens = []
        # Input a <start> token first
        word_id = self.vocab('<start>')
        word = Variable(vs.data.new(batch_size, 1).long().fill_(word_id))
        word = self.word_embed(word).squeeze(1)
        word = self.word_drop(word)

        for i in range(self.num_words):
            if not infer and captions[:, i].data.sum() == 0:
                # The id of <pad> is 0, if all the word id is 0,
                # then come to an end and stop prediction
                break
            a = self.attend_layer(lstm_hidden, vs_)
            if infer:
                attens.append(a)
            a = a.unsqueeze(1)
            # Consider the concatenation of the full feature and saliency feature
            V = torch.bmm(a, v).squeeze(1)

            t = word + V
            lstm_hidden, lstm_cell = self.lstm_cell(t, (lstm_hidden, lstm_cell))
            lstm_hidden = self.lstm_drop(lstm_hidden)

            word_logits = self.linear(lstm_hidden)
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                # teacher forcing mode
                word_id = captions[:, i]
            else:
                # non-teacher forcing mode
                word_id = word_logits.max(1)[1]
            if infer:
                # If inference mode, then return the word id
                outputs.append(word_id)
            else:
                # If no inference mode, then training mode, return the logits
                outputs.append(word_logits)
            # Get the word representation of the next input word
            word = self.word_embed(word_id).squeeze(1)
            word = self.word_drop(word)
        # Unsqueeze(1) will draw a vector (n) to a column vector (nx1)
        # Each vector in outputs is one-time output of each batch
        # Draw them to column vectors and concatenate them, then we have the final outputs
        outputs = torch.cat([o.unsqueeze(1) for o in outputs], 1).contiguous()
        return outputs, attens

    def sample(self, video_feats):
        '''
        During sampling, we do not input caption, and forward without teacher-forcing
        '''
        return self.forward(video_feats, None, teacher_forcing_ratio=0.0)

    def decode_tokens(self, tokens):
        '''
        Get the caption according to word id(token) list and dictionary
        '''
        words = []
        for token in tokens:
            if token == self.vocab('<end>'):
                break
            word = self.vocab.idx2word[token]
            words.append(word)
        caption = ' '.join(words)
        return caption
