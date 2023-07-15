import os
import re
import time
import pickle
import json
import random
from random import seed, randrange
import argparse

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.model_selection import train_test_split

# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModel
# from nltk.translate.bleu_score import sentence_bleu
# import sacrebleu
# import bert_score

# from models.LSTMBahdanau import Encoder, Decoder, Seq2Seq
# import models as Model
from models.BiGRU import Encoder, Decoder, Seq2Seq
from utils.tokenizer import Tokenizer, pad_sequences, MyData
from utils.preprocess import preprocess_1, preprocess_2
from trainer import train, loss_function, sort_within_batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--dataset_path', required=True, help='path to dataset (csv)')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--model', required=True, help='select model')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--hidden_size', type=int, default=1024, help='the size of the LSTM hidden state')
    parser.add_argument('--embedding_dim', type=int, default=512, help='the size of the embedding dimension')
    parser.add_argument('--dropout', type=int, default=.5, help='dropout ratio')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layer in LSTM cell')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping value. default=5')
    parser.add_argument('--split_ratio', type=float, default=0.2,
                        help='assign ratio for split dataset')
    parser.add_argument('--max_length', type=int, default=50, help='maximum-sentence-length')
    parser.add_argument('--tokenizer_path', type=str, help='tokenizer path')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--wandb_log', action='store_true', help='using wandb')
    
    opt = parser.parse_args()
    
    RANDOM_SEED = opt.manualSeed
    dataset_dir = opt.dataset_path
    
    if not opt.exp_name:
        opt.exp_name = f'{opt.model}-{opt.dataset}'
#         opt.exp_name += f'-MaxLen{opt.max_length}'
        
    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)
    
    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)
    
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)
    
    df = pd.read_csv(opt.dataset_path)
    df = df.dropna()
    df['questions_preprocessed'] = df['questions'].apply(preprocess_1)
    df['answers_preprocessed'] = df['answers'].apply(preprocess_1)
    
    if opt.tokenizer_path is not None:
        with open(opt.tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        tokenizer = Tokenizer(pd.concat([df['questions'], df['answers']], axis=0).values, min_freq=1)
        with open(f'./saved_models/{opt.exp_name}/tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    max_len = opt.max_length+2
  
    df['questions_preprocessed'] = df['questions'].map(lambda x: preprocess_2(x))
    df['answers_preprocessed'] = df['answers'].map(lambda x: preprocess_2(x))

    df['questions_preprocessed'] = df['questions_preprocessed'].map(lambda x: tokenizer.text_to_sequence(x))
    df['questions_preprocessed'] = df['questions_preprocessed'].map(lambda x: pad_sequences(x, max_len))

    df['answers_preprocessed'] = df['answers_preprocessed'].map(lambda x: tokenizer.text_to_sequence(x))
    df['answers_preprocessed'] = df['answers_preprocessed'].map(lambda x: pad_sequences(x, max_len))
    
    df_train, df_test = train_test_split(df, test_size=opt.split_ratio, random_state=RANDOM_SEED)
    print(f"Train Data \t: {len(df_train)}")
    print(f"Test Data\t: {len(df_test)}\n")
    
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    INPUT_SIZE_ENCODER = len(tokenizer.vocab)+4
    INPUT_SIZE_DECODER = len(tokenizer.vocab)+4
    OUTPUT_SIZE = len(tokenizer.vocab)+4
    VOCAB_LEN = len(tokenizer.vocab)+4
    
    EMBEDDING_DIM = opt.embedding_dim
    HIDDEN_SIZE = opt.hidden_size
    BATCH_SIZE = opt.batch_size
    NUM_LAYERS = opt.num_layers
    DROPOUT = opt.dropout
    
    input_tensor_train = df_train['questions_preprocessed'].values.tolist()
    target_tensor_train = df_train['answers_preprocessed'].values.tolist()

    input_tensor_test = df_test['questions_preprocessed'].values.tolist()
    target_tensor_test = df_test['answers_preprocessed'].values.tolist()

    train_data = MyData(input_tensor_train, target_tensor_train)
    test_data = MyData(input_tensor_test, target_tensor_test)

    train_dataset = DataLoader(train_data, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)
    test_dataset = DataLoader(test_data, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)
    
#     if opt.model == 'BiLSTM':
#         Encoder = Model.BiLSTM.Encoder
#         Decoder = Model.BiLSTM.Decoder
#         Seq2Seq = Model.BiLSTM.Seq2Seq
#     else:
#         raise Exception("Model name invalid")
    
    encoder_net = Encoder(INPUT_SIZE_ENCODER, EMBEDDING_DIM, HIDDEN_SIZE, 
                  NUM_LAYERS, DROPOUT, pretrained_word_embedding=False, embedding_matrix=None, freeze=False).to(device)

    decoder_net = Decoder(INPUT_SIZE_DECODER, EMBEDDING_DIM, HIDDEN_SIZE, 
                          OUTPUT_SIZE, NUM_LAYERS, DROPOUT, pretrained_word_embedding=False, embedding_matrix=None, freeze=False).to(device)

    model = Seq2Seq(encoder_net, decoder_net, vocab_len=VOCAB_LEN).to(device)
    
    train(model=model, num_epochs=opt.num_epochs, lr=opt.lr, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=test_dataset, save_dir=opt.exp_name, RANDOM_SEED=RANDOM_SEED, device=device, opt=opt, crit='CEL')