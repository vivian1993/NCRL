import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import sample 
import random
from torch.nn.utils import clip_grad_norm_
import time
import pickle
import argparse
import numpy as np
import heapq
from os import path
from data import *
from utils import *
from model import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print_msg(str(device))

rule_conf = []

def sample_training_data(max_path_len, anchor_num, fact_rdf, entity2desced, head_rdict):
    print("Sampling training data...")
    anchors_rdf = sample_anchor_rdf(fact_rdf, num=anchor_num)
    train_rule, train_rule_idx = [],[]
    len2train_rule_idx = {}
    sample_number = 0
    for anchor_rdf in anchors_rdf:
        rule_seq, record = construct_rule_seq(fact_rdf, anchor_rdf, entity2desced, max_path_len, PRINT=False)
        sample_number += len(record)
        if len(rule_seq) > 0:
            train_rule += rule_seq
            for rule in rule_seq:
                idx = torch.LongTensor(rule2idx(rule, head_rdict))
                train_rule_idx.append(idx)
                # cluster rules according to its length
                body_len = len(idx) - 2
                if body_len in len2train_rule_idx.keys():
                    len2train_rule_idx[body_len] += [idx]
                else:
                    len2train_rule_idx[body_len] = [idx]
    rule_len_range = list(len2train_rule_idx.keys())
    print("Fact set number:{} Sample number:{}".format(len(fact_rdf), sample_number))
    return len2train_rule_idx


def enumerate_body(relation_num, rdict, body_len):
    import itertools
    all_body_idx = list(list(x) for x in itertools.product(range(relation_num), repeat=body_len))
    # transfer index to relation name
    idx2rel = rdict.idx2rel
    all_body = []
    for b_idx_ in all_body_idx:
        b_ = [idx2rel[x] for x in b_idx_]
        all_body.append(b_)
    return all_body_idx, all_body


def get_body(args, r_num, rdict, body_len):
    body_path = "../rules/{}/body_{}".format(args.data, body_len)
    if not path.isfile(body_path):
        _, body = enumerate_body(r_num, rdict, body_len=body_len)
        print("Dumping {}".format(body_path))
        with open(body_path, 'wb') as f:
            pickle.dump(body, f)
    else:
        print("Loading {} ".format(body_path))
        with open(body_path, 'rb') as f:
            body = pickle.load(f)
    return body


def generate_phb_matrix(args, dataset):
    """
    Load model
    """
    print("Loading models...")
    with open('../results/model_True', 'rb') as g:
        rnn = pickle.load(g)
    rnn.eval()
    head_rdict = dataset.get_head_relation_dict()
    rdict = dataset.get_relation_dict()
    r_num = head_rdict.__len__()-1
    body_2 = get_body(args, r_num, rdict, body_len=2)
    body = body_2 #+ body_3 #+ body_4
    body_list = ["|".join(b) for b in body]

    prob_matrix = np.zeros([len(body_list), r_num])
    for bi, body in enumerate(body_list):
        body_idx = body2idx(body, head_rdict)
        prob_body = 1
        rel2prob = []
        with torch.no_grad():
            inputs = torch.LongTensor([body_idx]).to(device)
            pred_head = rnn.predict_head_recursive(inputs)
            prob_ = pred_head.squeeze(0).tolist()
            for i_, p_ in enumerate(prob_):
                r_ = head_rdict.idx2rel[i_]
                if "inv_" in r_:
                    prob_[i_] = -1000.0
            prob_ = torch.softmax(torch.Tensor(prob_), dim=0).tolist()
            for i_, p_ in enumerate(prob_):
                head_r_ = head_rdict.idx2rel[i_]
                if "inv_" not in head_r_: 
                    rel2prob.append((head_r_, p_))
        for head_r_, p_ in rel2prob:
            if head_r_ != "None" and "inv_" not in head_r_:
                hi = head_rdict.rel2idx[head_r_]
                prob_matrix[bi][hi] = p_
    print("Saving prob of head-body matrix")
    with open('../rules/{}/phb_mat'.format(args.data), 'wb') as f:
        pickle.dump(prob_matrix, f)
    return
        

def generate_rules(args, dataset):

    rule_path = "../rules/{}/ours_top{}.txt".format(args.data, args.topk)

    head_rdict = dataset.get_head_relation_dict()
    rdict = dataset.get_relation_dict()
    r_num = head_rdict.__len__()-1
    body_2 = get_body(args, r_num, rdict, body_len=2)
    phb_mat_path = "../rules/{}/phb_mat".format(args.data)
    print("Loading {}".format(phb_mat_path))
    with open(phb_mat_path, 'rb') as f:
        prob_matrix = pickle.load(f)
    with open(rule_path, 'w') as f:
        for hi, head in head_rdict.idx2rel.items():
            if head != "None" and "inv_" not in head:
                prob_ = prob_matrix[:][hi]
                topk_idx = heapq.nlargest(args.topk, range(len(prob_)), prob_.take)
                for k in topk_idx:
                    body = body_2[k]
                    p = prob_[k]
                    rule = "{:.3f} ({:.3f})\t{} <-- ".format(p, p, head)
                    rule += ", ".join(body)
                    f.write(rule + '\n')
    return

if __name__ == '__main__':
    msg = "First Order Logic Rule Mining"
    print_msg(msg)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="fb15k-237", help="increase output verbosity")
    parser.add_argument("--topk", type=int, default=10, help="increase output verbosity")
    parser.add_argument("--phb", action="store_true", help="generate prob matrix phb")
    parser.add_argument("--gpu", type=int, default=1, help="increase output verbosity")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    # DataSet
    data_path = '../datasets/{}/'.format(args.data)
    dataset = Dataset(data_root=data_path, inv=True)
    print("Dataset:{}".format(data_path)) 

    if args.phb:
        print_msg(" Generate phb matrix ")
        generate_phb_matrix(args, dataset)


    print_msg(" Generate {} Rules For each Head".format(args.topk))
    generate_rules(args, dataset)

