from audioop import reverse
from wsgiref import headers
from xml.dom.minidom import Element
from data import *
import copy
import re
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import numpy as np
from scipy import sparse
from collections import defaultdict
import argparse
from utils import *

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

head2mrr = defaultdict(list)
head2hit_10 = defaultdict(list)
head2hit_1 = defaultdict(list)

class RuleDataset(Dataset):
    def __init__(self, r2mat, rules, e_num,idx2rel, args):
        self.e_num = e_num
        self.r2mat = r2mat
        self.rules = rules
        self.idx2rel = idx2rel
        self.len = len(self.rules)
        self.args = args

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        rel = self.idx2rel[idx]
        _rules = self.rules[rel]
        path_count = np.zeros(shape=(self.e_num,self.e_num))
        for rule in _rules:
            head, body, conf_1, conf_2 = rule
            if conf_1 >=self.args.threshold:
                body_adj = sparse.eye(self.e_num)
                for b_rel in body:
                    body_adj = body_adj * self.r2mat[b_rel] 
                    
                body_adj = body_adj * conf_1
                path_count+=body_adj
        
        return rel, path_count
    
    @staticmethod
    def collate_fn(data):
        head = [_[0] for _ in data]
        path_count = [_[1] for _ in data]
        return head, path_count

def sortSparseMatrix(m, r, rev=True, only_indices=False):
    """ Sort a row in matrix row and return column index
    """
    d = m.getrow(r)
    s = zip(d.indices, d.data)
    sorted_s = sorted(s, key=lambda v: v[1], reverse=rev)
    if only_indices:
        res = [element[0] for element in sorted_s]
    else:
        res = sorted_s
    return res


def remove_var(r):
    """R1(A, B), R2(B, C) --> R1, R2"""
    r = re.sub(r"\(\D?, \D?\)", "", r)
    return r


def parse_rule(r):
    """parse a rule into body and head"""
    r = remove_var(r)
    head, body = r.split(" <-- ")
    body = body.split(", ")
    return head, body


def load_rules(rule_path,all_rules,all_heads):
    with open(rule_path, 'r') as f:
        rules = f.readlines()
        for i_, rule in enumerate(rules):
            conf, r = rule.strip('\n').split('\t')
            conf_1, conf_2 = float(conf[0:5]), float(conf[-6:-1])
            head, body = parse_rule(r)
            # rule item: (head, body, conf_1, conf_2)
            if head not in all_rules:
                all_rules[head] = []
            all_rules[head].append((head, body, conf_1, conf_2))
            
            if head not in all_heads:
                all_heads.append(head)


def construct_rmat(idx2rel, idx2ent, ent2idx, fact_rdf):
    e_num = len(idx2ent)
    r2mat = {}
    # initialize rmat
    for idx, rel in idx2rel.items():
        mat = sparse.dok_matrix((e_num, e_num))
        r2mat[rel] = mat
    # fill rmat
    for rdf in fact_rdf:
        fact = parse_rdf(rdf)
        h, r, t = fact
        h_idx, t_idx = ent2idx[h], ent2idx[t]
        r2mat[r][h_idx, t_idx] = 1
    return r2mat      

def get_gt(dataset):
    # entity
    idx2ent, ent2idx = dataset.idx2ent, dataset.ent2idx
    fact_rdf, train_rdf, valid_rdf, test_rdf = dataset.fact_rdf, dataset.train_rdf, dataset.valid_rdf, dataset.test_rdf
    gt = defaultdict(list)
    all_rdf = fact_rdf + train_rdf + valid_rdf + test_rdf
    for rdf in all_rdf:
        h, r, t = parse_rdf(rdf)
        gt[(h, r)].append(ent2idx[t])
    return gt


def kg_completion(rules, dataset, args):
    """
    Input a set of rules
    Complete Querys from test_rdf based on rules and fact_rdf 
    """
    # rdf_data
    fact_rdf, train_rdf, valid_rdf, test_rdf = dataset.fact_rdf, dataset.train_rdf, dataset.valid_rdf, dataset.test_rdf
    # groud truth
    gt = get_gt(dataset)
    # relation
    rdict = dataset.get_relation_dict()
    head_rdict = dataset.get_head_relation_dict()
    rel2idx, idx2rel = rdict.rel2idx, rdict.idx2rel
    # entity
    idx2ent, ent2idx = dataset.idx2ent, dataset.ent2idx
    e_num = len(idx2ent)
    # construct relation matrix (following Neural-LP)
    r2mat = construct_rmat(idx2rel, idx2ent, ent2idx, fact_rdf+train_rdf+valid_rdf)

    # Eval Metric
    body2mat  = {}
    
    rule_dataset = RuleDataset(r2mat, rules, e_num, idx2rel, args)
    rule_loader = DataLoader(
            rule_dataset, 
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=RuleDataset.collate_fn
        )
    
    for epoch, sample in enumerate(rule_loader):
        heads, score_counts = sample
        
        for idx in range(len(heads)):
            head = heads[idx]
            score_count = score_counts[idx]
            body2mat[head] = score_count

    
    mrr, hits_1, hits_10  = [], [], []
    
    for q_i, query_rdf in enumerate(test_rdf):
        query = parse_rdf(query_rdf)
        q_h, q_r, q_t = query
        if q_r not in body2mat:
            continue
        print ("{}\t{}\t{}".format(q_h, q_r, q_t))
        pred = np.squeeze(np.array(body2mat[q_r][ent2idx[q_h]]))

        if pred[ent2idx[q_t]]!=0:
            pred_ranks = np.argsort(pred)[::-1]    
            
            truth = gt[(q_h, q_r)]
            truth = [t for t in truth if t!=ent2idx[q_t]]

            
            filtered_ranks = []
            for i in range(len(pred_ranks)):
                idx = pred_ranks[i]
                if idx not in truth:
                    filtered_ranks.append(idx)
                    
            rank = filtered_ranks.index(ent2idx[q_t])+1
            
        else:
            truth = gt[(q_h, q_r)]
            
            filtered_pred = []
            
            for i in range(len(pred)):
                if i not in truth:
                    filtered_pred.append(pred[i])
                    
            
            n_non_zero = np.count_nonzero(filtered_pred)
            rank = n_non_zero+1

        
        mrr.append(1.0/rank)
        head2mrr[q_r].append(1.0/rank)
        
        hits_1.append(1 if rank<=1 else 0)
        hits_10.append(1 if rank<=10 else 0)
        head2hit_1[q_r].append(1 if rank<=1 else 0)
        head2hit_10[q_r].append(1 if rank<=10 else 0)
        print("rank at {}: {}".format(q_i, rank))


    print("MRR: {} Hits@1: {} Hits@10: {}".format(np.mean(mrr), np.mean(hits_1), np.mean(hits_10)))


def feq(relation, fact_rdf):
    count = 0
    for rdf in fact_rdf:
        h, r, t= parse_rdf(rdf)
        if r == relation:
            count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="family")
    parser.add_argument("--rule", default="family")
    parser.add_argument('--cpu_num', type=int, default=mp.cpu_count()//2)   
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0)
    args = parser.parse_args()
    dataset = Dataset(data_root='../datasets/{}/'.format(args.data), inv=True)
    all_rules = {}
    all_rule_heads = []
   
    for L in range(2,4):
        file = "{}/{}_500_{}.txt".format(args.rule, args.rule, L)
        load_rules("{}".format(file), all_rules, all_rule_heads)
    
    for head in all_rules:
        all_rules[head] = all_rules[head][:args.top]
    
    fact_rdf, train_rdf, valid_rdf, test_rdf = dataset.fact_rdf, dataset.train_rdf, dataset.valid_rdf, dataset.test_rdf

    print_msg("distribution of test query")
    for head in all_rule_heads:
        count = feq(head, test_rdf)
        print("Head: {} Count: {}".format(head, count))
    
    print_msg("distribution of train query")
    for head in all_rule_heads:
        count = feq(head, fact_rdf+valid_rdf+train_rdf)
        print("Head: {} Count: {}".format(head, count))


    kg_completion(all_rules, dataset,args)
    
    print_msg("Stat on head and hit@1")
    for head, hits in head2hit_1.items():
        print(head, np.mean(hits))

    print_msg("Stat on head and hit@10")
    for head, hits in head2hit_10.items():
        print(head, np.mean(hits))

    print_msg("Stat on head and mrr")
    for head, mrr in head2mrr.items():
        print(head, np.mean(mrr))
