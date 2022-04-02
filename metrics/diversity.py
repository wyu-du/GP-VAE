import nltk
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
import argparse
import json

def distinct_4gram(rand_5):
    dist4_list = []
    for hyp5 in rand_5:
        hyp_4grams = []
        for hyp in hyp5:
            hyp_4grams += nltk.ngrams(hyp.split(), 4)
        total_4grams = len(hyp_4grams)
        unique_4grams = len(list(set(hyp_4grams)))
        if total_4grams == 0:
            continue
        dist_4 = unique_4grams/total_4grams
        dist4_list.append(dist_4)
    print('Distinct 4-grams:', np.mean(dist4_list))
    
def compute_self_bleu(rand_5):
    ref_list, hyp_list = [], []
    for i in range(len(rand_5)):
        hyp_all = rand_5[i]
        if len(hyp_all) < 2: continue
        for j, hyp in enumerate(hyp_all):
            cur_ref = hyp_all.copy()
            cur_ref.pop(j)
            tmp = []
            for ref in cur_ref:
                tmp.append(ref.split())
            ref_list.append(tmp)
            hyp_list.append(hyp.split())
            
    score = corpus_bleu(ref_list, hyp_list)
    print("Self-BLEU: "+str(score))
    
    
def unique_sentence(rand_10):
    uni_list = []
    for i in range(len(rand_10)):
        all_sent_list = rand_10[i]
        uni_sents = list(set(all_sent_list))
        uni_list.append(len(uni_sents)/len(all_sent_list))
        
    uni = np.mean(uni_list)
    print('Number of Unique Sentences:', uni)
    
    
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-op', '--output_path', type=str, default="../models/transformer/t5-base_GYAFC/")
    p.add_argument('-of', '--output_file', type=str, default="em_ae_2021-09-09-13-13-04/outs.json")
    args = p.parse_args()

#    rand_5 = [
#            ['you can now check on a facebook chatbot', 'you can now check this .', 
#             'you can now check on a facebook chatbot', 'you should check it on facebook',
#             'please check on a facebook chatbot'],
#            ]
#    rand_10 = [
#            ['you can now check on a facebook chatbot', 'you can now check this .', 
#             'you can now check on a facebook chatbot', 'you should check it on facebook',
#             'please check on a facebook chatbot', 
#             'you can now check on a facebook chatbot', 'you can now check this .', 
#             'you can now check on a facebook chatbot', 'you should check it on facebook',
#             'please check on a facebook chatbot'],
#            ]

    with open(args.output_path + args.output_file, 'r') as f:
        outputs = json.load(f)['values']
    rand_5 = [output['generated'].split('\t')[:5] for output in outputs]
    rand_10 = [output['generated'].split('\t') for output in outputs]

    distinct_4gram(rand_5)
    compute_self_bleu(rand_5)
    unique_sentence(rand_10)