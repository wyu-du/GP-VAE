from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import numpy as np
import argparse
import json

def compute_bleu(references, candidates):
    ref_list, dec_list = [], []
    for i in range(len(candidates)):
        dec_list.append(candidates[i].split())
        if type(references[i]) is list:
            tmp = []
            for ref in references[i]:
                tmp.append(ref.split())
            ref_list.append(tmp)
        else:
            ref_list.append([references[i].split()])
    score = corpus_bleu(ref_list, dec_list)
    print("BLEU: "+str(score))
    
def compute_meteor(references, candidates):
    score_list = []
    for i in range(len(candidates)):
        if type(references[i]) is list:
            ref_list = references[i]
        else:
            ref_list = [references[i]]
        score = meteor_score(ref_list, candidates[i])
        score_list.append(score)
    print("METEOR: "+str(np.mean(score_list)))
    
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-rp', '--reference_path', type=str, default="../data/GYAFC/")
    p.add_argument('-op', '--output_path', type=str, default="../models/transformer/t5-base_GYAFC/")
    p.add_argument('-rf', '--reference_file', type=str, default="em/tst_all.tsv")
    p.add_argument('-of', '--output_file', type=str, default="em_ae_2021-09-09-13-13-04/outs.json")
    args = p.parse_args()

#    references = [
#            ['you can now check on a facebook chatbot', 'you can now check this .', 
#             'you can now check on a facebook chatbot', 'you should check it on facebook',
#             'please check on a facebook chatbot'],
#            ]
#    candidates = ['you can now check on a facebook chatbot', ]
    
    with open(args.reference_path + args.reference_file, 'r') as f:
        refs = f.read().split('\n')
        refs.remove('')
    references = [ref.split('\t')[1:] for ref in refs]
    
    with open(args.output_path + args.output_file, 'r') as f:
        outputs = json.load(f)['values']
    candidates = [output['generated'].split('\t')[0] for output in outputs]

    compute_bleu(references, candidates)
    compute_meteor(references, candidates)