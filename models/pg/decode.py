import os
import time
import nltk
import numpy as np
import pandas as pd
import random
import torch
from torch.autograd import Variable
from heapq import nlargest
from copynet.utils import get_decode_input_from_batch

class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage
        
    def extend(self, token, log_prob, state, context, coverage):
        return Beam(tokens = self.tokens+[token],
                    log_probs = self.log_probs+[log_prob],
                    state = state,
                    context = context,
                    coverage = coverage)
        
    @property
    def latest_token(self):
        return self.tokens[-1]
    
    @property
    def avg_log_prob(self):
        return sum(self.log_probs)/len(self.tokens)
    
    
class BeamSearch(object):
    def __init__(self, model, txtfield, args):
        self.model = model
        self.args = args
        self.beam_size = args.beam_size
        self.max_len = args.max_len
        self.using_cuda = args.using_cuda
        self.model_type = args.model_type
        self.vocab_size = args.vocab_size
        self.pad = txtfield.vocab.stoi[txtfield.pad_token] # index of <pad>
        self.unk = txtfield.vocab.stoi[txtfield.unk_token] # index of <unk>
        self.sos = txtfield.vocab.stoi[txtfield.init_token] # index of <sos>
        self.eos = txtfield.vocab.stoi[txtfield.eos_token] # index of <eos>
        
    def sort_beams(self, beams):
        return sorted(beams, key=lambda h:h.avg_log_prob, reverse=True)
    
    def ids_to_tokens(self, input_ids):
        if self.sos in input_ids:
            input_ids.remove(self.sos)
        try:
            fst_stop_idx = input_ids.index(self.eos)
            input_ids = input_ids[:fst_stop_idx]
        except ValueError:
            input_ids = input_ids
        input_words = [self.model.txtfield.vocab.itos[t] for t in input_ids]
        return input_words
    
    def dec_ids_to_tokens(self, dec_ids, oovs):
        dec_words = []
        for t in dec_ids:
            if t < self.vocab_size:
                if t == self.eos: 
                    break
                try:
                    word = self.model.txtfield.vocab.itos[t]
                except:
                    word = self.model.txtfield.vocab.itos[self.unk]
            else:
                wid = oovs[t - self.vocab_size]
                word = self.model.txtfield.vocab.itos[wid]
            dec_words.append(word)
        return dec_words
    
    def beam_search(self, batch, bid):
        sorted_input, sorted_lengths, enc_padding_mask, sorted_idx,\
        enc_input_ext, extra_zeros, c_t_0, coverage_t_0, batch_oovs = \
        get_decode_input_from_batch(batch, self.model.txtfield, self.args, bid)
        
        enc_outputs, enc_hidden = self.model.encoder(sorted_input, sorted_lengths, sorted_idx)
        dec_h, dec_c = self.model.reduce_state(enc_hidden, sorted_idx)
        
        dec_h = dec_h.squeeze(0) # B x 2H
        dec_c = dec_c.squeeze(0) # B x 2H
        
        if self.model_type != "copynet":
            z = self.model.mean(enc_outputs)
        else:
            z = self.model.hidden2latent(enc_outputs)
            
        beams = [Beam(tokens=[self.sos], log_probs=[0.0], 
                      state=(dec_h[0], dec_c[0]),
                      context=c_t_0[0],
                      coverage=(coverage_t_0[0])) 
                for i in range(self.beam_size)] # B x H
        results = []
        steps = 0
        while steps < self.max_len and len(results) < self.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t<self.vocab_size else self.unk 
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if self.using_cuda: y_t_1 = y_t_1.cuda()
            
            all_state_h, all_state_c, all_context = [], [], []
            for beam in beams:
                all_state_h.append(beam.state[0])
                all_state_c.append(beam.state[1])
                all_context.append(beam.context)
            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0),
                     torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0) 
            
            coverage_t_1 = None
            all_coverage = []
            for beam in beams:
                all_coverage.append(beam.coverage)
            coverage_t_1 = torch.stack(all_coverage, 0)
            
            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1, z,
                                                                                    enc_padding_mask,  c_t_1, 
                                                                                    extra_zeros, enc_input_ext,
                                                                                    coverage_t_1, steps)
            log_probs = torch.log(final_dist)
            topk_logp, topk_ids = torch.topk(log_probs, self.beam_size*2)
            
            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze(0)
            dec_c = dec_c.squeeze(0)
            
            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i])

                for j in range(self.beam_size*2):
                    new_beam = h.extend(token=topk_ids[i,j].item(),
                                        log_prob=topk_logp[i,j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
                    all_beams.append(new_beam)
            
            beams = []
            for h in self.sort_beams(all_beams):
                beams.append(h)
                if len(beams) == self.beam_size: 
                    break
            
            for h in beams:
                if h.latest_token == self.eos:
                    results.append(h)
                else:
                    continue
            steps += 1
            
        if len(results) == 0:
            results = beams
        beams_sorted = self.sort_beams(results)
        result = {i:self.dec_ids_to_tokens(beams.tokens[1:], batch_oovs[0]) 
                    for i, beams in enumerate(beams_sorted)}
        return result
    
    def sample_beam_search(self, batch, bid):
        sorted_input, sorted_lengths, enc_padding_mask, sorted_idx,\
        enc_input_ext, extra_zeros, c_t_0, coverage_t_0, batch_oovs = \
        get_decode_input_from_batch(batch, self.model.txtfield, self.args, bid)
        
        enc_outputs, enc_hidden = self.model.encoder(sorted_input, sorted_lengths, sorted_idx)
        dec_h, dec_c = self.model.reduce_state(enc_hidden, sorted_idx)
        
        dec_h = dec_h.squeeze(0) # B x 2H
        dec_c = dec_c.squeeze(0) # B x 2H
        
        if self.model_type != "copynet":
            mean = self.model.mean(enc_outputs) # B x L x K
            logvar = self.model.logvar(enc_outputs) # B x L x K
#            z = self.model.reparameterize(mean, logvar)
            z = self.reparameterize_std(mean, logvar, self.args.std)
        else:
            z = self.model.hidden2latent(enc_outputs)
            
        beams = [Beam(tokens=[self.sos], log_probs=[0.0], 
                      state=(dec_h[0], dec_c[0]),
                      context=c_t_0[0],
                      coverage=(coverage_t_0[0])) 
                for i in range(self.beam_size)] # B x H
        results = []
        steps = 0
        while steps < self.max_len and len(results) < self.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t<self.vocab_size else self.unk 
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if self.using_cuda: y_t_1 = y_t_1.cuda()
            
            all_state_h, all_state_c, all_context = [], [], []
            for beam in beams:
                all_state_h.append(beam.state[0])
                all_state_c.append(beam.state[1])
                all_context.append(beam.context)
            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0),
                     torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0) 
            
            coverage_t_1 = None
            all_coverage = []
            for beam in beams:
                all_coverage.append(beam.coverage)
            coverage_t_1 = torch.stack(all_coverage, 0)
            
            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1, z,
                                                                                    enc_padding_mask,  c_t_1, 
                                                                                    extra_zeros, enc_input_ext,
                                                                                    coverage_t_1, steps)
            log_probs = torch.log(final_dist)
            topk_logp, topk_ids = torch.topk(log_probs, self.beam_size*2)
            
            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze(0)
            dec_c = dec_c.squeeze(0)
            
            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i])

                for j in range(self.beam_size*2):
                    new_beam = h.extend(token=topk_ids[i,j].item(),
                                        log_prob=topk_logp[i,j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
                    all_beams.append(new_beam)
            
            beams = []
            for h in self.sort_beams(all_beams):
                beams.append(h)
                if len(beams) == self.beam_size: 
                    break
            
            for h in beams:
                if h.latest_token == self.eos:
                    results.append(h)
                else:
                    continue
            steps += 1
            
        if len(results) == 0:
            results = beams
        beams_sorted = self.sort_beams(results)
        result = {}
        for i, beams in enumerate(beams_sorted):
            sent = self.dec_ids_to_tokens(beams.tokens[1:], batch_oovs[0])
            result[i] = sent
            print(' '.join(sent))
        return result
        
    def reparameterize_std(self, mu, logsigma, scaler):
        # set random seed
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
            
        std = logsigma.mul(0.5) 
        std = std.exp_() * scaler
        if self.using_cuda:
            eps = torch.cuda.FloatTensor(mu.size()).normal_()
        else:
            eps = torch.FloatTensor(mu.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def std_z_list(self, z_list):
        z = torch.cat(z_list, dim=0) # 101 x L x K
        z = torch.mean(z.sum(dim=2), dim=1) # B 
        return torch.std(z).item()
    
    def select_random_decoded(self, output_ids_dict, topn=5):
        all_sent_list = []
        for sid, decoded_words in output_ids_dict.items():
            all_sent_list.append(' '.join(decoded_words))
        uni_sents = list(set(all_sent_list))
        if len(uni_sents) < topn:
            num_samples = len(uni_sents)
        else:
            num_samples = topn
        topn_sents = random.sample(uni_sents, num_samples)
        return topn_sents
    
    def select_best_decoded(self, ref_words, output_ids_dict, topn=5):
        all_sent_dict = {}
        for sid, decoded_words in output_ids_dict.items():
            sent_score = self.rouge_1(decoded_words, ref_words)
            all_sent_dict[' '.join(decoded_words)] = sent_score
        uni_sents = list(set(all_sent_dict.keys()))
        if len(uni_sents) < topn:
            num_samples = len(uni_sents)
        else:
            num_samples = topn
        topn_largest = nlargest(num_samples, all_sent_dict, key=all_sent_dict.get)
        uni_score = len(uni_sents)/len(output_ids_dict.keys())
        return topn_largest, uni_score
    
    def count_match(self, ref, dec):
        counts = 0.
        for d_word in dec:
            if d_word in ref:
                counts += 1
        return counts
    
    def rouge_1(self, decoded_words, ref_words):
        if len(ref_words) == 0:
            recall = 0.
        else:
            recall = self.count_match(ref_words, decoded_words)/len(ref_words)
        if len(decoded_words) == 0:
            precision = 0.
        else:
            precision = self.count_match(ref_words, decoded_words)/len(decoded_words)
        if recall+precision == 0:
            f1_score = 0.
        else:
            f1_score = 2*recall*precision/(recall+precision)
        return f1_score
    
    def get_ref_list(self, batch, i, args):
        ref_list = []
        if args.data_file in ["../data/GYAFC/fr", "../data/GYAFC/em"]:
            for trg_batch in [batch.trg, batch.trg2, batch.trg3, batch.trg4]:
                target_out = trg_batch.transpose(0,1)[i] # L 
                ref_ids = [t.item() for t in target_out]
                ref_words = self.ids_to_tokens(ref_ids)
                ref_list.append(' '.join(ref_words))
        else:
            target_out = batch.trg.transpose(0,1)[i] # L 
            ref_ids = [t.item() for t in target_out]
            ref_words = self.ids_to_tokens(ref_ids)
            ref_list.append(' '.join(ref_words))
        return ref_list 
    
    def decode_beam(self, test_iter, txtfield, args, num_samples=3000):
        start = time.time()
        counter = 0
        bleu_scores = []
        uni_scores = []
        dec_1_list = []
        best_5_list = []
        rand_5_list = []
        ref_list = []
        input_list = []
        
        for b, batch in enumerate(test_iter):
            with torch.no_grad():
                for i in range(batch.src.size(1)):
                    output_ids_dict = self.beam_search(batch, i)
                    
                    input_ids = [t.item() for t in batch.src[:,i]]
                    input_words = self.ids_to_tokens(input_ids)
                    
                    cur_ref_list = self.get_ref_list(batch, i, args)
                    ref_words = cur_ref_list[0].split()
                    
                    decoded_words = output_ids_dict[0]
                    topn_decoded_sents, uni_score = self.select_best_decoded(ref_words, output_ids_dict, args.topn)
                    randn_decoded_sents = self.select_random_decoded(output_ids_dict, args.topn)
                    uni_scores.append(uni_score)
                    
                    print('Input words: '+' '.join(input_words))
                    print('Ref words: '+' '.join(ref_words))
                    print('Pred words: '+' '.join(decoded_words))
                    print('Percent of unique sentences: '+str(uni_score))
                    print()
                    
                    dec_1_list.append(' '.join(decoded_words))
                    best_5_list.append('\t'.join(topn_decoded_sents))
                    rand_5_list.append('\t'.join(randn_decoded_sents))
                    ref_list.append('\t'.join(cur_ref_list))
                    input_list.append(' '.join(input_words))
                    sent_bleu = nltk.translate.bleu_score.sentence_bleu([ref_words], decoded_words)
                    bleu_scores.append(sent_bleu)
                    
                    counter += 1
                    if counter % 1000 == 0:
                        print('%d example in %d sec'%(counter, time.time() - start))
                        start = time.time()
            if counter > num_samples: break 
        
        avg_uni = round(np.mean(uni_scores), 3)
        print('Average BLEU score:', np.mean(bleu_scores))
        print('Average Unique sentences:', avg_uni)
        out = pd.DataFrame(columns=['source', 'ref', 'decode_1', 'bleu_1', 'best_5', 'rand_5'])
        out['source']=input_list
        out['ref']=ref_list
        out['decode_1']=dec_1_list
        out['bleu_1']=bleu_scores
        out['best_5']=best_5_list
        out['rand_5']=rand_5_list
        
        model_path = args.model_file
        model_name = model_path.split('/')[-1]
        data_file = args.data_file.split('/')[-1]
        out_path = 'saved_models/logs_'+data_file+'/output/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        mode = 'bs'
        out_file = out_path+model_name[:-3]+'_'+mode+'_uni_'+str(avg_uni)+'.csv'
        out.to_csv(out_file, index=False, sep='\t')
        

    def decode_sample(self, test_iter, txtfield, args, num_samples=3000):
        start = time.time()
        counter = 0
        bleu_scores = []
        uni_scores = []
        dec_1_list = []
        best_5_list = []
        rand_5_list = []
        ref_list = []
        input_list = []
        
        for b, batch in enumerate(test_iter):
            with torch.no_grad():
                for i in range(batch.src.size(1)):
                    output_ids_dict = self.sample_beam_search(batch, i)
                    
                    input_ids = [t.item() for t in batch.src[:,i]]
                    input_words = self.ids_to_tokens(input_ids)
                    
                    cur_ref_list = self.get_ref_list(batch, i, args)
                    ref_words = cur_ref_list[0].split()
                    
                    decoded_words = output_ids_dict[0]
                    topn_decoded_sents, uni_score = self.select_best_decoded(ref_words, output_ids_dict, args.topn)
                    randn_decoded_sents = self.select_random_decoded(output_ids_dict, args.topn)
                    uni_scores.append(uni_score)
                    
                    print('Input words: '+' '.join(input_words))
                    print('Ref words: '+' '.join(ref_words))
                    print('Pred words: '+' '.join(decoded_words))
                    print('Percent of unique sentences: '+str(uni_score))
                    print()
                    
                    dec_1_list.append(' '.join(decoded_words))
                    best_5_list.append('\t'.join(topn_decoded_sents))
                    rand_5_list.append('\t'.join(randn_decoded_sents))
                    ref_list.append('\t'.join(cur_ref_list))
                    input_list.append(' '.join(input_words))
                    sent_bleu = nltk.translate.bleu_score.sentence_bleu([ref_words], decoded_words)
                    bleu_scores.append(sent_bleu)
                    
                    counter += 1
                    if counter % 1000 == 0:
                        print('%d example in %d sec'%(counter, time.time() - start))
                        start = time.time()
            if counter > num_samples: break 
        
        avg_uni = round(np.mean(uni_scores), 3)
        print('Average BLEU score:', np.mean(bleu_scores))
        print('Average Unique sentences:', avg_uni)
        out = pd.DataFrame(columns=['source', 'ref', 'decode_1', 'bleu_1', 'best_5', 'rand_5'])
        out['source']=input_list
        out['ref']=ref_list
        out['decode_1']=dec_1_list
        out['bleu_1']=bleu_scores
        out['best_5']=best_5_list
        out['rand_5']=rand_5_list
        
        model_path = args.model_file
        model_name = model_path.split('/')[-1]
        data_file = args.data_file.split('/')[-1]
        out_path = 'saved_models/logs_'+data_file+'/output/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        mode = 'sample_bs'
        out_file = out_path+model_name[:-3]+'_'+mode+'_uni_'+str(avg_uni)+'.csv'
        out.to_csv(out_file, index=False, sep='\t')