import os
import torch
from torchtext.data import Field, TabularDataset, BucketIterator
import spacy
spacy_en = spacy.load('en')


def tokenize_fn(text):
    """ 
    Tokenization function
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def reader(suffix=".tsv", rpath="../data/twitter_url", batch_size=32):
    """
    """
    # Create a Dataset instance
    if rpath in ["../data/GYAFC/fr", "../data/GYAFC/em"]:
        TXT = Field(tokenize=tokenize_fn, init_token='<sos>', eos_token='<eos>', lower=True)
        fields = [("src", TXT), ("trg", TXT)]
        fields2 = [("src", TXT), ("trg", TXT), ("trg2", TXT), ("trg3", TXT), ("trg4", TXT)]
        trn_data = TabularDataset(os.path.join(rpath,'trn'+suffix), format="TSV", fields=fields, skip_header=False)
        val_data = TabularDataset(os.path.join(rpath,'val'+suffix), format="TSV", fields=fields, skip_header=False)
        tst_data = TabularDataset(os.path.join(rpath,'tst_all'+suffix), format="TSV", fields=fields2, skip_header=False)
    else:
        TXT = Field(tokenize=tokenize_fn, init_token='<sos>', eos_token='<eos>', lower=True)
        fields = [("src", TXT), ("trg", TXT)]
        trn_data = TabularDataset(os.path.join(rpath,'trn'+suffix), format="TSV", fields=fields, skip_header=False)
        val_data = TabularDataset(os.path.join(rpath,'val'+suffix), format="TSV", fields=fields, skip_header=False)
        tst_data = TabularDataset(os.path.join(rpath,'tst'+suffix), format="TSV", fields=fields, skip_header=False)
    
    # Build vocab using training data
    TXT.build_vocab(trn_data, min_freq=1, vectors="glove.6B.100d") # max_size=15000
    
    # Create iterator
    train_iter, val_iter, test_iter = BucketIterator.splits((trn_data, val_data, tst_data), 
                                                            batch_size=batch_size, 
                                                            sort_key=lambda x: len(x.src),
                                                            sort=True,
                                                            shuffle=False,
                                                            repeat=False)
    return train_iter, val_iter, test_iter, TXT


def kl_anneal_weight(step, args):
    """
    The logistic anneal function.
    """
    if args.kw == 0:
        if args.using_cuda:
            kl_weight = torch.cuda.FloatTensor([1.0])
        else:
            kl_weight = torch.FloatTensor([1.0])
    else:
        if args.using_cuda:
            step = torch.cuda.FloatTensor([-args.kw * (step - args.x0)])
        else:
            step = torch.FloatTensor([-args.kw * (step - args.x0)])
        kl_weight = 1/(1 + torch.exp(step))
    return kl_weight


def log_Normal_multivariate(x, mean, sigma, dim=None):
    x_mu = torch.sum(x - mean, dim=dim, keepdim=True) # B x L x 1
    x_mu_t = x_mu.transpose(1,2) # B x 1 x L
    det = torch.det(sigma) # B
    log_normal = -0.5 * (torch.log(det) + torch.matmul(torch.matmul(x_mu_t, torch.inverse(sigma)), x_mu).squeeze())
    return log_normal
    

def log_Normal_diag(x, mean, log_var, dim=None):
    log_normal = -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))
    return torch.sum(log_normal, dim)
    
    
def log_Normal_standard(x, dim=None):
    log_normal = -0.5 * torch.pow(x, 2)
    return torch.sum(log_normal, dim)
    
    
def get_output_from_batch(batch, batch_oovs, txtfield, args):
    dec_input = batch.trg[:-1]
    dec_input = dec_input.transpose(0,1)  # B x L-1
    dec_target = batch.trg[1:]
    dec_target = dec_target.transpose(0,1)  # B x L-1
    ones = torch.ones_like(dec_target)
    zeros = torch.zeros_like(dec_target)
    pad_id = txtfield.vocab.stoi[txtfield.pad_token]
    
    for b in range(dec_input.size(0)):
        for i in range(dec_input.size(1)):
            inp_wid = dec_input[b,i].item()
            if inp_wid > args.vocab_size:
                dec_input[b,i] = txtfield.vocab.stoi[txtfield.unk_token] # index of <unk>
    
    for b in range(dec_target.size(0)):
        for i in range(dec_target.size(1)):
            tar_wid = dec_target[b,i].item()
            if batch_oovs is not None:
                if tar_wid in batch_oovs[b]:
                    # Overwrite decoder target sequence so it uses the temp source OOV id
                    tmp_id = batch_oovs[b].index(tar_wid) + args.vocab_size
                    dec_target[b,i] = tmp_id
                
    if args.using_cuda: 
        dec_input = dec_input.cuda()
        dec_target = dec_target.cuda()
        ones = ones.cuda()
        zeros = zeros.cuda()
    mask = torch.where(dec_target == pad_id, zeros, ones)
    return dec_input, dec_target, mask
    
    
def get_input_from_batch(batch, txtfield, args):
    enc_input = batch.src.transpose(0,1)  # B x L
    max_len = enc_input.size(1)
    if max_len > args.max_len:
        max_len = args.max_len
    enc_input = enc_input[:, :max_len]
    c_t_1 = torch.zeros(enc_input.size(0), args.hidden_size*2) # B x H
    enc_input_ext = None
    extra_zeros = None
    coverage = None
    
    if args.using_cuda: 
        enc_input = enc_input.cuda()
        c_t_1 = c_t_1.cuda()
        
    enc_input, enc_input_ext, max_art_oovs, batch_oovs = get_extend_vocab(enc_input, txtfield, args)
    # max_art_oovs is the max over all the article oov list in the batch
    if max_art_oovs > 0:
        extra_zeros = torch.zeros((enc_input.size(0), max_art_oovs))
        if args.using_cuda: extra_zeros = extra_zeros.cuda()
            
    sorted_input, sorted_lengths, enc_padding_mask, sorted_idx = compute_seqence_length(enc_input, txtfield, args)
    max_len = sorted_input.size(1)
    enc_input_ext = enc_input_ext[:, :max_len]
    
    coverage = torch.zeros_like(enc_input, dtype=torch.float) # B x L
    coverage = coverage[:, :max_len]
    if args.using_cuda: coverage = coverage.cuda()
    return sorted_input, sorted_lengths, enc_padding_mask, sorted_idx, enc_input_ext, extra_zeros, c_t_1, coverage, batch_oovs


def get_extend_vocab(enc_input, txtfield, args):
    batch_ids = []
    batch_oovs = []
    max_art_oovs = 0
    for b in range(enc_input.size(0)):
        ids = []
        oovs = []
        for i in range(enc_input.size(1)):
            wid = enc_input[b,i].item()
            if wid >= args.vocab_size: 
                enc_input[b,i] = txtfield.vocab.stoi[txtfield.unk_token] # index of <unk>
                # if word is OOV, record the relative position of the word in current sentence.
                # e.g. for vocab_size = 5000, the first OOV has id = 5001, the second OOV has id = 5002
                if wid not in oovs:
                    oovs.append(wid)
                oov_num = oovs.index(wid)
                ids.append(args.vocab_size + oov_num)
            else:
                ids.append(wid)
        batch_ids.append(ids)
        batch_oovs.append(oovs)
        if len(oovs) > max_art_oovs:
            max_art_oovs = len(oovs)
    if args.using_cuda:
        enc_batch_extend_vocab = torch.cuda.LongTensor(batch_ids, device=torch.device('cuda'))
    else:
        enc_batch_extend_vocab = torch.LongTensor(batch_ids)
        
    return enc_input, enc_batch_extend_vocab, max_art_oovs, batch_oovs
    

def compute_seqence_length(enc_input, txtfield, args):
    """
    Compute the sequence length of the input sequence.
    
    input - B x L
    """
    ones = torch.ones_like(enc_input)
    zeros = torch.zeros_like(enc_input)
    if args.using_cuda: 
        ones = ones.cuda()
        zeros = zeros.cuda()
    pad = txtfield.vocab.stoi[txtfield.pad_token] # index of <pad>
    mask = torch.where(enc_input == pad, zeros, ones)
    seq_lens = mask.sum(dim=1) # B
    sorted_lengths, sorted_idx = torch.sort(seq_lens, descending=True)
    max_len = sorted_lengths[0].item()
    sorted_input = enc_input[sorted_idx, :max_len]
    enc_padding_mask = mask[:, :max_len]
    
    return sorted_input, sorted_lengths, enc_padding_mask, sorted_idx
    
    
def get_decode_input_from_batch(batch, txtfield, args, bid):
    enc_input = batch.src.transpose(0,1)[bid].unsqueeze(0)  # 1 x L
    max_len = enc_input.size(1)
    if max_len > args.max_len:
        max_len = args.max_len
    enc_input = enc_input[:, :max_len]
    enc_input = enc_input.expand(args.sample_num, enc_input.size(1)).contiguous() # B x L
    c_t_1 = torch.zeros(enc_input.size(0), args.hidden_size*2) # B x 2H
    enc_input_ext = None
    extra_zeros = None
    coverage = None
    
    if args.using_cuda: 
        enc_input = enc_input.cuda()
        c_t_1 = c_t_1.cuda()
    
    enc_input, enc_input_ext, max_art_oovs, batch_oovs = get_extend_vocab(enc_input, txtfield, args)
    
    # max_art_oovs is the max over all the article oov list in the batch
    if max_art_oovs > 0:
        extra_zeros = torch.zeros((enc_input.size(0), max_art_oovs))
        if args.using_cuda: extra_zeros = extra_zeros.cuda()
    
    sorted_input, sorted_lengths, enc_padding_mask, sorted_idx = compute_seqence_length(enc_input, txtfield, args)
    max_len = sorted_input.size(1)
    enc_input_ext = enc_input_ext[:, :max_len]
    
    coverage = torch.zeros_like(enc_input, dtype=torch.float) # B x L
    coverage = coverage[:, :max_len]
    if args.using_cuda: coverage = coverage.cuda()
    return sorted_input, sorted_lengths, enc_padding_mask, sorted_idx, enc_input_ext, extra_zeros, c_t_1, coverage, batch_oovs