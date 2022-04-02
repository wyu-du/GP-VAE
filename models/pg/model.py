import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from copynet.utils import get_input_from_batch, get_output_from_batch, log_Normal_diag

#torch.manual_seed(123)
#if torch.cuda.is_available():
#    torch.cuda.manual_seed_all(123)


def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-0.02, 0.02)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    linear.weight.data.normal_(std=1e-4)
    if linear.bias is not None:
        linear.bias.data.normal_(std=1e-4)

def init_wt_normal(wt):
    wt.data.normal_(std=1e-4)

def init_wt_unif(wt):
    wt.data.uniform_(-0.02, 0.02)
    


class Copynet(nn.Module):
    def __init__(self, args, txtfield):
        super(Copynet, self).__init__()
        # ---------------------------------
        # Configuration
        self.args = args
        self.txtfield = txtfield
        self.pad = txtfield.vocab.stoi[txtfield.pad_token] # index of <pad>
        self.vocab_size = args.vocab_size
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.latent_size = args.latent_size
        self.using_cuda = args.using_cuda
        
        # ---------------------------------
        # Model Arch
        # Encoder
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        init_wt_normal(self.embedding.weight)
        self.bi_lstm = nn.LSTM(self.embed_size, self.hidden_size, num_layers=1,
                            batch_first=True, bidirectional=True)
        init_lstm_wt(self.bi_lstm)
        self.hidden2latent = nn.Linear(self.hidden_size*2, self.latent_size)
        init_linear_wt(self.hidden2latent)
        self.latent2hidden = nn.Linear(self.latent_size, self.hidden_size*2)
        init_linear_wt(self.latent2hidden)
        self.reduce_h = nn.Linear(self.hidden_size*2, self.hidden_size)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(self.hidden_size*2, self.hidden_size)
        init_linear_wt(self.reduce_c)
        
        # Attention
        self.W_h = nn.Linear(self.hidden_size*2, self.hidden_size*2, bias=False)
        self.W_c = nn.Linear(1, self.hidden_size*2, bias=False)
        self.decode_proj = nn.Linear(self.hidden_size*2, self.hidden_size*2)
        self.v = nn.Linear(self.hidden_size*2, 1, bias=False)
        
        # Decoder
        self.x_context = nn.Linear(self.hidden_size*2 + self.embed_size, self.embed_size)
        self.uni_lstm = nn.LSTM(self.embed_size, self.hidden_size, num_layers=1,
                            batch_first=True, bidirectional=False)
        init_lstm_wt(self.uni_lstm)
        self.p_gen_linear = nn.Linear(self.hidden_size*4 + self.embed_size, 1)
        self.out1 = nn.Linear(self.hidden_size*3, self.hidden_size)
        self.out2 = nn.Linear(self.hidden_size, self.vocab_size)
        init_linear_wt(self.out2)
        # ---------------------------------
        
    def forward(self, batch, test=False):
        sorted_input, sorted_lengths, enc_padding_mask, sorted_idx, \
        enc_input_ext, extra_zeros, c_t_1, coverage, batch_oovs = \
                get_input_from_batch(batch, self.txtfield, self.args)
        dec_input, dec_target, dec_padding_mask = \
                get_output_from_batch(batch, batch_oovs, self.txtfield, self.args)
        
        enc_outputs, enc_hidden = self.encoder(sorted_input, sorted_lengths, sorted_idx)
        s_t_1 = self.reduce_state(enc_hidden, sorted_idx)
        
        z = self.hidden2latent(enc_outputs) # B x L x K
        
        nll_list = []
        for step in range(dec_input.size(1)):
            y_t_1 = dec_input[:, step]
            target = dec_target[:, step]
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.decoder(y_t_1, s_t_1, z, 
                                                                                     enc_padding_mask, c_t_1, 
                                                                                     extra_zeros, enc_input_ext, 
                                                                                     coverage, step)
            
            # Compute the NLL
            probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_nll = -torch.log(probs + 1e-12)
            step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
            step_nll = step_nll + 1.0 * step_coverage_loss
            coverage = next_coverage
            nll_list.append(step_nll)
        batch_nll = torch.stack(nll_list, 1) # B x L
        nll = torch.sum(batch_nll * dec_padding_mask, dim=1)  # B
        nll = nll / dec_padding_mask.sum(dim=1)
        nll = torch.mean(nll)
        
        kld = self.compute_kld()
        return nll, kld
        
    def encoder(self, sorted_input, sorted_lengths, sorted_idx):
        """
        enc_input - B x L
        """
        emb = self.embedding(sorted_input)
        packed = pack_padded_sequence(emb, sorted_lengths, batch_first=True)
        output, hidden = self.bi_lstm(packed) # hidden = ((2 x B x H), (2 x B x H))
        enc_outputs, _ = pad_packed_sequence(output, batch_first=True) # B x L x 2H
        
        _, reversed_idx = torch.sort(sorted_idx)
        enc_outputs = enc_outputs[reversed_idx]
        return enc_outputs, hidden
    
    def reduce_state(self, hidden, sorted_idx):
        h, c = hidden
        h_in = h.transpose(0, 1).contiguous().view(-1, self.hidden_size*2) # B x 2H
        reduced_h = F.relu(self.reduce_h(h_in))  # B x H
        
        c_in = c.transpose(0, 1).contiguous().view(-1, self.hidden_size*2) # B x 2H
        reduced_c = F.relu(self.reduce_c(c_in))  # B x H
        
        _, reversed_idx = torch.sort(sorted_idx)
        reduced_h = reduced_h[reversed_idx]
        reduced_c = reduced_c[reversed_idx]
        return (reduced_h.unsqueeze(0), reduced_c.unsqueeze(0)) # 1 x B x H
    
    def attention(self, s_t_hat, enc_outputs, enc_padding_mask, coverage):
        b, l, h = list(enc_outputs.size())
        
        enc_fea = enc_outputs.contiguous().view(-1, h) # BL x 2H
        enc_fea = self.W_h(enc_fea) # BL x 2H
        
        dec_fea = self.decode_proj(s_t_hat) # B x 2H
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, l, h).contiguous() # B x L x 2H
        dec_fea_expanded = dec_fea_expanded.view(-1, h) # BL x 2H
        
        attn_fea = enc_fea + dec_fea_expanded # BL x 2H
        
        coverage_input = coverage.contiguous().view(-1, 1) # BL x 1
        coverage_fea = self.W_c(coverage_input) # BL x 2H
        attn_fea = attn_fea + coverage_fea
        
        e = torch.tanh(attn_fea) # BL x 2H
        scores = self.v(e) # BL x 1
        scores = scores.view(-1, l) # B x L
        
        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask # B x L
        attn_dist = attn_dist_.unsqueeze(1)  # B x 1 x L
        
        c_t = torch.bmm(attn_dist, enc_outputs) # B x 1 x 2H
        c_t = c_t.squeeze(1) # B x 2H
        
        attn_dist = attn_dist.squeeze(1) # B x L
        coverage = coverage.contiguous().view(-1, l) # B x L
        coverage = coverage + attn_dist
        return c_t, attn_dist, coverage
    
    def decoder(self, y_t_1, s_t_1, z, enc_padding_mask, 
                c_t_1, extra_zeros, enc_input_ext, coverage, step):
        z_enc_outputs = self.latent2hidden(z) # B x L x 2H
        if step == 0:
            h_dec, c_dec = s_t_1
            s_t_hat = torch.cat((h_dec.squeeze(0), c_dec.squeeze(0)),1) # B x 2H
            c_t, _, coverage = self.attention(s_t_hat, z_enc_outputs, enc_padding_mask, coverage)
        
        y_t_1_emb = self.embedding(y_t_1) # B x E
        x = self.x_context(torch.cat((c_t_1, y_t_1_emb), 1)) # B x E
        lstm_out, s_t = self.uni_lstm(x.unsqueeze(1), s_t_1)
        
        h_dec, c_dec = s_t
        s_t_hat = torch.cat((h_dec.squeeze(0), c_dec.squeeze(0)),1) # B x 2H
        c_t, attn_dist, coverage = self.attention(s_t_hat, z_enc_outputs, enc_padding_mask, coverage)
        
        output = torch.cat((lstm_out.squeeze(1), c_t), 1)
        output = self.out1(output) # B x H
        output = self.out2(output) # B x V
        vocab_dist = F.softmax(output, dim=1)
        
        p_gen = None
        p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (4H + E)
        p_gen = self.p_gen_linear(p_gen_input) # B x 1
        p_gen = torch.sigmoid(p_gen)  
        vocab_dist_ = p_gen * vocab_dist # B x V
        attn_dist_ = (1 - p_gen) * attn_dist # B x L
        if extra_zeros is not None:
            vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)
        final_dist = vocab_dist_.scatter_add(1, enc_input_ext, attn_dist_)
        return final_dist, s_t, c_t, attn_dist, p_gen, coverage
        
    def compute_kld(self):
        zero = torch.zeros(1)
        if self.using_cuda: zero = zero.cuda()
        return zero
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.using_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)



class Normal(Copynet):
    def __init__(self, args, txtfield):
        super(Normal, self).__init__(args, txtfield)
        # ---------------------------------
        # Model Arch
        # encoder
        self.mean = nn.Linear(self.hidden_size*2, self.latent_size)
        init_linear_wt(self.mean)
        self.logvar = nn.Linear(self.hidden_size*2, self.latent_size)
        init_linear_wt(self.logvar)
        # ---------------------------------
        
    def forward(self, batch, test=False):
        sorted_input, sorted_lengths, enc_padding_mask, sorted_idx, \
        enc_input_ext, extra_zeros, c_t_1, coverage, batch_oovs = \
                get_input_from_batch(batch, self.txtfield, self.args)
        dec_input, dec_target, dec_padding_mask = \
                get_output_from_batch(batch, batch_oovs, self.txtfield, self.args)
        
        enc_outputs, enc_hidden = self.encoder(sorted_input, sorted_lengths, sorted_idx)
        s_t_1 = self.reduce_state(enc_hidden, sorted_idx)
        
        mean = self.mean(enc_outputs) # B x L x K
        logvar = self.logvar(enc_outputs) # B x L x K
        if test:
            z = mean
        else:
            z = self.reparameterize(mean, logvar)
        
        nll_list = []
        for step in range(dec_input.size(1)):
            y_t_1 = dec_input[:, step]
            target = dec_target[:, step]
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.decoder(y_t_1, s_t_1, z, 
                                                                                     enc_padding_mask, c_t_1, 
                                                                                     extra_zeros, enc_input_ext, 
                                                                                     coverage, step)
            
            # Compute the NLL
            probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_nll = -torch.log(probs + 1e-12)
            step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
            step_nll = step_nll + 1.0 * step_coverage_loss
            coverage = next_coverage
            nll_list.append(step_nll)
        batch_nll = torch.stack(nll_list, 1) # B x T
        nll = torch.sum(batch_nll * dec_padding_mask, dim=1)  # B
        nll = nll / dec_padding_mask.sum(dim=1)
        nll = torch.mean(nll)
        
        kld = self.compute_kld(z, mean, logvar) # B x L
        kld = torch.sum(kld * enc_padding_mask, dim=1)  # B
        kld = torch.mean(kld)
        return nll, kld
        
    def compute_kld(self, z, mean, logvar):
        k = mean.size(2)
        log_det = - logvar.sum(dim=2) # B x L
        trace = torch.sum(torch.exp(logvar), dim=2)
        mean = torch.sum(torch.pow(mean, 2), dim=2)
        kld = log_det - k + trace + mean
        kld = 0.5 * kld # B x L
        return kld
    
    
    
class MoG(Copynet):
    def __init__(self, args, txtfield):
        super(MoG, self).__init__(args, txtfield)
        # ---------------------------------
        # Configuration
        self.components = args.components
        
        # ---------------------------------
        # Model Arch
        # encoder
        self.mean = nn.Linear(self.hidden_size*2, self.latent_size)
        init_linear_wt(self.mean)
        self.logvar = nn.Linear(self.hidden_size*2, self.latent_size)
        init_linear_wt(self.logvar)
        
        # GMM prior
        prior_means = nn.Parameter(torch.randn((1, self.components, self.latent_size)), 
                                   requires_grad=True) # 1 x C x K
        self.register_parameter('prior_means', prior_means)
        prior_logvar = nn.Parameter(torch.randn((1, self.components, self.latent_size)), 
                                      requires_grad=True) # 1 x C x K
        self.register_parameter('prior_logvar', prior_logvar)
        # ---------------------------------
        
    def forward(self, batch, test=False):
        sorted_input, sorted_lengths, enc_padding_mask, sorted_idx, \
        enc_input_ext, extra_zeros, c_t_1, coverage, batch_oovs = \
                get_input_from_batch(batch, self.txtfield, self.args)
        dec_input, dec_target, dec_padding_mask = \
                get_output_from_batch(batch, batch_oovs, self.txtfield, self.args)
        
        enc_outputs, enc_hidden = self.encoder(sorted_input, sorted_lengths, sorted_idx)
        s_t_1 = self.reduce_state(enc_hidden, sorted_idx)
        
        mean = self.mean(enc_outputs) # B x L x K
        logvar = self.logvar(enc_outputs) # B x L x K
        if test:
            z = mean
        else:
            z = self.reparameterize(mean, logvar)
        
        nll_list = []
        for step in range(dec_input.size(1)):
            y_t_1 = dec_input[:, step]
            target = dec_target[:, step]
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.decoder(y_t_1, s_t_1, z, 
                                                                                     enc_padding_mask, c_t_1, 
                                                                                     extra_zeros, enc_input_ext, 
                                                                                     coverage, step)
            
            # Compute the NLL
            probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_nll = -torch.log(probs + 1e-12)
            step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
            step_nll = step_nll + 1.0 * step_coverage_loss
            coverage = next_coverage
            nll_list.append(step_nll)
        batch_nll = torch.stack(nll_list, 1) # B x L
        nll = torch.sum(batch_nll * dec_padding_mask, dim=1)  # B
        nll = nll / dec_padding_mask.sum(dim=1)
        nll = torch.mean(nll)
        
        kld = self.compute_kld(z, mean, logvar) # B x L
        kld = torch.sum(kld * enc_padding_mask, dim=1)  # B
        kld = torch.mean(kld)
        return nll, kld
        
    def compute_kld(self, z, mean, logvar):
        log_q_z = log_Normal_diag(z, mean, logvar, dim=2)
        log_p_z = self.log_p_z(z)
        
        kld = log_q_z - log_p_z # B x L
        return kld
    
    def log_p_z(self, z):
        if self.using_cuda:
            C = torch.cuda.FloatTensor([self.components])
        else:
            C = torch.FloatTensor([self.components])
        
        b, l, h = list(z.size())
        z_expand = z.unsqueeze(2).expand(b, l, self.components, h).contiguous() # B x L x C x K
        a = log_Normal_diag(z_expand, self.prior_means.unsqueeze(0), self.prior_logvar.unsqueeze(0), dim=3) - torch.log(C)  # B x L x C
        a_max, _ = torch.max(a, 2)  # B x L

        # calculte log-sum-exp
        log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(2)), 2))  # B x L
        return log_prior
    
    

class Vamp(Copynet):
    def __init__(self, args, txtfield):
        super(Vamp, self).__init__(args, txtfield)
        # ---------------------------------
        # Configuration
        self.components = args.components
        
        # ---------------------------------
        # Model Arch
        # encoder
        self.mean = nn.Linear(self.hidden_size*2, self.latent_size)
        init_linear_wt(self.mean)
        self.logvar = nn.Linear(self.hidden_size*2, self.latent_size)
        init_linear_wt(self.logvar)
        
        # pseudo-inputs for Vamp prior
        self.pseudo_inputs = torch.ones((self.components, 1), dtype=torch.long) * self.pad
        if self.using_cuda: self.pseudo_inputs = self.pseudo_inputs.cuda()
        self.pseudo_embed = nn.Linear(self.embed_size, self.embed_size)
        init_linear_wt(self.pseudo_embed)
        # ---------------------------------
        
    def forward(self, batch, test=False):
        sorted_input, sorted_lengths, enc_padding_mask, sorted_idx, \
        enc_input_ext, extra_zeros, c_t_1, coverage, batch_oovs = \
                get_input_from_batch(batch, self.txtfield, self.args)
        dec_input, dec_target, dec_padding_mask = \
                get_output_from_batch(batch, batch_oovs, self.txtfield, self.args)
        
        enc_outputs, enc_hidden = self.encoder(sorted_input, sorted_lengths, sorted_idx)
        s_t_1 = self.reduce_state(enc_hidden, sorted_idx)
        
        mean = self.mean(enc_outputs) # B x L x K
        logvar = self.logvar(enc_outputs) # B x L x K
        if test:
            z = mean
        else:
            z = self.reparameterize(mean, logvar)
        
        nll_list = []
        for step in range(dec_input.size(1)):
            y_t_1 = dec_input[:, step]
            target = dec_target[:, step]
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.decoder(y_t_1, s_t_1, z, 
                                                                                     enc_padding_mask, c_t_1, 
                                                                                     extra_zeros, enc_input_ext, 
                                                                                     coverage, step)
            
            # Compute the NLL
            probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_nll = -torch.log(probs + 1e-12)
            step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
            step_nll = step_nll + 1.0 * step_coverage_loss
            coverage = next_coverage
            nll_list.append(step_nll)
        batch_nll = torch.stack(nll_list, 1) # B x L
        nll = torch.sum(batch_nll * dec_padding_mask, dim=1)  # B
        nll = nll / dec_padding_mask.sum(dim=1)
        nll = torch.mean(nll)
        
        kld = self.compute_kld(z, mean, logvar) # B x L
        kld = torch.sum(kld * enc_padding_mask, dim=1)  # B
        kld = torch.mean(kld)
        return nll, kld
        
    def compute_kld(self, z, mean, logvar):
        log_q_z = log_Normal_diag(z, mean, logvar, dim=2)
        log_p_z = self.log_p_z(z)
        
        kld = log_q_z - log_p_z # B x L
        return kld
    
    def log_p_z(self, z):
        if self.using_cuda:
            C = torch.cuda.FloatTensor([self.components])
        else:
            C = torch.FloatTensor([self.components])
            
        embed = self.pseudo_embed(self.embedding(self.pseudo_inputs)) # C x 1 x E
        pseudo_enc_outputs, _ = self.bi_lstm(embed) # C x 1 x 2H
        pseudo_means = self.mean(pseudo_enc_outputs) # C x 1 x 2H
        pseudo_logvar = self.logvar(pseudo_enc_outputs)
        pseudo_means = pseudo_means.transpose(0, 1).unsqueeze(0) # 1 x 1 x C x 2H
        pseudo_logvar = pseudo_logvar.transpose(0, 1).unsqueeze(0) # 1 x 1 x C x 2H
        
        b, l, h = list(z.size())
        z_expand = z.unsqueeze(2).expand(b, l, self.components, h).contiguous() # B x L x C x 2H
        a = log_Normal_diag(z_expand, pseudo_means, pseudo_logvar, dim=3) - torch.log(C)  # B x L x C
        a_max, _ = torch.max(a, 2)  # B x L

        # calculte log-sum-exp
        log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(2)), 2))  # B x L
        return log_prior



class GP_Full(Copynet):
    def __init__(self, args, txtfield):
        super(GP_Full, self).__init__(args, txtfield)
        # ---------------------------------
        # Model Arch
        # encoder
        self.mean = nn.Linear(self.hidden_size*2, self.latent_size)
        init_linear_wt(self.mean)
        self.logvar = nn.Linear(self.hidden_size*2, self.latent_size)
        init_linear_wt(self.logvar)
        self.enc_latent = nn.Linear(self.hidden_size*2, self.latent_size)
        init_linear_wt(self.enc_latent)
        
        # parameters for kernel
        kernel_v = nn.Parameter(torch.ones(1)*args.kernel_v, requires_grad=False)
        self.register_parameter('kernel_v', kernel_v)
        kernel_r = nn.Parameter(torch.ones(1)*args.kernel_r, requires_grad=False)
        self.register_parameter('kernel_r', kernel_r)
        # ---------------------------------
        
    def forward(self, batch, test=False):
        sorted_input, sorted_lengths, enc_padding_mask, sorted_idx, \
        enc_input_ext, extra_zeros, c_t_1, coverage, batch_oovs = \
                get_input_from_batch(batch, self.txtfield, self.args)
        dec_input, dec_target, dec_padding_mask = \
                get_output_from_batch(batch, batch_oovs, self.txtfield, self.args)
        
        enc_outputs, enc_hidden = self.encoder(sorted_input, sorted_lengths, sorted_idx)
        s_t_1 = self.reduce_state(enc_hidden, sorted_idx)
        
        mean = self.mean(enc_outputs) # B x L x K
        logvar = self.logvar(enc_outputs) # B x L x K
        if test:
            z = mean
        else:
            z = self.reparameterize(mean, logvar)
        
        nll_list = []
        for step in range(dec_input.size(1)):
            y_t_1 = dec_input[:, step]
            target = dec_target[:, step]
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.decoder(y_t_1, s_t_1, z, 
                                                                                     enc_padding_mask, c_t_1, 
                                                                                     extra_zeros, enc_input_ext, 
                                                                                     coverage, step)
            
            # Compute the NLL
            probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_nll = -torch.log(probs + 1e-12)
            step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
            step_nll = step_nll + 1.0 * step_coverage_loss
            coverage = next_coverage
            nll_list.append(step_nll)
        batch_nll = torch.stack(nll_list, 1) # B x T
        nll = torch.sum(batch_nll * dec_padding_mask, dim=1)  # B
        nll = nll / dec_padding_mask.sum(dim=1)
        nll = torch.mean(nll)
        
        p_mean, p_var = self.compute_prior(enc_outputs)
        q_mean, q_var = self.compute_posterior(mean, logvar)
        kld = self.compute_kld(p_mean, p_var, q_mean, q_var) # B
        kld = torch.mean(kld)
        return nll, kld
    
    def compute_prior(self, enc_outputs):
        """
        GP prior p(z|x) = N(mu(x), K(x, x'))
        
        enc_outputs - B x L x 2H
        """
        b, l, h = list(enc_outputs.size())
        mean = enc_outputs.sum(dim=2) # B x L
        var = torch.zeros((b, l, l), requires_grad=False) # B x L x L
        if self.using_cuda: var = var.cuda()
        for i in range(l):
            for j in range(l):
                var[:, i, j] = self.kernel_func(enc_outputs[:,i,:], enc_outputs[:,j,:])
        return mean, var
    
    def kernel_func(self, x, y):
        """
        x, y - B x 2H
        """
        cov_xy = self.kernel_v * torch.exp(-0.5 * torch.sum(torch.pow((x - y)/self.kernel_r, 2), dim=1))
        return cov_xy
    
    def compute_posterior(self, mean, logvar):
        """
        variational posterior q(z|x) = N(mu(x), f(x))
        
        mean, logvar - B x L x K
        """
        mean = mean.sum(dim=2) # B x L
        x_var = torch.exp(logvar).sum(dim=2) # B x L
        
        var_batch = []
        for b in range(mean.size(0)):
            identity_matrix = torch.eye(x_var.size(1))
            if self.using_cuda: identity_matrix = identity_matrix.cuda()
            var_batch.append(x_var[b]*identity_matrix)
        var = torch.stack(var_batch, dim=0) # B x L x L
        return mean, var
        
    def compute_kld(self, p_mean, p_var, q_mean, q_var):
        k = p_var.size(1)
        
        log_det = torch.logdet(p_var) - torch.logdet(q_var) 
        if torch.isnan(log_det).int().sum() > 0:
            if torch.isnan(q_var).int().sum() > 0:
                print('q_var has nan!!!')
                print(q_var)
        
        try:
            p_var_inv = torch.inverse(p_var) # B x L x L
            trace_batch = torch.matmul(p_var_inv, q_var) # B x L x L
            trace_list = [torch.trace(trace_batch[i]) for i in range(trace_batch.size(0))]
            trace = torch.stack(trace_list, dim=0) # B
            
            mean_diff = p_mean.unsqueeze(2) - q_mean.unsqueeze(2) # B x L x 1
            mean = torch.matmul(torch.matmul(mean_diff.transpose(1,2), p_var_inv), mean_diff) # B x K x K
            
            kld = log_det - k + trace + torch.mean(mean, dim=(1,2))
            kld = 0.5 * kld # B
        except:
            zeros = torch.zeros(p_mean.size(0))
            if self.using_cuda: zeros = zeros.cuda()
            kld = zeros
            print('zero kld!!!')
        return kld