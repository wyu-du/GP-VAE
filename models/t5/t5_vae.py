# -*- coding: utf-8 -*-

import json
import argparse
import time
import os
from utils import parse_data, batchify_data, make_batch_inputs, prepare_eval, test_ppl

from datasets import Dataset
from transformers import T5TokenizerFast, T5ForConditionalGeneration
# https://huggingface.co/transformers/_modules/transformers/models/t5/tokenization_t5_fast.html#T5TokenizerFast
# https://huggingface.co/transformers/_modules/transformers/models/t5/modeling_t5.html#T5ForConditionalGeneration
# https://huggingface.co/transformers/_modules/transformers/generation_utils.html#GenerationMixin.generate
# https://huggingface.co/blog/how-to-generate
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import Conv1D

# Device and fp16 management.
import torch
from torch import nn
from torch.nn import functional as F
from packaging import version

if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast


class AverageSelfAttention(nn.Module):
    def __init__(self, attention_size):
        super(AverageSelfAttention, self).__init__()
        w = torch.empty(attention_size)
        nn.init.normal_(w, std=0.02)
        self.attention_weights = nn.Parameter(w)
        self.softmax = nn.Softmax(dim=-1)
        self.non_linearity = F.gelu

    def forward(self, inputs, attention_mask=None):
        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.non_linearity(inputs.matmul(self.attention_weights))

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        if attention_mask is not None:
            scores = scores + attention_mask

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################
        scores = self.softmax(scores)

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations = weighted.sum(1).squeeze(1)

        return representations, scores


class Seq2SeqModel(T5ForConditionalGeneration):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, **kwargs)
        self.averageSelfAttention_post = AverageSelfAttention(config.d_model)
        self.mean_post = Conv1D(config.d_model, config.d_model)
        self.logvar_post = Conv1D(config.d_model, config.d_model)
        self.input_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def posterior(self, hidden_states, attention_mask):
        representations, _ = self.averageSelfAttention_post(hidden_states, attention_mask)
        post_mean = self.mean_post(representations)
        post_logvar = self.logvar_post(representations)
        return post_mean, post_logvar

    def reparameterize(self, mean, logvar, z=None):
        std = logvar.mul(0.5).exp() * self.scaler
        if z is None:
            z = torch.randn(std.size(), device=mean.device, dtype=mean.dtype)
        return z.mul(std) + mean

    def kl_loss(self, mean1, logvar1, mean2, logvar2):
        exponential = logvar1 - logvar2 - torch.pow(mean1 - mean2, 2) / logvar2.exp() - torch.exp(logvar1 - logvar2) + 1
        result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
        return result.mean()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        # added z code here
        posterior_mean, posterior_logvar = self.posterior(hidden_states, attention_mask)
        prior_mean = torch.zeros([hidden_states.size(0), posterior_mean.size(-1)]) \
            .to(posterior_mean.dtype).to(posterior_mean.device)
        prior_logvar = torch.zeros([hidden_states.size(0), posterior_logvar.size(-1)]) \
            .to(posterior_logvar.dtype).to(posterior_logvar.device)

        if self.from_mean:
            z = posterior_mean
        else:
            z =  self.reparameterize(posterior_mean, posterior_logvar)

        input_proj = self.input_proj(z).unsqueeze(1)  # B x 1 x K
        hidden_states = hidden_states + input_proj

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # kl_loss
            kl_loss = self.kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar)
            loss = loss + kl_loss

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            #        cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            #            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }


class Seq2SeqTrainer(Trainer):
    """Class to finetune a Seq2Seq model."""

    def __init__(
            self,
            num_beams=4,
            max_length=32,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_beams = num_beams
        self.max_length = max_length

    def compute_loss(self, model, inputs):
        """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
        outputs = model(input_ids=inputs['input_ids'],
                        # decoder_input_ids=inputs['labels'][:,:-1],
                        labels=inputs['labels'])
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.label_smoother is not None and "labels" in inputs:
            return self.label_smoother(outputs, inputs["labels"])
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            return outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
    Runs the model to either generate a sequence and/or compute the loss.
    """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        # Compute loss with labels first.
        with torch.no_grad():
            if self.args.fp16 and _use_native_amp:
                with autocast():
                    outputs = model(input_ids=inputs['input_ids'],
                                    # decoder_input_ids=inputs['labels'][:,:-1],
                                    labels=inputs['labels'])
            else:
                outputs = model(input_ids=inputs['input_ids'],
                                # decoder_input_ids=inputs['labels'][:,:-1],
                                labels=inputs['labels'])
            if has_labels:
                loss = outputs[0].mean().detach()
            else:
                loss = None
        # If we're only computing the conditional log-likelihood, return.
        if prediction_loss_only:
            return (loss, None, None)
        # Otherwise run model.generate() to get predictions.
        if isinstance(model, torch.nn.DataParallel):
            preds = model.module.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                num_beams=self.num_beams,
                max_length=self.max_length,
            )
        else:
            preds = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                num_beams=self.num_beams,
                max_length=self.max_length,
            )
        if len(preds) == 1:
            preds = preds[0]
        # Pad predictions if necessary so they can be concatenated across batches.
        if preds.shape[-1] < self.max_length:
            preds = torch.nn.functional.pad(
                preds, (0, self.max_length - preds.shape[-1]),
                mode='constant',
                value=self.tokenizer.pad_token_id
            )
        # Post-process labels.
        if has_labels:
            labels = inputs.get('labels')
        else:
            labels = None
        return (loss, preds, labels)


def train(args):
    # Load the dataset
    trn_df = parse_data(in_file=f'../../data/{args.dataset}/trn.tsv')
    val_df = parse_data(in_file=f'../../data/{args.dataset}/val.tsv')

    # Load the pre-trained model
    ckpt_path = None
    if args.task == 'train':
        ckpt_path = args.model_name
    else:
        ckpt_path = f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/checkpoint-{args.ckpt}"
        # update timestamp and create new path for ckpt
        args.timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    tokenizer = T5TokenizerFast.from_pretrained(ckpt_path)
    print(f"Vocab size: {len(tokenizer)}")

    train_data_tokenized = batchify_data(trn_df, tokenizer, args)
    valid_data_tokenized = batchify_data(val_df, tokenizer, args)

    model = Seq2SeqModel.from_pretrained(ckpt_path)
    model = model.to('cuda:0')
    model.from_mean = args.from_mean
    model.scaler = 1.0

    # Training Setup
    train_args = TrainingArguments(
        output_dir=f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}",
        do_train=True,
        do_eval=True,
        save_strategy="steps",
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=1000,
        logging_steps=100,
        # optimization args, the trainer uses the Adam optimizer
        # and has a linear warmup for the learning rate
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=1e-04,
        num_train_epochs=args.epochs,
        warmup_steps=0,
        lr_scheduler_type='constant',
        # misc args
        seed=42,
        save_total_limit=1,  # limit the total amount of checkpoints
        disable_tqdm=False,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        local_rank=args.local_rank
    )

    trainer = Seq2SeqTrainer(
        num_beams=args.beam_size,
        max_length=args.decoder_max_length,
        model=model,
        args=train_args,
        train_dataset=train_data_tokenized,
        eval_dataset=valid_data_tokenized,
        tokenizer=tokenizer,
    )

    # Now that we have the trainer set up, we can finetune.
    trainer.train()


def beam_generate_sentences(batch,
                            model, 
                            tokenizer,
                            args, 
                            device='cuda:0'):
    # Create batch inputs.
    features = make_batch_inputs(
      batch=batch, 
      tokenizer=tokenizer, 
      args=args, 
      device=device)
    # Generate with beam search.
    generated_ids = model.generate(
      input_ids=features['input_ids'], 
      attention_mask=features['attention_mask'],
      num_beams=args.beam_size,
      max_length=args.max_generation_length,
      num_return_sequences=1,
    )
    # Use model tokenizer to decode to text.
    generated_sentences = [
      tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
      for gen_ids in generated_ids
    ]
    print(generated_sentences)
    return ['\t'.join(generated_sentences)]


def sample_sentences(batch,
                     model, 
                     tokenizer,
                     args, 
                     device='cuda:0'):
    # Create batch inputs.
    features = make_batch_inputs(
      batch=batch, 
      tokenizer=tokenizer, 
      args=args, 
      device=device)
    
    generated_sentences = []
    for i in range(args.num_return_sequences):
        # Generate with beam search.
        generated_ids = model.generate(
          input_ids=features['input_ids'], 
          attention_mask=features['attention_mask'],
          num_beams=args.beam_size,
          max_length=args.max_generation_length,
          num_return_sequences=1,
        )
        # Use model tokenizer to decode to text.
        generated_sentences += [
          tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
          for gen_ids in generated_ids
        ]
    print(generated_sentences)
    return ['\t'.join(generated_sentences)]


def test(args):
    te_df = parse_data(in_file=f'../../data/{args.dataset}/tst.tsv')
    print('Data loaded!!!')

    # Load the model
    if args.timestamp == '0':
        tokenizer = T5TokenizerFast.from_pretrained(f"{args.model_name}")
    else:
        tokenizer = T5TokenizerFast.from_pretrained(
            f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/checkpoint-{args.ckpt}")
    print(f"Vocab size: {len(tokenizer)}")

    if args.timestamp == '0':
        model = Seq2SeqModel.from_pretrained(f"{args.model_name}")
    else:
        model = Seq2SeqModel.from_pretrained(
            f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/checkpoint-{args.ckpt}")
    model = model.to('cuda:0')
    model.from_mean = args.from_mean
    model.scaler = args.scaler
    print('Model loaded!!!')

    #    ppl = test_ppl(te_df, model, tokenizer, args)
    #    ppl = round(ppl, 2)
    #    print(f"Test ppl: {ppl}")

    # Make predictions
    if args.from_mean:
        test_output = Dataset.from_pandas(te_df).map(
            lambda batch: {'generated': beam_generate_sentences(
                batch,
                model, 
                tokenizer, 
                args,
                device='cuda:0')
            },
            batched=True, 
            batch_size=1,
        )
    else:
        test_output = Dataset.from_pandas(te_df).map(
            lambda batch: {'generated': sample_sentences(
                batch,
                model, 
                tokenizer, 
                args,
                device='cuda:0')
            },
            batched=True, 
            batch_size=1,
        )

    # prepare evaluation data
    ref_list, pred_list = prepare_eval(list(test_output))
    reference_dict = {
        "language": "en",
        "values": ref_list,
    }
    prediction_dict = {
        "language": "en",
        "values": pred_list,
    }

    if args.timestamp == '0':
        os.makedirs(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}")

    with open(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/refs.json", 'w') as f:
        f.write(json.dumps(reference_dict, indent=2))
    if args.from_mean:
        with open(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/outs_mean.json", 'w') as f:
            f.write(json.dumps(prediction_dict, indent=2))
    else:
        with open(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/outs.json", 'w') as f:
            f.write(json.dumps(prediction_dict, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-t', '--task', type=str, default="train",
                   help="specify the task to do: (train)ing, ft(finetune), (eval)uation")
    p.add_argument('-c', '--ckpt', type=str, default="193280",
                   help="Model checkpoint")
    p.add_argument('-time', '--timestamp', type=str, default='2021-02-14-04-57-04',
                   help="Model checkpoint")
    p.add_argument('-f', '--flag', type=str, default='vae',
                   help="Model checkpoint")
    p.add_argument('-d', '--dataset', type=str, default="GYAFC/em",
                   help="specify the dataset: GYAFC/em, GYAFC/fr")
    p.add_argument('--model_name', type=str, default="t5-base",
                   help="specify the model name: t5-base, facebook/blenderbot-400M-distill")
    p.add_argument('-s', '--scaler', type=float, default=1.0)
    p.add_argument('--from_mean', action='store_true', 
                   help="specify whether sample from mean during generation")
    p.add_argument('-bz', '--batch_size', type=int, default=16)
    p.add_argument('-e', '--epochs', type=int, default=10)
    p.add_argument('--encoder_max_length', type=int, default=50)
    p.add_argument('--decoder_max_length', type=int, default=50)
    p.add_argument('--max_generation_length', type=int, default=60)
    p.add_argument('--beam_size', type=int, default=10)
    p.add_argument('--num_return_sequences', type=int, default=10)
    p.add_argument('--local_rank', type=int, default=-1,
                   help="Multiple GPU training")
    args = p.parse_args()

    if args.task == 'train':
        args.timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        train(args)
    elif args.task == 'ft':
        train(args)
    else:
        test(args)
