# -*- coding: utf-8 -*-

import torch
import pandas as pd
from datasets import Dataset


def parse_data(in_file='../../data/GYAFC/em/trn.tsv'):
    with open(in_file, 'r') as f:
        data = f.read().split('\n')
        data.remove('')
    contexted = []
    for i, line in enumerate(data):
        source_txt = line.split('\t')[0]
        target_txt = line.split('\t')[1]
        row = (i, source_txt, target_txt)
        contexted.append(row)
    columns = ['id', 'source', 'target']
    data_df = pd.DataFrame.from_records(contexted, columns=columns)
    return data_df

# Specific to dataset.
def construct_input_for_batch(tokenizer, batch, args):
    """
    Function that takes a batch from a dataset and constructs the corresponding 
    input string.
    """
    source, target = [], []
    for inp, out in zip(batch['source'], batch['target']):
        source.append(inp.strip())
        target.append(out.strip())
    if batch['id'][0] == 0:
        print(source[0])
        print(target[0])
        print()
    return source, target

def make_batch_inputs(batch, tokenizer, args, device='cuda:0'):
  """
  Function that takes a batch from a dataset and transforms it 
  """
  # Concatenate the concept names for each example in the batch.
  input_lists, _ = construct_input_for_batch(tokenizer, batch, args)
  # Use the model's tokenizer to create the batch input_ids.
  batch_features = tokenizer(input_lists, padding=True, return_tensors='pt')
  # Move all inputs to the device.
  batch_features = dict([(k, v.to(device)) for k, v in batch_features.items()])
  return batch_features

def make_batch_data(batch, tokenizer, args, device='cuda:0'):
  """
  Function that takes a batch from a dataset and transforms it 
  """
  # Concatenate the concept names for each example in the batch.
  input_lists, label_list = construct_input_for_batch(tokenizer, batch, args)
  # Use the model's tokenizer to create the batch input_ids.
  batch_features = tokenizer(input_lists, padding=True, return_tensors='pt')
  batch_labels = tokenizer(label_list, padding=True, return_tensors='pt')
  # Move all inputs to the device.
  batch_features = dict([(k, v.to(device)) for k, v in batch_features.items()])
  batch_labels = dict([(k, v.to(device)) for k, v in batch_labels.items()])
  return batch_features, batch_labels

def batch_tokenize(dataset_batch, tokenizer, args):
  """
  Reuse the function defined above to construct the batch (source, target) and 
  run them through the tokenizer.
  """
  source, target = construct_input_for_batch(tokenizer, dataset_batch, args)
  res = {
          "input_ids": tokenizer(
              source,
              padding='max_length', 
              truncation=True,
              max_length=args.encoder_max_length
          )["input_ids"],
          "labels": tokenizer(
              target,
              padding='max_length', 
              truncation=True,
              max_length=args.decoder_max_length
          )["input_ids"],
  }
  return res

def batchify_data(df, tokenizer, args):
  dataset = Dataset.from_pandas(df)
  data_tokenized = dataset.map(
    lambda batch: batch_tokenize(batch, tokenizer, args),
    batched=True
  )
  return data_tokenized

def compute_loss(batch, model, tokenizer, args):
  batch_feature, batch_label = make_batch_data(batch, tokenizer, args)
  with torch.no_grad():
    outputs = model(input_ids=batch_feature['input_ids'],
                    labels=batch_label['input_ids'])
    eval_loss = outputs.loss.item()
  return [eval_loss] 

def test_ppl(val_df, model, tokenizer, args):
  loss_dict = Dataset.from_pandas(val_df).map(
    lambda batch: {'loss': compute_loss(batch, model, tokenizer, args)},
    batched=True,
    batch_size=1,
  )
  
  eval_loss = 0.
  nb_eval_steps = 0
  for item in list(loss_dict):
      eval_loss += item['loss']
      nb_eval_steps += 1
  eval_loss = eval_loss / nb_eval_steps
  ppl = torch.exp(torch.tensor(eval_loss))
  return ppl.item()

def prepare_eval(output_list):
    ref_list, pred_list = [], []
    for item in output_list:
        pred_list.append({"generated": item['generated']})
        ref_list.append({"target": [item['target']]})
    return ref_list, pred_list
