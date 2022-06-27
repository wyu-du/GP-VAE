import os
import logging
import datetime
import tensorflow as tf
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

import torch
from torch import autograd
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from utils import kl_anneal_weight



def train(model, optimizer, train_iter, val_iter, txtfield, args):
    """ Training with validation
    """
    # -----------------------------
    # create model ckpt path
    curr_time = datetime.datetime.now()
    time_stamp = "{}-{}-{}-{}-{}-{}".format(curr_time.year, curr_time.month, curr_time.day,
                                            curr_time.hour, curr_time.minute, curr_time.second)
    data_file = args.data_file.split('/')[-1]
    wpath = "logs_"+data_file
    fprefix = '{}_k{}_h{}_c{}_r{}_v{}_t{}'.format(args.model_type, args.latent_size, 
                                                  args.hidden_size, args.components, 
                                                  args.kernel_r, args.kernel_v,
                                                  time_stamp)
    if not os.path.isdir(wpath):
        os.makedirs(wpath)
    flog = os.path.join(wpath, fprefix+".log")
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %I:%M:%S %p',
                        handlers=[
                            logging.FileHandler(flog), # to file
                            logging.StreamHandler() # to stdout
                        ],
                        level=logging.INFO)
    logging.info("File name prefix: {}".format(fprefix))
    logging.info(args)
    logging.info(model)
    
    # -----------------------------
    # create tensorboard logs
    eval_log = 'tensorboard_logs_'+data_file+'/' + fprefix
    if not os.path.exists(eval_log):
        os.makedirs(eval_log)
    summary_writer = tf.summary.FileWriter(eval_log)
    
    # -----------------------------
    # start training
    global_step = 0
    early_stop = 0
    best_val_loss = 1000
    for e in range(1, args.epochs+1):
        for b, batch in enumerate(train_iter):
            model.train()
            optimizer.zero_grad()
            
            #with autograd.detect_anomaly():
            nll, kld = model(batch)
            kl_weight = kl_anneal_weight(global_step, args)
                
            if args.model_type in ['copynet']:
                loss = nll
            else:
                if kld == 0:
                    loss = nll
                    print('Skip current batch!!!')
                else:
                    loss = nll + kl_weight * kld
                    
            loss.backward()
            clip_grad_norm(model.parameters(), args.grad_clip) 
            optimizer.step()
            global_step += 1
            if global_step % args.print_step == 0:
                logging.info("[Epoch: %d] [Batch: %d/%d] [NLL: %.4f] [KLD: %.4f]" % 
                             (e, b+1, len(train_iter), nll, kld))
                # --------------------------------
                # Add nll, kld into tensorboard
                train_nll = tf.Summary()
                train_nll.value.add(tag='train_nll', simple_value=nll)
                summary_writer.add_summary(train_nll, global_step=global_step)
                summary_writer.flush()
                train_kld = tf.Summary()
                train_kld.value.add(tag='train_kld', simple_value=kld)
                summary_writer.add_summary(train_kld, global_step=global_step)
                summary_writer.flush()
                kld_weight = tf.Summary()
                kld_weight.value.add(tag='kld_weight', simple_value=kl_weight)
                summary_writer.add_summary(kld_weight, global_step=global_step)
                summary_writer.flush()

        if e % args.val_step == 0:
            # ------------------------------------
            # Print loss information, update the best model
            val_nll, val_kld = evaluate(model, val_iter, txtfield, args)
            logging.info("Val => [Current NLL: %5.4f] [Current KLD: %5.4f] [Best Loss: %5.4f]" 
                         % (val_nll, val_kld, best_val_loss))
            # --------------------------------
            # Save model
            val_loss = val_nll + val_kld
            early_stop += 1
            if val_loss < best_val_loss:
                print("[!] saving model...")
                torch.save(model.state_dict(), os.path.join(wpath, fprefix+".pt"))
                best_val_loss = val_loss
                early_stop = 0
            # --------------------------------
            # Add val_loss into tensorboard
            loss_sum = tf.Summary()
            loss_sum.value.add(tag='val_loss', simple_value=val_loss)
            summary_writer.add_summary(loss_sum, global_step=int(e/args.val_step))
            summary_writer.flush()
            
        if early_stop > 10: 
            print('No improvement after 10 epochs, stop training ...')
            break


def evaluate(model, val_iter, txtfield, args):
    model.eval()
    total_nll, total_kld, total_batch = 0., 0., 0.
    for b, batch in enumerate(val_iter):
        with torch.no_grad():
            nll, kld = model(batch, test=True)
            total_nll += nll.item()
            total_kld += kld.item()
            total_batch += 1
            
    avg_nll = total_nll / total_batch
    avg_kld = total_kld / total_batch
    return avg_nll, avg_kld
