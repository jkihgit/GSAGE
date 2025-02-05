import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel

import numpy as np
import time
from sklearn.metrics import f1_score, classification_report
from datetime import datetime
import argparse
import sklearn
from _thread import start_new_thread
from functools import wraps
import traceback

from graphsage.loader import load_reddit, load_ppi, load_pubmed, load_synth
from graphsage.graphsage import GraphSAGE 

from torch.cuda.nvtx import range_push as nvtxPush
from torch.cuda.nvtx import range_pop as nvtxPop
from torch.cuda.nvtx import mark as nvtxTimestamp 

torch.set_default_tensor_type('torch.cuda.FloatTensor')
mpmethod = 'fork' # Linux
# mpmethod = 'spawn' # Windows

def ArgmaxF1(y_out, y_truth):
    return f1_score(y_truth.argmax(axis=1), y_out.argmax(axis=1), average="micro")

def MultilabelF1(y_out, y_truth):
    y_out[y_out > 0.5] = 1
    y_out[y_out <= 0.5] = 0
    # print ("Classification report: \n", (classification_report(y_truth, y_out)))
    return f1_score(y_truth, y_out, average="micro")

f1_func = ArgmaxF1

def thread_wrapped_func(func):
    """
    Wraps a process entry point to make it work with OpenMP.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = torch.multiprocessing.Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function

d = datetime.now()
def run(args, dataset, mgpu_data):
    def inference(x, model, batch_size):
        retval = []
        batch_count = len(x)//batch_size
        model.eval()
        for batch in range(batch_count):
            batch_deg_conf = model.test_forward(torch.LongTensor(x[batch*batch_size:(batch+1)*batch_size]).cuda()).cpu().data.numpy()
            retval.append(batch_deg_conf)
        model.train()
        retval = np.array(retval)
        retval = retval.reshape([-1, retval.shape[-1]])
        return retval

    n_gpus, devices, pid, gpu = mgpu_data
    batch_size = args.batch_size
    epoch_count = args.num_epochs

    if n_gpus > 1:
        nodes_per_gpu = len(dataset['train_labels']) // n_gpus
        dataset['train_nodes'] = dataset['train_nodes'][gpu*nodes_per_gpu : (gpu+1)*nodes_per_gpu]
        dataset['train_labels'] = dataset['train_labels'][gpu*nodes_per_gpu : (gpu+1)*nodes_per_gpu]
        dev_id = devices[pid]
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='12345')
        # dist_init_method = None
        world_size = n_gpus
        torch.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=pid)
        torch.cuda.set_device(dev_id)
        graphsage_raw = GraphSAGE(args, dataset).cuda()
        graphsage = DistributedDataParallel(graphsage_raw, device_ids=[dev_id], output_device=dev_id, bucket_cap_mb=args.nccl_buffer_size)
    else:
        graphsage_raw = graphsage = GraphSAGE(args, dataset).cuda()
    params = filter(lambda p : p.requires_grad, graphsage.parameters())
    # params = graphsage.parameters()
    # param_size = 0
    # for param in params:
    #     l = param.size()
    #     print ('param:', l)
    #     l = np.prod(l)
    #     param_size += l
    # print ('Parameter size:', param_size)
    opt = args.optimizer.lower()
    if opt == 'sgd':
        optimizer = torch.optim.SGD(params, lr=0.7)
    elif opt == 'adam':
        optimizer = torch.optim.Adam(params, lr=0.01)
    else:
        raise NotImplemented
    xent = nn.BCEWithLogitsLoss().cuda()
    # drop last batch if not enough nodes for batch
    batch_count = len(dataset['train_labels'])//batch_size
    batch_times = []
    epoch_times = []
    tests_since_improvement = 0
    best_f1 = 0
    best_epoch = 0
    patience_over = False

    print ('pid', pid, batch_count, 'batches / epoch')
    for epoch in range(epoch_count):
        print ('pid', pid, 'epoch', epoch)
        dataset['train_nodes'], dataset['train_labels'] = sklearn.utils.shuffle(dataset['train_nodes'], dataset['train_labels'])
        epoch_start_time = time.time()
        for batch in range(batch_count):  
            start_time = time.time()
            nvtxPush('Batch_Input_Overhead')
            batch_nodes_cpu = dataset['train_nodes'][batch*batch_size:(batch+1)*batch_size]
            batch_nodes = torch.LongTensor(batch_nodes_cpu).cuda()
            nvtxPop()
            nvtxPush('Forward')
            batch_preds = graphsage.forward(batch_nodes)
            nvtxPop()
            nvtxPush('Loss')
            loss = xent(batch_preds, Variable(torch.FloatTensor((dataset['train_labels'][batch*batch_size:(batch+1)*batch_size])).cuda()))
            nvtxPop()
            nvtxPush('Gradient')
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_value_(params, 5.0)
            loss.backward()
            nvtxPop()
            nvtxPush('Optimizer')
            optimizer.step()
            nvtxPop()
            end_time = time.time()
            # enabling barrier decreases NCCL F32 time from 6ms to 1ms and decreases batch time from 17ms to 16ms, and adds a NCCL U8 kernel
            # if n_gpus > 1:
            #     torch.distributed.barrier()
            batch_times.append(end_time-start_time)
            if args.fast_stop:
                assert False
        epoch_times.append(time.time()-epoch_start_time)
        print ('pid', pid, 'loss', loss.item())

        if (pid==0) and (((epoch+1) % args.eval_every) == 0):
            test_output = inference(dataset['test_nodes'], graphsage_raw, args.val_batch_size)
            testF1 = f1_func(test_output, dataset['test_labels'])
            print ("Testing F1:", testF1)
            if (args.patience > 0) and (not patience_over):
                if best_f1 < testF1:
                    tests_since_improvement = 0
                    best_f1 = testF1
                    best_epoch = epoch
                    # normally we would save the model state, but we're worried about running time, not getting a good model
                else:
                    tests_since_improvement += 1
                if tests_since_improvement >= args.patience:
                    print ('patience_over', tests_since_improvement, args.patience, epoch)
                    print ('best_f1', best_f1)
                    print ('best_epoch', best_epoch)
                    patience_over = True

    if args.validation and (pid==0):
        val_output = inference(dataset['val_nodes'], graphsage_raw, args.val_batch_size)
        print ('Validation F1:', f1_func(val_output, dataset['val_labels']))

    print ('pid', pid, "Average batch time:", np.mean(batch_times))
    print ('pid', pid, "Average epoch time:", np.mean(epoch_times))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Non-DGL Pytorch GraphSAGE")
    argparser.add_argument('--gpu', type=str, default='0', help="Comma separated list of GPU device IDs.")
    argparser.add_argument('--num-epochs', type=int, default=100)
    argparser.add_argument('--num-hidden', type=int, default=512)
    argparser.add_argument('--samples1', type=int, default=25)
    argparser.add_argument('--samples2', type=int, default=10)
    argparser.add_argument('--batch-size', type=int, default=512)
    argparser.add_argument('--val-batch-size', type=int, default=64)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--fast-stop', type=int, default=0)
    argparser.add_argument('--validation', type=int, default=1)
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--dataset-location', type=str, default='../pickled_data/')
    argparser.add_argument('--optimizer', type=str, default='adam')
    argparser.add_argument('--nccl-buffer-size', type=int, default=25, help='in MB')
    argparser.add_argument('--patience', type=int, default=-1, help='if >0 will run until this many tests without improvement')
    argparser.add_argument('--num-classes', type=int, default=-1, help='if >0 will affect the number of classes in the dataset. This means accuracy becomes mostly meaningless. Especially for PPI\'s multilabel...label. Also used by synthetic dataset')
    argparser.add_argument('--num-nodes', type=int, default=5120, help='Used by synthetic dataset')
    argparser.add_argument('--feat-size', type=int, default=512, help='Used by synthetic dataset')
    argparser.add_argument('--avg-deg', type=int, default=100, help='Used by synthetic dataset')
    argparser.add_argument('--max-deg', type=int, default=128, help='Used by synthetic dataset')
    args = argparser.parse_args()    
    print ('GPU:', args.gpu)
    print ('EPOCHS:', args.num_epochs)
    print ('BS:', args.batch_size)
    print ('FAST STOP:', args.fast_stop)
    print ('VAL:', args.validation)
    print ('DS:', args.dataset)
    print ('OPT:', args.optimizer)
    print ('BUFF:', args.nccl_buffer_size)
    print ('PATIENCE:', args.patience)
    print ('CLASSES', args.num_classes)
    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)
    # pytorch problems + lazy coding
    assert devices == list(range(n_gpus))

    loadtime = datetime.now()
    load_func_args = args.num_classes
    ds = args.dataset.strip().lower()
    if ds == 'reddit':
        load_func = load_reddit
    elif ds == 'pubmed':
        load_func = load_pubmed
    elif ds == 'ppi':
        load_func = load_ppi
        f1_func = MultilabelF1
    elif (ds == 'synthetic') or (ds == 'synth'):
        load_func = load_synth
        load_func_args = (args.num_nodes, args.feat_size, args.num_classes, args.avg_deg, args.max_deg)
        print ('NUM CLASSES:', args.num_classes)
        print ('NUM NODES:', args.num_nodes)
        print ('AVG DEG:', args.avg_deg)
        print ('MAX DEG:', args.max_deg)
        print ('NUM FEATS:', args.feat_size)
    else:
        raise NotImplemented
    dataset = load_func(args.dataset_location, load_func_args)
    print ('Loading took', str(datetime.now()-loadtime))
    if n_gpus < 2:
        print ('single GPU')
        run(args, dataset, (n_gpus, devices, 0, 0))
    else:
        print ('multi GPU')
        torch.multiprocessing.set_start_method(mpmethod)
        procs = []
        gpu = 0
        for pid in range(n_gpus):
            p = torch.multiprocessing.Process(target=thread_wrapped_func(run), args=(args, dataset, (n_gpus, devices, pid, gpu)))
            p.start()
            procs.append(p)
            gpu += 1
        for p in procs:
            p.join()
    print("WALLCLOCK", str(datetime.now()-d))
