import os
import subprocess
import random
import datetime
import shutil
import numpy as np
import torch
import torch.utils.data
import torch.distributed as dist
from config import config_parser
from tensorboardX import SummaryWriter
from loaders.create_training_dataset import get_training_dataset
from trainer import BaseTrainer
import time
from eval import eval_one_step
from loaders.raft import RAFTEvalDataset
# from trainer_tcnn import TcnnTrainer
from trainer_triplanedep import TriplaneDepTrainer
from trainer_combo import ComboTrainer


seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(args):
    seq_name = os.path.basename(args.data_dir.rstrip('/'))
    now = time.strftime("%y%m%d-%H%M", time.localtime())
    out_dir = os.path.join(args.save_dir, '{}_{}_{}'.format(now, args.expname, seq_name))
    os.makedirs(out_dir, exist_ok=True)
    print('optimizing for {}...\n output is saved in {}'.format(seq_name, out_dir))

    args.out_dir = out_dir

    # save the args and config files
    f = os.path.join(out_dir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            if not arg.startswith('_'):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))

    if args.config:
        f = os.path.join(out_dir, 'config.txt')
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    if "RGB"in args.data_dir:
        log_dir = 'RGB_logs/{}_{}_{}'.format(now, args.expname, seq_name)
    else:
        log_dir = 'logs/{}_{}_{}'.format(now, args.expname, seq_name)

    writer = SummaryWriter(log_dir)

    g = torch.Generator()
    g.manual_seed(args.loader_seed)
    dataset, data_sampler = get_training_dataset(args, max_interval=args.start_interval)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.num_pairs,
                                              worker_init_fn=seed_worker,
                                              generator=g,
                                              num_workers=args.num_workers,
                                              sampler=data_sampler,
                                              shuffle=True if data_sampler is None else False,
                                              pin_memory=True)
    
    # eval_dataset = RAFTEvalDataset(args, max_interval=args.start_interval)
    # eval_data_loader = torch.utils.data.DataLoader(eval_dataset,
    #                                                batch_size = 32,
    #                                                 shuffle = True,
    #                                                 num_workers = 8,
    #                                                 pin_memory = True)
    # eval_batch = next(iter(eval_data_loader))
    # del eval_data_loader
    # del eval_dataset
    eval_it = 500

    # get trainer
    if args.trainer == 'triplanedep':
        trainer = TriplaneDepTrainer(args)
    elif args.trainer == 'combo':
        trainer = ComboTrainer(args)
    else:
        trainer = BaseTrainer(args)

    start_step = trainer.step + 1
    step = start_step
    epoch = 0
    t = time.time()
    t_100 = time.time()
    t_list = []
    t_compute = []
    run_time_acc = 0
    run_time_st = time.time()
    load_100 = 0
    load_st = time.time()
    forw_list = []
    back_list = []
    torch.cuda.empty_cache()
    while step < args.num_iters + start_step + 1:
        for batch in data_loader:
            load_100 += time.time() - load_st
            st = time.time()
            # trainer.eval_one_step(step)
            trainer.train_one_step(step, batch) 
            
            # t_compute.append(time.time() - st)
            # forw_list.append(trainer.forward_time)
            # back_list.append(trainer.backtime)
            # torch.cuda.empty_cache()
            # if step % 100 == 0:
                # torch.cuda.empty_cache()

            if  (step % eval_it == 0 or step == 100) and trainer.eval:
                run_time_acc += time.time() - run_time_st
                if hasattr(trainer, 'feature_mlp') and trainer.feature_mlp is not None:
                    trainer.feature_mlp.eval()
                trainer.deform_mlp.eval()
                if hasattr(trainer, 'color_mlp') and trainer.color_mlp is not None:
                    trainer.color_mlp.eval()
                if hasattr(trainer, 'RTs'):
                    trainer.RTs.eval()
                
                with torch.no_grad():
                    # trainer.scalars_to_log = {}
                    res = eval_one_step(trainer, step, depth_err=0.04)
                    if hasattr(trainer, 'RTs'):
                        print("Rts = ")
                        print(trainer.RTs.Prots)
                        print(trainer.RTs.Qrots)
                        print(trainer.RTs.Diags)
                        print( trainer.RTs.Ts)
                
                if step < 10:
                    print("loss = ", trainer.scalars_to_log)
                    
                # trainer.log(writer, step)
                res['run_time'] = run_time_acc
                res['step'] = step
                np.save(os.path.join(out_dir, 'eval', f'metric_{step:08d}.npy'), res)
                
                if hasattr(trainer, 'feature_mlp') and trainer.feature_mlp is not None:
                    trainer.feature_mlp.train()
                trainer.deform_mlp.train()
                if hasattr(trainer, 'color_mlp') and trainer.color_mlp is not None:
                    trainer.color_mlp.train()
                if hasattr(trainer, 'RTs'):
                    trainer.RTs.train()
                
                
                run_time_st = time.time()
                
            elif step % eval_it == 0:
                print("eval = ", trainer.eval)
            
            if step % 100 == 0:
                trainer.log(writer, step)
                dataset.set_step(step)
                if args.inc_step > 0:
                    dataset.increase_range()
           

            
            # if step % args.i_print == 0:
            #     print("deform sample = ", trainer.deform_mlp.layers1[0].map_st.sample_time)
            #     print("deform forward = ", trainer.deform_mlp.layers1[0].map_st.forward_time)
            #     print("color sample = ", trainer.color_mlp.sample_time)
            #     print("color forward = ", trainer.color_mlp.forward_time)
            #     print("back_time = ", trainer.backtime)

            

            # dataset.set_max_interval(args.start_interval + step // 2000)
            
            if step % 100 == 0: # and hasattr(dataset, 'center_range'):
                
                if hasattr(trainer, 'forward_time'):
                    print(f"forw_time = {trainer.forward_time:.4f}")
                    
                if hasattr(trainer, 'backtime'):
                    print(f"back_time = {trainer.backtime:.4f}")
                    
                if hasattr(trainer, 'sample_time'):
                    print(f"sample_time = {trainer.sample_time:.4f}")
                if hasattr(trainer, 'loss_time'):
                    print(f"loss_time = {trainer.loss_time:.4f}")
                if hasattr(trainer, 'pred_time'):
                    print(f"pred_time = {trainer.pred_time:.4f}")
                if hasattr(trainer, 'data_time'):
                    print(f"data_time = {trainer.data_time:.4f}")
                if hasattr(trainer, 'read_time'):
                    print(f"read_time = {trainer.read_time:.4f}")
                if hasattr(trainer, "cuda_time"):
                    print(f"cuda_time = {trainer.cuda_time:.4f}")

                print(f"full_time = {time.time() - t:.4f}")
                print(f"100_time = {time.time() - t_100:.4f}")
                print(f"100_load_time = {load_100:.4f}")
                load_100 = 0
                
                # np.save(os.path.join(out_dir, 'time.npy'), np.array(t_list))
                # np.save(os.path.join(out_dir, 'compute.npy'), np.array(t_compute))
                # np.save(os.path.join(out_dir, 'forw.npy'), np.array(forw_list))
                # np.save(os.path.join(out_dir, 'back.npy'), np.array(back_list))
                if 'ids1' not in batch.keys():
                    
                    for k, subbatch in batch.items():
                        print("batch: ", k)
                        print("ids1", subbatch['ids1'])
                        print("ids2", subbatch['ids2'])
                else:
                    print("ids1", batch['ids1'])
                    print("ids2", batch['ids2'])
                t_100 = time.time()
            
            step += 1
            # t_list.append(time.time() - t)
            
            t = time.time()
            #     dataset.increase_center_range_by(4)
                

            if step >= args.num_iters + start_step + 1:
                break
            load_st = time.time()

        epoch += 1
        if args.distributed:
            data_sampler.set_epoch(epoch)


if __name__ == '__main__':
    args = config_parser()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    train(args)

