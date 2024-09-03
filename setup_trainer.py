from trainer_triplanedep import TriplaneDepTrainer
from trainer_combo import ComboTrainer
import torch
import random
import numpy as np
from trainer import BaseTrainer

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

def setup_trainer(args, eval=False):
    if args.trainer == 'triplanedep':
        trainer = TriplaneDepTrainer(args)
    elif args.trainer == "combo":
        trainer = ComboTrainer(args)
    else:
        trainer = BaseTrainer(args)

    if eval:
        if hasattr(trainer, "color_mlp") and trainer.color_mlp is not None:
            trainer.color_mlp.eval()
        if hasattr(trainer, "deform_mlp") and trainer.deform_mlp is not None:
            trainer.deform_mlp.eval()
        if hasattr(trainer, "feature_mlp") and trainer.feature_mlp is not None:
            trainer.feature_mlp.eval()
        if hasattr(trainer, "depthmem") and trainer.depthmem is not None:
            trainer.depthmem.eval()
    return trainer