import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, training_args, last_epoch=-1):
        warmup_ratio = training_args.warmup_ratio
        total_steps = training_args.num_train_samples * training_args.epochs // (training_args.total_train_batch_size // 8)
        
        self.warmup_steps = warmup_ratio * total_steps
        self.total_steps = total_steps
        self.min_lr = training_args.min_lr
        self.initial_lr = training_args.lr
        super(CosineWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.initial_lr * (self.last_epoch + 1) / self.warmup_steps for _ in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [max(self.min_lr, self.initial_lr * cosine_decay) for _ in self.base_lrs]
