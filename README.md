# Explain My Surprise: Learning Efficient Long-Term Memory by predicting uncertain outcomes

## Reproducing long-term dependency experiments:

### T-maze-L envirnment (where L is a minimal length of temporal dependency in this env) 
There are two possible scenarious to apply MemUP for Reinforcement Learning setting:
1. Two-phase training: (I) First, pretrain MemUP with fixed policy, then (II) train policy with the pretrained memory.
2. Train MemUP and policy simultaneously

Two-phase training (described in the paper) is easier for debug and hyperparameter search, 
as you can test each component separately. 
On the other hand, the simultaneous training is more compact and allow to train everything using one script.

Training MemUP and policy simultaneously on T-Maze-1000 (as in paper but with simultaneous training):
```commandline
python3 examples\tmaze\rllib\train_memup_policy.py -c configs/reproduce/t-maze/policy_and_memory_1k.yaml -l 1000 -s 1 -ld logs/tmp/tmaze-1k/ppo/joint/seed1
```

Two-phase training on T-Maze-20k (20000 steps):
1. MemUP pretraining: 
```commandline
python3 examples\tmaze\train_mem_only.py -l 20000 -r 20 -s 1
```
  The results will be saved in `logs\tmp\t-maze-20000\mem-only\seed1`
2. Policy training:
```commandline
python3 examples\tmaze\rllib\train_memup_policy.py -c configs/reproduce/t-maze/policy_only_20k.yaml  -l 20000 -m logs\tmp\t-maze-20000\mem-only\seed1\memory_and_acc.pt -s 1
```
We have not yet tested simultaneous training for this length.

### Copy-L task (where L+20 is a minimal length of temporal dependency in this task)

MemUP training on Copy-1020 with rollout 20 and seed 1: 
```commandline
python3 examples/copy_task/train.py -l 1000 -r 20 -s 1
```
MemUP training on Copy-5020 with rollout 100 and seed 1: 
```commandline
python3 examples/copy_task/train.py -l 5000 -r 100 -s 1
```

The results will be saved in logs/tmp/copy_{length}_...

----
