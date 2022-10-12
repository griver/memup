import argparse
import os
import os.path as os_path
import sys
sys.path.append(os.getcwd())
from typing import Dict
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from memup import MemUPMemory, TopKSelector, DummyTailDetector, \
    MemUPEvalSampler, CurrentStepSelector, PredictionErrorBasedDetector, TruncatedMemUPSampler
from memup import training
from datagen import TrajectoryBuffer, RandomSampler, OrderedSampler
from datagen.preprocessing import AddDoneFlag, Composite, AddTailEvalTargets
from networks import SLRecMemory, SLRecPredictor
import generator as copy_task
from memup.predictor import RNNMemUPPredictor, AccumulatedContextPreprocessor
from memup.training import fix_seed
from metrics.accuracy import AccuracyMetric
from metrics.cross_entropy import CrossEntropyMetric
from rl.training_utils import disable_gradients, enable_gradients
from training.accumulator import Accumulator
from argparse import ArgumentParser


class TrainParameters(ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(add_help=False, *args, **kwargs)
        self.add_argument('-l', '--length', type=int, default=100)
        self.add_argument('-s', '--seed', type=int, default=0)
        self.add_argument('-cu', '--cuda', type=int, default=0)
        self.add_argument('-ld', '--logdir', type=str, default='logs/tmp/')
        self.add_argument('-r', '--rollout', type=int, default=20)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[
            TrainParameters()
        ]
    )
    args = parser.parse_args()
    fix_seed(args.seed)

    num_copy_symbols = 10
    time_dependency = args.length
    vocab_size = 20

    pred_freq = 10
    rollout = args.rollout

    batch_size = 128
    num_epochs = 1001
    num_batches = 1000

    rnn_dim = 128
    embedding_dim = 128
    device = torch.device(f'cuda:{args.cuda}')
    print(f"Rollout: {rollout}, pred_freq: {pred_freq}, length: {time_dependency}, n_symbols: {num_copy_symbols}")
    summary_dir = os_path.join(args.logdir, f"copy_{time_dependency}_{time.time()}")

    print("logs:", summary_dir)
    writer = SummaryWriter(summary_dir)

    #creates trajectories
    generator = copy_task.CopyTaskGenerator(num_copy_symbols, time_dependency)

    # contains trajectories
    # each added trajectory preprocesses with AddDoneFlag
    train_buffer = TrajectoryBuffer(
        10000,
        static_preprocessor=AddDoneFlag(),
    )
    train_buffer.add_trajectories([generator.gen_trajectory() for _ in range(10000)])

    eval_buffer = TrajectoryBuffer(
        1000,
        Composite([AddDoneFlag(), AddTailEvalTargets(num_copy_symbols)])
    )
    eval_buffer.add_trajectories([generator.gen_trajectory() for _ in range(1000)])
    #============================================
    #=========== Memory Init ====================

    weights = None

    memory_net = SLRecMemory(embedding_dim, rnn_dim, vocab_size).to(device)
    memup_memory = MemUPMemory(memory_net)
    memory_net_accumulator = Accumulator[SLRecMemory](
        memory_net,
        SLRecMemory(embedding_dim, rnn_dim, vocab_size).to(device),
        decay=0.97)
    accumulated_memup_memory = MemUPMemory(memory_net_accumulator.get_module())

    # ==============================================
    # =========== Predictor Init ===================
    predictor_net = SLRecPredictor(embedding_dim, rnn_dim, vocab_size, num_outputs=10).to(device)
    memup_predictor = RNNMemUPPredictor(predictor_net, target_key='y')

    pred_net_accumulator = Accumulator[SLRecPredictor](
        predictor_net,
        SLRecPredictor(embedding_dim, rnn_dim, vocab_size, num_outputs=10).to(device),
        decay=0.97)

    accumulated_memup_predictor = RNNMemUPPredictor(pred_net_accumulator.get_module(), target_key='y')

    train_sampler_1 = TruncatedMemUPSampler(
        RandomSampler(train_buffer),
        TopKSelector(num_copy_symbols, time_dependent_selection=True),
        Composite([
            AccumulatedContextPreprocessor(accumulated_memup_predictor),
            PredictionErrorBasedDetector(
            accumulated_memup_memory,
            accumulated_memup_predictor,
            torch.nn.CrossEntropyLoss(reduction='none'))
        ]),
        prediction_frequency=pred_freq,
        rollout=rollout
    )

    train_sampler_2 = TruncatedMemUPSampler(
        RandomSampler(train_buffer),
        CurrentStepSelector(),
        Composite([
            AccumulatedContextPreprocessor(accumulated_memup_predictor),
            DummyTailDetector(num_copy_symbols),
        ]),
        prediction_frequency=1,
        rollout=rollout
    )

    eval_sampler = MemUPEvalSampler(
        OrderedSampler(eval_buffer),
        dynamic_preprocessor=AccumulatedContextPreprocessor(accumulated_memup_predictor))

    #============= Optimization ====================
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        memory_net.parameters(),
        lr=2e-4
    )
    predictor_optimizer = torch.optim.Adam(
        predictor_net.parameters(),
        lr=2e-4
    )

    print("pretrain detector")
    disable_gradients(memory_net)

    mem_state_2 = memup_memory.init_state(batch_size)
    pred_state_2 = memup_predictor.init_state(batch_size)

    for e in range(20):

        print("==" * 7, 'EPOCH #{}'.format(e + 1), "==" * 7)

        pred_net_accumulator.accumulate(predictor_net)

        for batch in train_sampler_2.epoch(batch_size, num_batches):
            out = training.update_on_batch_two_rnn(
                batch,
                memup_memory,
                memup_predictor,
                mem_state_2,
                pred_state_2,
                criterion,
                predictor_optimizer,
            )
            loss = out['loss']
            mem_state_2 = out['mem_state']
            pred_state_2 = out['pred_state']

    enable_gradients(memory_net)

    print("main loop training")
    mem_state_1 = memup_memory.init_state(batch_size)
    # =============================================
    for e in range(num_epochs):
        # ===== memup learning ==========
        start_time = time.time()
        print("==" * 7, 'EPOCH #{}'.format(e+1), "==" * 7)
        epoch_batches = 0
        total_loss = 0.

        pred_net_accumulator.accumulate(predictor_net)
        memory_net_accumulator.accumulate(memory_net)

        disable_gradients(predictor_net.lstm)

        for batch in train_sampler_1.epoch(batch_size, num_batches):

            out = training.update_on_batch_two_rnn(
                batch,
                memup_memory,
                memup_predictor,
                mem_state_1,
                None,
                criterion,
                optimizer,
                multiplier=0.1
            )

            loss = out['loss']
            mem_state_1 = out['mem_state']
            total_loss += loss

        print(total_loss)
        enable_gradients(predictor_net.lstm)
        disable_gradients(memory_net)

        for batch in train_sampler_2.epoch(batch_size, num_batches):

            out = training.update_on_batch_two_rnn(
                batch,
                memup_memory,
                memup_predictor,
                mem_state_2,
                pred_state_2,
                criterion,
                predictor_optimizer,
            )

            loss = out['loss']
            mem_state_2 = out['mem_state']
            pred_state_2 = out['pred_state']
            total_loss += loss

        print(total_loss)
        enable_gradients(memory_net)

        time_diff = time.time() - start_time
        print(f'UPS: {num_batches / time_diff:.2f}')

        eval_loss: Dict[str, float] = training.eval_memory_and_predictor_two_rnn(
            eval_sampler, accumulated_memup_memory, accumulated_memup_predictor, [AccuracyMetric(), CrossEntropyMetric()]
        )
        print(f"Evaluation Loss: Accuracy:{eval_loss['Accuracy']:.4f}, CrossEntropy:{eval_loss['CrossEntropy']:.5f}")
        writer.add_scalar("Accuracy", eval_loss['Accuracy'], e)
        writer.add_scalar("CrossEntropy", eval_loss['CrossEntropy'], e)

        if e % 100 == 0 and e > 0:
            path = os_path.join(summary_dir, f"/copy_{time_dependency}_epoch_{e}.pt")
            torch.save({
                "memory": memory_net.state_dict(),
                "predictor": predictor_net.state_dict(),
                "memory_acc": memory_net_accumulator.get_module().state_dict(),
                "predictor_acc": pred_net_accumulator.get_module().state_dict()
            }, path)

