import sys
import os
sys.path.append(os.getcwd())

import torch
from itertools import chain
import time
import numpy as np
import random

from rl import envs, memup_trainer
from rl.generators import SequentialEpisodeGenerator
from rl.preprocessing import DiscountedReturn
from rl.memup_trainer import RLMemUPTrainer, RLMemSummary

from memup import MemUPMemory, MemUPPredictor, training
from memup import TopKSelector, CompositeSelector, CurrentStepSelector
from memup import TruncatedMemUPSampler, MemUPEvalSampler
from memup import PredictionErrorBasedDetector

from metrics import MSEMetric
from memup.nets import accumulate
from tmaze_networks import TMazeRecMemory, TMazePredictor
from datagen import TrajectoryBuffer, RandomSampler, OrderedSampler
from datagen.preprocessing import Composite, AddTailEvalTargets
import argparse

def detector_accuracy_on_tmaze(batch, scaling=1.):
    """
    Check if uncertainty detector works right
    best accuracy is ~0.5(top 1 + curr step predictions)
    """
    accuracy = []
    for tr, idx in zip(batch.trajs, batch.target_idx):
        val = (np.array(idx) == (len(tr) - 1))
        acc = val.mean()/scaling
        accuracy.append(acc)

    return np.mean(accuracy)


def handle_commandline():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # episode generation arguments:
    parser.add_argument('-l', '--length', type=int, default=1000, help='length of t-maze environment')
    parser.add_argument('-r', '--rollout', type=int, default=1, help='Truncated BPTT length')
    parser.add_argument('-pe', '--predict-every', type=int, default=1, help='how often to make long-term predictions inside a rollout')
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-ld', '--logdir', type=str, default=None, help='path to save tensorboard logs')
    args = parser.parse_args()
    if args.logdir is None:
        args.logdir = f"logs/tmp/t-maze-{args.length}/mem-only/seed{args.seed}/"

    assert 1 <= args.predict_every <= args.rollout, '(1 <= predict_every <= rollout) is False'

    return args


if __name__ == '__main__':
    args = handle_commandline()
    env_config = dict(env_id=f't-maze-lnr-{args.length}')
    training.ensure_dir(args.logdir)

    #pred_freq = 1
    #rollout = 20

    num_eval_episodes = 100
    batch_size = 64
    num_epochs = 200
    num_batches = 1000
    buffer_size = 4000

    rnn_dim = 256
    gamma = 0.0
    device = torch.device('cuda:0')

    target_key = 'return'
    # memory_as_context is less stable as for now...
    # doesn't learn with rollout=1, batch_size=64
    # but learns with rollout=10, batch_size=128
    memory_as_context = False
    training.fix_seed(args.seed)

    # train and test env have different seeds
    train_env, test_env = envs.train_and_test_from_config(env_config)
    # generates episodes given env and policy
    # uses random policy if policy is not specified:
    train_gen = SequentialEpisodeGenerator(train_env)
    test_gen = SequentialEpisodeGenerator(test_env)

    obs_shape = train_env.observation_space['observation'].shape
    num_actions = train_env.action_space.n

    # if memory_as_context=True then accumulated memory weights are used for
    # context embedding in target events,
    # otherwise we use respective observation
    pred_obs_shape = (rnn_dim,) if memory_as_context else obs_shape
    pred_input_key = 'context' if memory_as_context else 'observation'


    # =========== Memory Init ====================
    create_mem = lambda: TMazeRecMemory(
        obs_shape, num_actions, rnn_dim
    ).to(device)

    memory_net = create_mem()

    memup_memory = MemUPMemory(memory_net)

    mem_state = memup_memory.init_state(batch_size)
    # =============================================
    # =========== Predictor Init ===================
    create_pred = lambda: TMazePredictor(
        rnn_dim, pred_obs_shape, num_actions, pred_input_key
    ).to(device)

    predictor_net = create_pred()

    memup_predictor = MemUPPredictor(predictor_net, target_key)
    # =============================================
    # =========== optimization part ===============
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        chain(memory_net.parameters(), predictor_net.parameters()),
        lr=5e-4, #weight_decay=1e-7
    )
    # =============================================
    # Initialize stuff that create MemUPBatches:
    print('loading train buffer...')
    train_buffer = TrajectoryBuffer(buffer_size, DiscountedReturn(gamma))
    train_buffer.add_trajectories( train_gen.gen_trajs(buffer_size, verbose=True) )
    # memory and predictor accumulators are used
    # for uncertainty detection
    # and (optionally) memory_acc can be used for context embedding
    predictor_acc = create_pred().cpu()
    memory_acc = create_mem().cpu()

    # Uses accumulated versions of memory_acc and predictor_acc
    # to estimate uncartainty via prediction error
    # if context_key is specified then
    # hidden_states of memory_acc are added to the trajectories
    unc_detector = PredictionErrorBasedDetector(
        MemUPMemory(memory_acc),
        MemUPPredictor(predictor_acc, target_key),
        error_metric=torch.nn.MSELoss(reduction='none'),
        context_key='context' if memory_as_context else None
    )

    # Samples from buffer and constructs MemUPBatch
    train_sampler = TruncatedMemUPSampler(
        RandomSampler(train_buffer),
        # how to select targets for prediction
        # right now at each prediction step two targets are selected
        # (step with the highest uncertainty estimate, current step)
        CompositeSelector([
            TopKSelector(1, time_dependent_selection=True),
            CurrentStepSelector()
        ]),
        # uncertainty detector is simply a preprocessor that
        # adds uncertainty_key to the trajectories in the batch
        unc_detector,
        # determines which steps in rollout are selected for prediction:
        prediction_frequency=args.predict_every,
        # length of trajectory subsequences to be processed:
        rollout=args.rollout,
    )

    #=========== creating evaluation sampler ==============
    # We need to test model on the different samples:
    print('loading eval buffer...')
    eval_buffer = TrajectoryBuffer(
        num_eval_episodes,
        # Eval buffer assumes that trajectories have
        # a special key that contains target steps at which
        # we want to test our models predictions
        # AddTailEvalTargets(1) preprocessor
        # adds this key to trajectories
        # it stores only index of the last step for each trajectory
        Composite([DiscountedReturn(gamma), AddTailEvalTargets(1)])
    )
    eval_buffer.add_trajectories(test_gen.gen_trajs(num_eval_episodes))

    if memory_as_context:
        eval_prepoc = unc_detector
    eval_sampler = MemUPEvalSampler(
        OrderedSampler(eval_buffer),
        dynamic_preprocessor=unc_detector if memory_as_context else None
    )
    # =========== /creating evaluation sampler =============
    # ========== training =================================

    def update_accumulators():
        accumulate(memory_net, memory_acc, 0.995)
        accumulate(predictor_net, predictor_acc, 0.995)

    config = dict(
        num_batches=num_batches,
        batch_size=batch_size,
        logdir=args.logdir
    )


    trainer = RLMemUPTrainer(
        config,
        train_sampler=train_sampler,
        memory=memup_memory,
        predictor=memup_predictor,
        criterion=criterion,
        optimizer=optimizer,
        summary=RLMemSummary(args.logdir),
        eval_sampler=eval_sampler,
        eval_criterions=[MSEMetric()],
        update_callbacks=[update_accumulators],
        verbose=True
    )
    for i in range(num_epochs//10):
        trainer.train_n_epochs(10, eval=True)
        print('saving the model...')
        save_path = os.path.join(args.logdir, "memory_and_acc.pt")
        torch.save({"memory": memory_net.state_dict(),
                     "memory_acc": memory_acc.state_dict()},
                    save_path)

    trainer.close()

