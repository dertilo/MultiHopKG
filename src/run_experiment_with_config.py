from src.experiments import run_experiment
from src.parse_args import parser


if __name__ == '__main__':
    config = {
        'data_dir':'../data/umls',
        'train':True,
        'model':'point',
        'bandwidth':400,
        'entity_dim':200,
        'relation_dim':200,
        'history_dim':200,
        'history_num_layers':3,
        'num_rollouts':20,
        'num_rollout_steps':2,
        'bucket_interval':10,
        'num_epochs':3,
        'num_wait_epochs':200,
        'num_peek_epochs':2,
        'batch_size':128,
        'train_batch_size':128,
        'dev_batch_size':32,
        'margin':1,
        'learning_rate':0.001,
        'baseline':'n/a',
        'grad_norm':0,
        'emb_dropout_rate':0.3,
        'ff_dropout_rate':0.1,
        'action_dropout_rate':0.9,
        'action_dropout_anneal_interval':1000,
        'beta':0.05,
        'beam_size':128,
        'num_path_per_entity':-1,
        'use_action_space_bucketing':True,
        'gpu':0

    }
    args = parser.parse_known_args()
    [setattr(args,k,v) for k,v in config.items()]
    run_experiment(args)
