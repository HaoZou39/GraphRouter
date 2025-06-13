import os
import pandas as pd
import numpy as np
from utils.utility import to_pickled_df, pad_history
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('history_length', 10, 'uniform history length')

def main(argv):
    data_directory = '../data'
    length = FLAGS.history_length

    train_sessions = pd.read_pickle(os.path.join(data_directory, 'sampled_train.df'))
    groups = train_sessions.groupby('route_id')
    ids = train_sessions.route_id.unique()

    state, len_state, action, next_state, len_next_state, is_done = [], [], [], [], [], []

    for rid in ids:
        group = groups.get_group(rid)
        history = []
        for idx, row in group.iterrows():
            s = list(history)
            len_state.append(length if len(s) >= length else 1 if len(s) == 0 else len(s))

            s = pad_history(s, length, np.zeros_like(row['feature_vec']))
            a = row['feature_vec']
            state.append(s)
            action.append(a)
            history.append(row['feature_vec'])
            next_s = list(history)
            len_next_state.append(length if len(next_s) >= length else 1 if len(next_s) == 0 else len(next_s))
            next_s = pad_history(next_s, length, np.zeros_like(row['feature_vec']))
            next_state.append(next_s)
            is_done.append(False)
        is_done[-1] = True

    dic = {
        'state': state,
        'len_state': len_state,
        'action': action,
        'next_state': next_state,
        'len_next_state': len_next_state,
        'is_done': is_done
    }
    replay_buffer = pd.DataFrame(data=dic)
    to_pickled_df(data_directory, replay_buffer=replay_buffer)

    dic = {'state_size': [length], 'feature_dim': [len(state[0][0])]}
    data_statis = pd.DataFrame(data=dic)
    to_pickled_df(data_directory, data_statis=data_statis)

if __name__ == '__main__':
    app.run(main)