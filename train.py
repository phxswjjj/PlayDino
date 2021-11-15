import os
import pickle
import random
from collections import deque

import numpy as np

from dino.game import Env, Game
from dino.model import create_model

OBJECTS_DIR = 'objects'


def load_obj(name):
    global OBJECTS_DIR

    file_path = os.path.join(OBJECTS_DIR, f'{name}.pkl')
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_obj(data, name):
    global OBJECTS_DIR

    file_path = os.path.join(OBJECTS_DIR, f'{name}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def init_data():
    global OBJECTS_DIR

    if not os.path.exists(OBJECTS_DIR):
        os.makedirs(OBJECTS_DIR)

    history_path = os.path.join(OBJECTS_DIR, 'history.pkl')
    if not os.path.exists(history_path):
        history = deque(maxlen=5000)
        save_obj(history, 'history')

def load_data():
    global OBJECTS_DIR

    history_path = os.path.join(OBJECTS_DIR, 'history.pkl')
    history = load_obj('history')

    return history

def train_model(model, env):

    history = load_data()

    default_actions = np.zeros(env.ACTIONS_COUNT)
    default_actions[env.ACTIONS_OF_NOTHING] = True
    observation, reward, terminated = env.step(default_actions)

    images_of_t = np.stack([observation] * 4, axis=2)
    images_of_t = images_of_t.reshape(1, images_of_t.shape[0], images_of_t.shape[1], images_of_t.shape[2])
    images_of_prev_t = images_of_t
    images_of_t0 = images_of_t

    t = 0
    while t < 1e4:
        actions_of_t = np.zeros(env.ACTIONS_COUNT)
        reward_of_t = 0
        action_index = 0

        action_index = random.randrange(env.ACTIONS_COUNT)
        actions_of_t[action_index] = True

        observation, reward_of_t, terminated = env.step(actions_of_t)
        observation.reshape(1, observation.shape[0], observation.shape[1], 1)
        images_of_t = np.append(observation, images_of_t[:, :, :, :3], axis=3)
        data = (images_of_t, action_index, reward_of_t, actions_of_t)
        history.append(data)
        print(data)

        # if terminated, reset frame
        images_of_prev_t = images_of_t0 if terminated else images_of_t

        t += 1


if __name__ == '__main__':
    init_data()

    model = create_model()

    game = Game()
    env = Env(game)
    env.reset()

    train_model(model, env)

    agent.end()
    print('Done')
