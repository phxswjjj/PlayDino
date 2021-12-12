import os
import pickle
import random
from collections import deque
from datetime import datetime
import cv2

import numpy as np

from dino.game import Env, Game
from dino.model import create_model

OBJECTS_DIR = 'objects'
BATCH_SIZE = 32


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
        played_history = deque(maxlen=5000)
        save_obj(played_history, 'history')


def load_data() -> deque:
    played_history = load_obj('history')

    return played_history


def train_model(model, env):

    played_history: deque = load_data()
    if os.path.exists('model.h5'):
        model.load_weights('model.h5')

    is_random = True
    if len(played_history) > 1e2:
        is_random = False
        print('prediction mode')

    # init actions state, 0-index is duck, 1-index is jump, 2-index is nothing
    default_actions = np.zeros(env.ACTIONS_COUNT)
    # game start by jump
    default_actions[env.ACTIONS_OF_JUMP] = True
    observation_img, reward, terminated = env.step(default_actions)

    # observation_img is 80*80
    # images_of_t is 80*80*4
    images_of_t = np.stack([observation_img] * 4, axis=2)
    # images_of_t is 1*80*80*4
    images_of_t = images_of_t.reshape(
        1, images_of_t.shape[0], images_of_t.shape[1], images_of_t.shape[2])
    images_of_prev_t = images_of_t
    images_of_t0 = images_of_t

    t = 0
    while t < 1e4:
        actions_of_t = np.zeros(env.ACTIONS_COUNT)
        reward_of_t = 0

        if is_random:
            # random action: 0=duck or 1=jump, 2=nothing
            action_index = random.randrange(env.ACTIONS_COUNT)
        else:
            # predict action
            q = model.predict(images_of_t)
            action_index = np.argmax(q)

        actions_of_t[action_index] = True

        # observation_img is 80*80
        observation_img, reward_of_t, terminated = env.step(actions_of_t)
        display_img = cv2.resize(observation_img, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imshow('dino', display_img)
        if cv2.waitKey(1) == ord('q'):
            break
        
        # observation_img is 1*80*80*1
        observation_img = observation_img.reshape(
            1, observation_img.shape[0], observation_img.shape[1], 1)
        images_of_t = np.append(
            observation_img, images_of_t[:, :, :, :3], axis=3)

        if t > BATCH_SIZE:
            # train model
            # batchHistories = random.sample(played_history, BATCH_SIZE)
            batchHistories = []
            for i in range(BATCH_SIZE, 0, -1):
                batchHistories.append(played_history[i-1])
            # 32*80*80*4
            inputs = np.zeros(
                (BATCH_SIZE, images_of_t.shape[1], images_of_t.shape[2], images_of_t.shape[3]))
            q = model.predict(images_of_t)
            # 32*3
            targets = np.zeros((BATCH_SIZE, env.ACTIONS_COUNT))
            for i, batchHistory in enumerate(iterable=batchHistories):
                images_of_history, action_index_of_history, reward_of_history, actions_of_history = batchHistory
                inputs[i] = images_of_history
                targets[i] = model.predict(images_of_history)
                if terminated:
                    targets[i, action_index_of_history] = -1
                else:
                    targets[i, action_index_of_history] = reward_of_history + \
                        0.99 * np.max(q)

            loss = model.train_on_batch(inputs, targets)

            print(t, ['duck', 'jump', 'nothing'][action_index],
                  reward_of_t, loss, terminated)

        data = (images_of_t, action_index, reward_of_t, actions_of_t)
        played_history.append(data)

        # if terminated, reset frame
        images_of_prev_t = images_of_t0 if terminated else images_of_t

        t += 1

    # save model
    model.save('model', overwrite=True)
    model.save_weights(filepath='model.h5', overwrite=True)
    save_obj(played_history, 'history')


if __name__ == '__main__':
    init_data()

    model = create_model()

    game = Game()
    env = Env(game)
    env.reset()

    train_model(model, env)

    env.end()
    cv2.destroyAllWindows()
    print('Done')
