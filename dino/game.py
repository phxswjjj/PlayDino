import base64
import os
from io import BytesIO
from typing import Final

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys


class Game:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--mute-audio")

        driver = webdriver.Chrome(chrome_options=chrome_options)
        driver.set_window_position(x=-10, y=0)
        driver.get('https://chromedino.com/')
        driver.execute_script("Runner.config.ACCELERATION=0")
        driver.execute_script(
            "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'")
        self.driver = driver

    def is_crashed(self):
        return self.driver.execute_script("return Runner.instance_.crashed")

    def is_playing(self):
        return self.driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        self.driver.execute_script("Runner.instance_.restart()")

    def jump(self):
        self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def duck(self):
        self.driver.find_element_by_tag_name("body").send_keys(Keys.SPACE)

    def get_score(self):
        score_ary = self.driver.execute_script(
            "return Runner.instance_.distanceMeter.digits")
        return int(''.join(score_ary))

    def end(self):
        self.driver.close()

    def grab_image(self):
        getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); \
        return canvasRunner.toDataURL().substring(22)"
        image_b64 = self.driver.execute_script(getbase64Script)
        img = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        img = img[:300, :500]
        img = cv2.resize(img, (80, 80))
        return img

    def show_image(self, img):
        img = cv2.resize(img, (800, 400))
        cv2.imshow('image', img)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            self.end()
            cv2.destroyAllWindows()


class Env:
    ACTIONS_OF_DUCK: Final = 0
    ACTIONS_OF_JUMP: Final = 1
    ACTIONS_OF_NA: Final = 2
    ACTIONS_COUNT: Final = 3

    def __init__(self, agent):
        self.agent = agent

        loss_file_path = 'objects/loss_df.csv'
        self.loss_df = pd.read_csv(loss_file_path) \
            if os.path.isfile(loss_file_path) else pd.DataFrame(columns=['loss'])

        scores_file_path = 'objects/scores_df.csv'
        self.scores_df = pd.read_csv(scores_file_path) \
            if os.path.isfile(loss_file_path) else pd.DataFrame(columns=['scores'])

        actions_file_path = 'objects/actions_df.csv'
        self.actions_df = pd.read_csv(actions_file_path) \
            if os.path.isfile(actions_file_path) else pd.DataFrame(columns=['actions'])

        q_value_file_path = 'objects/qvalues_df.csv'
        self.q_values_df = pd.read_csv(q_value_file_path) \
            if os.path.isfile(q_value_file_path) else pd.DataFrame(columns=['qvalues'])

    def _log_score(self, score):
        self.scores_df.loc[len(self.scores_df)] = score

    def reset(self):
        self.agent.jump()

    def step(self, actions: [bool, bool, bool]):
        '''
        actions:
            [duck, jump, nothing]

        return:
            [img, reward, is_game_over]
        '''
        agent = self.agent

        action_index = self.ACTIONS_OF_NA
        for i, act in enumerate(actions):
            if act:
                self.actions_df.loc[len(self.actions_df)] = i
                action_index = i
                break
        
        todo = [agent.duck, agent.jump, lambda: None][action_index]

        score = agent.get_score()
        reward = 0.1
        is_game_over = False
        
        todo()

        img = agent.grab_image()
        if agent.is_crashed():
            self._log_score(score)
            reward = -1
            is_game_over = True
            agent.restart()
        return img, reward, is_game_over


if __name__ == '__main__':
    agent = Game()
    env = Env(agent=agent)
    env.reset()

    from time import sleep
    for i in range(10):
        sleep(1)
        agent.jump()

        actions_of_t = np.zeros(env.ACTIONS_COUNT)
        actions_of_t[env.ACTIONS_OF_JUMP] = True
        observation, reward_of_t, terminated = env.step(actions_of_t)
        agent.show_image(img=observation)

    cv2.destroyAllWindows()
    agent.end()
