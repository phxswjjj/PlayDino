import os

import pandas as pd
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
        img = self.driver.get_screenshot_as_png()
        img = img[:300, :500]
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (84, 84))
        return img

    def show_image(self, img):
        cv2.imshow('image', img)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            self.end()
            cv2.destroyAllWindows()


class Env:
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

    def reset(self):
        self.agent.jump()

    def step(self, actions):
        self.actions_df.loc[len(self.actions_df)] = actions[1]
        score = self.agent.get_score()
        reward = 0.1
        is_game_over = False
        agent = self.agent
        if actions[1] == 1:
            agent.jump()
        img = agent.grab_image()
        if agent.is_crashed():
            self.scores_df.loc[len(self.scores_df)] = score
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
        
    agent.end()
