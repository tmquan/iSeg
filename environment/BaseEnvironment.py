import abc
import gym

class BaseEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}
    # reward_range = (0.0, 1.0)
    @abc.abstractmethod
    def __init__(self):
        self.__version__ = "0.0.1"
        print("Init BaseEnvironment")
        # Modify the observation space, low, high and shape values according to your custom environment's needs
        # self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,))
        # Modify the action space, and dimension according to your custom environment's needs
        # self.action_space = gym.spaces.Discrete(4)
        pass

    @abc.abstractmethod
    def step(self, action):
        """
        Runs one time-step of the environment's dynamics. The reset() method is called at the end of every episode
        :param action: The action to be executed in the environment
        :return: (observation, reward, done, info)
            observation (object):
                Observation from the environment at the current time-step
            reward (float):
                Reward from the environment due to the previous action performed
            done (bool):
                a boolean, indicating whether the episode has ended
            info (dict):
                a dictionary containing additional information about the previous action
        """
        # Implement your step method here
        # return (observation, reward, done, info)
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Reset the environment state and returns an initial observation
        Returns
        -------
        observation (object): The initial observation for the new episode after reset
        :return:
        """

    @abc.abstractmethod
    def render(self, mode='human', close=False):
        """
        :param mode:
        :return:
        """
        pass
