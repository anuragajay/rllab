from mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
import numpy as np

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger


class PegEnv(MujocoEnv, Serializable):

    FILE = 'pr2_arm3d.xml'

    def __init__(self, *args, **kwargs):
        super(PegEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        self.PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])
        self.t = 0
        self.T = 100
        self.init_body_pos = self.model.body_pos[1]

    def get_current_obs(self):        
        return np.concatenate([
            self.model.data.qpos.flatten(),
            self.model.data.qvel.flatten(),
            self.model.data.site_xpos.flatten(),
            np.zeros_like(self.model.data.site_xpos.flatten()),
        ]).reshape(-1)

    def step(self, action):
        dt = 0.05
        self.t += 1
        eepts_before = self.model.data.site_xpos.flatten() 
        self.forward_dynamics(action)
        eepts_after = self.model.data.site_xpos.flatten()
        eepts_vel = (eepts_after - eepts_before)/dt
        dim = eepts_vel.shape[0]
        reward = self.cost(action)
        done = False
        ob = self.get_current_obs()
        ob[-dim:] = eepts_vel
        return Step(ob, float(reward), done)

    def reset_mujoco(self):
        self.model.data.qpos = self.init_qpos = np.array([0.1, 0.1, -1.54, -1.7, 1.54, -0.2, 0])
        self.model.data.qvel = self.init_qvel
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl
        if not hasattr(self, init_body_pos):
            self.init_body_pos = self.model.body_pos[1]
        self.model.body_pos[1] = self.init_body_pos + np.array([0.2*np.random.rand()-0.1, 0.2*np.random.rand()-0.1, 0])
        self.t = 0

    def cost(self, action):
        reward = 0
        wu = 1e-3/self.PR2_GAINS
        cost_action = (wu*(action**2)).sum()
        reward -= cost_action
        
        target = np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2])
        wp = np.array([2, 2, 1, 2, 2, 1])
        
        l1, l2 = 0.1, 10.0
        alpha = 1e-5
        wpm = 1
        wp = wp*wpm
        d = self.model.data.site_xpos.flatten() - target
        sqrtwp = np.sqrt(wp)
        dsclsq = d * sqrtwp
        dscl = d * wp
        l = 0.5 * np.sum(dsclsq ** 2) * l2 + \
                0.5 * np.log(alpha + np.sum(dscl ** 2)) * l1
        reward -= l

        l1, l2 = 1.0, 0.0
        if self.t == self.T:
            wpm = 10.0
        else:
            wpm = 0
        wp = wp*wpm
        sqrtwp = np.sqrt(wp)
        dsclsq = d * sqrtwp
        dscl = d * wp
        l = 0.5 * np.sum(dsclsq ** 2) * l2 + \
                0.5 * np.log(alpha + np.sum(dscl ** 2)) * l1
        reward -= l

        return reward

