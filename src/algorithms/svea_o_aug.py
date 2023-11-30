import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import augmentations
import algorithms.modules as m
from algorithms.sac import SAC
from augmentations import *


class SVEA_O_AUG(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.svea_alpha = args.svea_alpha
		self.svea_beta = args.svea_beta
		self.aug_func = globals()[args.augmentation.rstrip()]
		self.args=args
	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		if self.svea_alpha == self.svea_beta:
			obs = utils.cat(obs, augmentations.random_overlay(obs.clone(),self.args))
			action = utils.cat(action, action)
			target_Q = utils.cat(target_Q, target_Q)

			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = (self.svea_alpha + self.svea_beta) * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
		else:
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = self.svea_alpha * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

			obs_aug = augmentations.random_overlay(obs.clone(),self.args)
			current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
			critic_loss += self.svea_beta * \
				(F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

		if L is not None:
			L.log('train_critic/loss', critic_loss, step)
			
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

	def update(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_svea()

		if self.aug_func == 'random_mask_freq_FAN':
			obs = self.aug_func(obs, FAN_ANGLE=self.args.fan_angle)
			next_obs = self.aug_func(next_obs, FAN_ANGLE=self.args.fan_angle)

		if self.args.augmentation in ["mix_freq","mix_freq2_1","mix_freq2_2","mix_freq2_3",
									  "mix_freq2_4","mix_freq2_5","mix_freq3"]:

			obs2, action2, reward2, next_obs2, not_done2 = replay_buffer.sample_svea()
			obs=self.aug_func(obs,obs2,self.args)
			next_obs=self.aug_func(next_obs,next_obs2,self.args)

		else:
			obs = self.aug_func(obs, self.args)
			next_obs = self.aug_func(next_obs, self.args)

		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()
