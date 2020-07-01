# Deep-Q-Network (DQN) model for self driving car

import os
import random

from torch import cat
from torch import load
from torch import save
from torch import Tensor
from torch import LongTensor

from torch.nn import Linear
from torch.nn import Module
from torch.nn.functional import relu
from torch.nn.functional import smooth_l1_loss
from torch.nn.functional import softmax

from torch.optim import Adam
# from torch.autograd import Variable		# <-- API no longer needed

'''
	{#} Changed lines of code := 
		Changed function calls --> 102 ('_concat_seq_tensors')
		For 'Variable' API --> 122, 213
		For Tensor --> 172, 297 (used 'requires_grad_()' method for autograd with tensors)
	{#} Description :=
		No longer necessary for 'Variable' API to use autograd with tensors.
	{#} Reference => 
		(1) https://pytorch.org/docs/stable/autograd.html#variable-deprecated
		(2) https://pytorch.org/docs/stable/autograd.html#torch.Tensor.requires_grad
		(3) https://pytorch.org/docs/stable/tensors.html#torch.Tensor.requires_grad_
'''

car_brain_file = "CAR_BRAIN.pth"			# File name to save/load the current weights of DQN model


class Network(Module):							# Module => Base class for all neural network modules.
	""" ANN architecture """
	def __init__(self, input_size, nb_action):
		super(Network, self).__init__()
		self.input_size = input_size			# Total number of inputs in input state vector
		self.nb_action = nb_action				# Total number of output actions

		'''
			First add Linear layer with 'input_size' neurons in input layer & 
			30 neurons in hidden layer. This is fully connected layer between input layer 
			& hidden layer, hence name 'fc1'.
		'''
		self.fc1 = Linear(in_features=input_size, out_features=30)

		'''
			Add another Linear layer with 30 neurons from hidden layer & 
			'nb_action' neurons in output layer to get an action to be played. 
			Similarly, 'fc2' means another fully connected layer between hidden layer & output layer.
		'''
		self.fc2 = Linear(in_features=30, out_features=nb_action)


	def forward(self, state):
		'''
			Pass/forward propagate input signal (the input 'state') to 
			hidden layer (with 'self.fc1(state)'), then using ReLU activation function 
			(to learn non-linearity or break linearity with non-linear operations) 
			with hidden layer as input: 'relu(self.fc1(state))'.
		'''
		non_linear_hidden_signal = relu(self.fc1(state))

		'''
			Just forward propagate signal from hidden layer to output layer. 
			No activation function used because here 'q_values' are returned & 
			we'll use "softmax" method (in 'DQN' class below) to randomly select 
			the specific action depending on Q-value & probability.
		'''
		q_values = self.fc2(non_linear_hidden_signal)
		return q_values


class ReplayMemory(object):
	""" Experience replay (to store previous transitions (state, action, reward, next_state) tuple upto some 'capacity' """
	def __init__(self, capacity):
		self._capacity = capacity			# Save the maximum capacity of experience memory
		self._experience_memory = []		# Save the collection of tuples of different transitions


	def push(self, transition):
		self._experience_memory.append(transition)			# Add new transition to our experience memory list
		if len(self._experience_memory) > self._capacity:	# Check if we crossed the MAX capacity
			del self._experience_memory[0]					# Delete the oldest experience if crossed MAX capacity


	def sample(self, batch_size):
		'''
			Get some random samples of size 'batch_size' from '_experience_memory' list. 
			Also, '*' => Unzipping operator
		'''
		samples = zip(*random.sample(self._experience_memory, batch_size))

		'''
			Wrap each separate batches of states, actions, rewards & next states into 
			corresponding transition.
		'''
		return map(self._concat_seq_tensors, samples)


	def _encompass_T_G(self, m_sample):
		'''
			##### DEPRECATED USE CASE #####
			Encompass a Tensor and a Gradient using 'Variable'.
			Concatenate given sequence of tensors in given dimension 
			('dim=0' means concatenate separate samples of state, action, 
			reward & next_state from different batches into single tensor).
		'''
		return Variable(cat(m_sample, dim=0))


	def _concat_seq_tensors(self, m_sample):
		'''
			Concatenate given sequence of tensors in given dimension 
			('dim=0' means concatenate separate samples of state, action, 
			reward & next_state from different batches into single tensor).
		'''
		return cat(m_sample, dim=0)


	def get_ex_mem_len(self):
		return len(self._experience_memory)	# Return the total length of '_experience_memory' list


class DQN(object):
	""" Deep-Q-learning network with AI model & experience replay """
	def __init__(self, input_size, nb_action, gamma):
		# Gamma which is the 'discount factor', will be used in Temporal Difference (TD) later.
		self.gamma = gamma

		# Object of ANN
		self.model = Network(input_size=input_size, nb_action=nb_action)

		# Object of experience replay with capacity 1,00,000
		self.memory = ReplayMemory(capacity=100_000)

		'''
			Create object of 'Adam' optimizer to compute gradients of loss w.r.t weights & 
		update those weights using mini-batch gradient descent in directions to reduce loss.
			The parameters to 'Adam' is passed the default parameter values using built-in method of 
		class 'Module' i.e. 'self.model.parameters()' below.
			Reference to 'parameters()' => https://pytorch.org/docs/stable/nn.html#torch.nn.Module.parameters
		'''
		self.optimizer = Adam(params=self.model.parameters())

		'''
			Create an object variable of type "torch.Tensor", 'last_state', to store 
		"last input state" in each transition consist of size 4 like: 
		'[orientation, signal1, signal2, signal3]'.

		(*) requires_grad_(requires_grad=True) → Tensor =>
			Main use case is to tell autograd to begin recording operations on a Tensor
			Parameter:
				requires_grad:
					Is True if gradients need to be computed for this Tensor, 
					False otherwise.
					If autograd should record operations on this tensor.
					Default: True.
			Reference => https://pytorch.org/docs/stable/tensors.html#torch.Tensor.requires_grad_

		(*) torch.unsqueeze(input, dim) =>
			Returns a new tensor with a dimension of size one inserted at the specified position.
			The method 'unsqueeze(0)' is used to add an additional dimension at whole tensor level.
		For e.g. if a tensor is like, [5.6078e+20, 4.5911e-41, 5.6078e+20, 4.5911e-41],
		then 'unsqueeze(0)' will return, [[5.6078e+20, 4.5911e-41, 5.6078e+20, 4.5911e-41]].
			Reference => https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
		'''
		self.last_state = Tensor(input_size).requires_grad_(True).unsqueeze(0)

		self.last_action = 0	# Store last action later.
		self.last_reward = 0	# Store last reward later.


	def select_action(self, state):
		'''
			'softmax()' the Q-values as input which is returned by 'self.model(Variable(state) * 100)', 
			here state is input state containing '[orientation, signal1, signal2, signal3]'.
			Then it returns the probabilities for each of these Q-values of 3 playable actions.
			Also multiplying Q-values (returned by object 'model') by 100 is to 
			regulate the exploration of our optimized actions.
			If we do not multiply, for small Q-values, the actions may be explored much larger times & 
			hence it'll require more time to get optimized actions.

			We can also write as 'self.model.forward(Variable(state) * 100)' instead of 'self.model(Variable(state) * 100)',
			but since forward is the only method of the Network class, it is sufficient to just call 'self.model'.

			Parameter 'dim' is used to spcify how the probabilities are distributed by 'softmax()'.
			For e.g. consider the following tensor of type "torch.Tensor",
			a = [[1., 2., 3., 4.],
				 [3., 4., 5., 6.]]
			When we use 'torch.nn.functional.softmax(a, dim=0)', the result will be,
				tensor([[0.1192, 0.1192, 0.1192, 0.1192],
						[0.8808, 0.8808, 0.8808, 0.8808]])
			Similarly, for 'torch.nn.functional.softmax(a, dim=1)', the result will be,
				tensor([[0.0321, 0.0871, 0.2369, 0.6439],
						[0.0321, 0.0871, 0.2369, 0.6439]])
			That means, for 'dim=0', the probabilities are distributed across columns 
			(sum of all probabilities in a COLUMN = '1'), 
			i.e. each value in specific row has same probability value.
			For 'dim=1', the probabilities are equally distributed across rows 
			(sum of all probabilities in a ROW = '1'),
			i.e. each value in specific column has same probability value.

			The 'dim' parameter is NOT used in book, but it'll return same probability values as 
			returned by 'dim=1', but we get a deprecation warning on execution, to avoid warning 
			passed 'dim=1' below.
		'''
		# probs = softmax(self.model(Variable(state) * 100), dim=1)		# <-- OLD
		probs = softmax(self.model(state * 100), dim=1)

		# Now we take random draw from probability distribution created by 'softmax()' previously.
		action = probs.multinomial(len(probs))

		# 'action' is TWO dimensional array with single value of type "torch.Tensor".
		return action.data[0, 0]


	def learn(self, batch_states, batch_actions, batch_rewards, batch_next_states):
		'''
			Model will learn using formula for Temporal Difference (TD).
			TD = (Reward for action {t} in state {t}) 
				+ gamma * max(Q-value of best action {t} performed in future state {t+1}) 
				- (Q-value for action {t} performed in state {t}) 

			In TD formula, TARGET is => 
				(Reward for action {t} in state {t})
				+ gamma * max(Q-value of best action {t} performed in future state {t+1})
			Also PREDICTION is =>
				(Q-value for action {t} performed in state {t}) 
		'''
		'''
			Get predictions Q(s, a) (i.e. "Q-value for action {t} performed in state {t}"), 
			'batch_outputs', using 'batch_states' by passing it to 'model'.
			Here we ONLY need Q-values of actions. So, with 'gather()' using 'batch_actions' &
			'unsqueeze(1)' is used to add extra-dimension for each ROW of tensor. 
			Then 'squeeze(1)' is used revert that existing dimension added by 'unsqueeze(1)'. 
			This will return batch of Q(s, a) predictions.
		'''
		batch_outputs = self.model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)

		'''
			Get predictions of Q-values for 'batch_next_outputs', 
			(i.e. "max(Q-value of best action {t} performed in future state {t+1})"),
			for best actions performed in future state (i.e. 'batch_next_states')
			Here, 'detach()' method used to return tensor "detached" from current graph.
			The 'max(1)[0]' method used to only take the maximum of the three Q-values.
		'''
		batch_next_outputs = self.model(batch_next_states).detach().max(1)[0]

		'''
			TARGET is =>
				'batch_targets' = 'batch_rewards' + gamma * 'batch_next_outputs'
		'''
		batch_targets = batch_rewards + self.gamma * batch_next_outputs

		'''
			Finally Temporal Difference Loss is the difference between two things =>
			Batch of TARGETs & batch of PREDICTIONs (i.e. outputs of Q-values)
			So, 'td_loss' = 'batch_targets' - 'batch_outputs'.

			But we have 'smooth_l1_loss()' function that calculate this difference by taking two 
			parameters =>
				First parameter take "inputs" i.e. the predictions of Q-values in 'batch_outputs'.
				Second parameter take "targets" i.e. our calculated targets in 'batch_targets'.
		'''
		td_loss = smooth_l1_loss(batch_outputs, batch_targets)

		'''
			First clear/zeroed all gradients of optimized tensors or weights using 'zero_grad()' method of optimizer.
			Reference => https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer.zero_grad
		'''
		self.optimizer.zero_grad()

		'''
			Backpropagate the loss error using 'backward()' method from 'td_loss'.
		'''
		td_loss.backward()

		'''
			Perform updating of weights using 'step()' method of 'optimizer' object of 'Adam' optimizer class.
			Reference => https://pytorch.org/docs/stable/optim.html#torch.optim.Adam.step
		'''
		self.optimizer.step()


	def update(self, new_state, new_reward):
		'''
			Get new state as Tensor from parameter 'new_state', with 
			'requires_grad_(True)' as explained in '__init__()' of 'DQN' class.
			Also convert type to "float" using 'float()' method in order to 
			make easy for further processing in 'softmax()' method.
		'''
		new_state = Tensor(new_state).requires_grad_(True).float().unsqueeze(0)

		'''
			Add 'last_state', 'last_action', 'last_reward', 'new_state' to experience memory, 
			using 'push()' method of 'memory' object which is used in further sampling.
			
			'self.last_action' => It is first converted to 'int' to get proper action of 
			either 0, 1 or 2. Then it is converted to "64-bit integer (signed)" using 'LongTensor'.
			'self.last_reward' => Convert to "torch.Tensor" type for torch computational functions.
		'''
		self.memory.push((self.last_state, LongTensor([int(self.last_action)]), Tensor([self.last_reward]), new_state))
		
		'''
			Get new action using 'select_action()' method with 'softmax' method technique.
		'''
		new_action = self.select_action(state=new_state)

		'''
			If experience memory length is greater than 100, we draw random batch of samples 
			containing batches of states, actions, rewards & next states of size 100 
			using 'self.memory.sample(batch_size=100)'.
			Then in 'learn()' method, we make model to learn using Temporal Difference (TD), 
			backpropagating TD-Loss & updating the weights of DQN model.
		'''
		if self.memory.get_ex_mem_len() > 100:
			batch_states, batch_actions, batch_rewards, batch_next_states = self.memory.sample(batch_size=100)
			self.learn(batch_states, batch_actions, batch_rewards, batch_next_states)
		
		# Save the current state, action & reward as last ones which is needed for experience memory.
		self.last_state = new_state
		self.last_action = new_action
		self.last_reward = new_reward

		# Return the new selected action to caller.
		return new_action


	def save(self):
		'''
			Below 'save()' function in torch module (imported on line no. 8).
			Saves an object to a disk file.
			Reference => https://pytorch.org/docs/stable/torch.html#torch.save
		'''
		save({
			"state_dict": self.model.state_dict(),
			"optimizer": self.optimizer.state_dict(),
			}, car_brain_file)
		print("+1 => Model weights saved !\n")


	def load(self):
		if os.path.isfile(car_brain_file):
			print("+1 => Found last checkpoint! Now loading...")

			'''
				Below 'load()' function in torch module (imported on line no. 7).
				Loads an object saved with torch.save() from a file.
				Reference => https://pytorch.org/docs/stable/torch.html#torch.load
			'''
			checkpoint = load(car_brain_file)
			self.model.load_state_dict(checkpoint["state_dict"])
			self.optimizer.load_state_dict(checkpoint["optimizer"])
			print("+1 => Model weights loaded successfully from last checkpoint!\n")
		else:
			print("\n0 => Last checkpoint file not found/exist...\n")
