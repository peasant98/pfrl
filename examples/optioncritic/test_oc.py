import pfrl
import torch
import torch.nn as nn
from pfrl.agents.option_critic import OptionCriticNetwork, OC

def main():
	feature_output_size = 8
	state_size = 6
	num_options = 8
	num_actions = 4
	featureNetwork = nn.Linear(state_size, feature_output_size)
	QNetwork = nn.Linear(feature_output_size, num_options)
	terminationNetwork = nn.Linear(feature_output_size, num_options)
	network = OptionCriticNetwork(featureNetwork, terminationNetwork, QNetwork, feature_output_size, num_options, num_actions)
	print(network)

if __name__=="__main__":
	main()