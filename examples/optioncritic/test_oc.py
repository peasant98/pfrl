import pfrl
import torch
import torch.nn as nn
from pfrl.agents.option_critic import OptionCriticNetwork, OC

def main():
	feature_output_size = 8
	state_size = 6
	featureNetwork = nn.linear(state_size, feature_output_size)
	QNetwork = nn.Linear(feature_output_size, num_options)
	terminationNetwork = nn.Linear(feature_output_size, num_options)
	network = OptionCriticNetwork(featureNetwork, terminationNetwork, QNetwork, feature_output_size)

if __name__=="__main__":
	main()