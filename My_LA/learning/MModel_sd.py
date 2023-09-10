import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class MModel_sd(torch.nn.Module):
	def __init__(self, encoder, proj_hidden_dim=300, device=None):
		super(MModel_sd, self).__init__()

		self.encoder = encoder
		self.input_proj_dim = self.encoder.out_graph_dim

		self.proj_head = Sequential(Linear(self.input_proj_dim, proj_hidden_dim), ReLU(inplace=True),
									   Linear(proj_hidden_dim, proj_hidden_dim))

		self.init_emb()
		self.device = device

	def init_emb(self):
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, batch, x, edge_index, edge_attr, edge_weight=None, weights=None ):

		z, node_emb = self.encoder(batch, x, edge_index, edge_attr, edge_weight, weights=weights)

		z = F.linear(z, weights['proj_head.0.weight'], bias=weights['proj_head.0.bias'])
		z = F.relu(z)
		z = F.linear(z, weights['proj_head.2.weight'], bias=weights['proj_head.2.bias'])
		z = F.batch_norm(z, torch.zeros(z.data.size()[1]).to(self.device), torch.ones(z.data.size()[1]).to(self.device),
		 				 weights['bn.weight'],bias=weights['bn.bias'])

		# z shape -> Batch x proj_hidden_dim
		return z, node_emb

	# @staticmethod
	# def calc_loss( x, x_aug, temperature=0.2, sym=True):
	# 	# x and x_aug shape -> Batch x proj_hidden_dim
	#
	# 	batch_size, _ = x.size()
	# 	x_abs = x.norm(dim=1)
	# 	x_aug_abs = x_aug.norm(dim=1)
	#
	# 	sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
	# 	sim_matrix = torch.exp(sim_matrix / temperature)
	# 	pos_sim = sim_matrix[range(batch_size), range(batch_size)]
	# 	if sym:
	#
	# 		loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
	# 		loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
	#
	# 		loss_0 = - torch.log(loss_0).mean()
	# 		loss_1 = - torch.log(loss_1).mean()
	# 		loss = (loss_0 + loss_1)/2.0
	# 	else:
	# 		loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
	# 		loss_1 = - torch.log(loss_1).mean()
	# 		return loss_1
	#
	# 	return loss
	#
	# @staticmethod
	# def calc_sim_loss( x, x_aug, temperature=0.2, sym=True):
	# 	# x and x_aug shape -> Batch x proj_hidden_dim
	#
	# 	batch_size, _ = x.size()
	# 	x_abs = x.norm(dim=1)
	# 	x_aug_abs = x_aug.norm(dim=1)
	#
	# 	sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
	# 	sim_matrix = torch.exp(sim_matrix / temperature)
	# 	pos_sim = sim_matrix[range(batch_size), range(batch_size)]
	#
	#
	# 	loss_0 = pos_sim
	#
	# 	loss_0 = - torch.log(loss_0).mean()
	# 	return loss_0
	#
	# @staticmethod
	# def calc_blt_loss( x, x_aug, temperature=0.2, sym=True):
	#
	# 	batch_size, _ = x.size()
	#
	# 	c = x.T @ x_aug
	#
	# 	# sum the cross-correlation matrix between all gpus
	# 	c.div_(batch_size)
	# 	# torch.distributed.all_reduce(c)
	#
	# 	on_diag = torch.diagonal(c).pow_(2).sum()
	# 	off_diag = off_diagonal(c).pow_(2).sum()
	# 	loss = on_diag + 1.0 * off_diag
	#
	# 	return loss



