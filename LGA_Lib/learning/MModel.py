import torch
from torch.nn import Sequential, Linear, ReLU

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class MModel(torch.nn.Module):
	def __init__(self, encoder, proj_hidden_dim=200):
		super(MModel, self).__init__()

		self.encoder = encoder
		self.input_proj_dim = self.encoder.out_graph_dim

		self.proj_head = Sequential(Linear(self.input_proj_dim, proj_hidden_dim), ReLU(inplace=True),
									   Linear(proj_hidden_dim, proj_hidden_dim))

		self.init_emb()
		self.bn = torch.nn.BatchNorm1d(proj_hidden_dim)

	def init_emb(self):
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, batch, x, edge_index, edge_attr, edge_weight=None):

		z, node_emb = self.encoder(batch, x, edge_index, edge_attr, edge_weight)

		z = self.proj_head(z)
		z = self.bn(z)
		# z shape -> Batch x proj_hidden_dim
		return z, node_emb

	@staticmethod
	def calc_loss( x, x_aug, temperature=0.2, sym=True):

		batch_size, _ = x.size()
		x_abs = x.norm(dim=1)
		x_aug_abs = x_aug.norm(dim=1)

		sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
		sim_matrix = torch.exp(sim_matrix / temperature)
		pos_sim = sim_matrix[range(batch_size), range(batch_size)]
		if sym:

			loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
			loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

			loss_0 = - torch.log(loss_0).mean()
			loss_1 = - torch.log(loss_1).mean()
			loss = (loss_0 + loss_1)/2.0
		else:
			loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
			loss_1 = - torch.log(loss_1).mean()
			return loss_1

		return loss

	@staticmethod
	def calc_sim_loss( x, x_aug, temperature=0.2, sym=True):

		batch_size, _ = x.size()
		x_abs = x.norm(dim=1)
		x_aug_abs = x_aug.norm(dim=1)

		sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
		sim_matrix = torch.exp(sim_matrix / temperature)
		pos_sim = sim_matrix[range(batch_size), range(batch_size)]


		loss_0 = pos_sim

		loss_0 = - torch.log(loss_0).mean()
		return loss_0

	@staticmethod
	def calc_feature_loss( x, x_aug):

		batch_size, dim = x.size()

		c = x.T @ x_aug

		c.div_(batch_size)
		r = c.max()-c.min()
		r = r.detach()
		c = c.div(r)


		on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
		off_diag = off_diagonal(c).pow_(2).sum()

		loss = 1.0 * on_diag + 1.0 * off_diag
		loss = loss.div(batch_size * batch_size)
		return loss

	@staticmethod
	def calc_instance_loss( x, x_aug):

		batch_size, _ = x.size()
		x_abs = x.norm(dim=1)
		x_aug_abs = x_aug.norm(dim=1)

		sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)

		pos_sim = sim_matrix[range(batch_size), range(batch_size)]

		loss = 2 * pos_sim.sum().div(batch_size * batch_size) - sim_matrix.sum().div(batch_size * batch_size)

		return loss



