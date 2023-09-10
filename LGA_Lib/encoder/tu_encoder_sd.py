import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool

from LGA_Lib.convs.wgin_conv import WGINConv_sd


class TUEncoder_sd(torch.nn.Module):
	def __init__(self, num_dataset_features, emb_dim=300, num_gc_layers=5, drop_ratio=0.0, pooling_type="standard", is_infograph=False):
		super(TUEncoder_sd, self).__init__()

		self.pooling_type = pooling_type
		self.emb_dim = emb_dim
		self.num_gc_layers = num_gc_layers
		self.drop_ratio = drop_ratio
		self.is_infograph = is_infograph

		self.out_node_dim = self.emb_dim
		if self.pooling_type == "standard":
			self.out_graph_dim = self.emb_dim
		elif self.pooling_type == "layerwise":
			self.out_graph_dim = self.emb_dim * self.num_gc_layers
		else:
			raise NotImplementedError

		self.convs = torch.nn.ModuleList()
		self.bns = torch.nn.ModuleList()

		self.wgin_sd = WGINConv_sd()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



	def forward(self, batch, x, edge_index, edge_attr=None, edge_weight=None, weights=None, encoder="encoder"):
		xs = []
		for i in range(self.num_gc_layers):
			x = self.wgin_sd(x, edge_index, edge_weight)
			x = F.linear(x, weights['{}.convs.{:d}.nn.0.weight'.format(encoder,i)], bias=weights['{}.convs.{:d}.nn.0.bias'.format(encoder,i)])
			x = F.relu(x)
			x = F.linear(x, weights['{}.convs.{:d}.nn.2.weight'.format(encoder,i)], bias=weights['{}.convs.{:d}.nn.2.bias'.format(encoder,i)])

			x = F.batch_norm(x,torch.zeros(x.data.size()[1]).to(self.device), torch.ones(x.data.size()[1]).to(self.device),
							 weights['{}.bns.{:d}.weight'.format(encoder,i)], bias=weights['{}.bns.{:d}.bias'.format(encoder,i)])

			if i == self.num_gc_layers - 1:
				# remove relu for the last layer
				x = F.dropout(x, self.drop_ratio, training=self.training)
			else:
				x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
			xs.append(x)

		# compute graph embedding using pooling
		if self.pooling_type == "standard":
			xpool = global_add_pool(x, batch)
			return xpool, x

		elif self.pooling_type == "layerwise":
			xpool = [global_add_pool(x, batch) for x in xs]
			xpool = torch.cat(xpool, 1)
			if self.is_infograph:
				return xpool, torch.cat(xs, 1)
			else:
				return xpool, x
		else:
			raise NotImplementedError

	def get_embeddings(self, loader, device, is_rand_label=False):
		ret = []
		y = []
		with torch.no_grad():
			for data in loader:
				if isinstance(data, list):
					data = data[0].to(device)
				data = data.to(device)
				batch, x, edge_index = data.batch, data.x, data.edge_index
				edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

				if x is None:
					x = torch.ones((batch.shape[0], 1)).to(device)
				x, _ = self.forward(batch, x, edge_index, edge_weight)

				ret.append(x.cpu().numpy())
				if is_rand_label:
					y.append(data.rand_label.cpu().numpy())
				else:
					y.append(data.y.cpu().numpy())
		ret = np.concatenate(ret, 0)
		y = np.concatenate(y, 0)
		return ret, y
