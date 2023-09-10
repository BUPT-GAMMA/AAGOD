import os
import re
import os.path as osp
from scipy import sparse as sp
import torch
import numpy as np
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Constant
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_scipy_sparse_matrix, degree, from_networkx
# from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred.dataset_pyg_aug import PygGraphPropPredDataset
# from dataset_pyg_aug import PygGraphPropPredDataset
from sklearn.model_selection import StratifiedKFold


def read_graph_file(DS, path):
    prefix = os.path.join(path, DS, DS)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    graph_indic = {}
    with open(filename_graph_indic) as f:
        i = 1
        for line in f:
            line = line.strip("\n")
            graph_indic[i] = int(line)
            i += 1

    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                node_labels += [int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')

    filename_node_attrs = prefix + '_node_attributes.txt'
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')

    label_has_zero = False
    filename_graphs = prefix + '_graph_labels.txt'
    graph_labels = []

    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)

    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])

    filename_adj = prefix + '_A.txt'
    adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
    index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0, e1))
            index_graph[graph_indic[e0]] += [e0, e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k] = [u - 1 for u in set(index_graph[k])]

    graphs = []
    for i in range(1, 1 + len(adj_list)):
        G = nx.from_edgelist(adj_list[i])
        G.graph['label'] = graph_labels[i - 1]

        mapping = {}
        it = 0
        for n in G.nodes:
            mapping[n] = it
            it += 1

        G_pyg = from_networkx(nx.relabel_nodes(G, mapping))
        G_pyg.y = G.graph['label']
        G_pyg.x = torch.ones((G_pyg.num_nodes,1))

        if G_pyg.num_nodes > 0:
            graphs.append(G_pyg)

    return graphs


def init_structural_encoding(gs, rw_dim=16, dg_dim=16):
    for g in gs:
        A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
        D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

        Dinv = sp.diags(D)
        RW = A * Dinv
        M = RW

        RWSE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(rw_dim-1):
            M_power = M_power * M
            RWSE.append(torch.from_numpy(M_power.diagonal()).float())
        RWSE = torch.stack(RWSE,dim=-1)

        g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(0, dg_dim - 1)
        DGSE = torch.zeros([g.num_nodes, dg_dim])
        for i in range(len(g_dg)):
            DGSE[i, int(g_dg[i])] = 1

        g['x_s'] = torch.cat([RWSE, DGSE], dim=1)

    return gs


def get_ogb_dataset(args, train_per=0.8, need_str_enc=True):
    if args.DS_pair is not None:
        DSS = args.DS_pair.split("+")
        DS, DS_ood = DSS[0], DSS[1]
    else:
        DS, DS_ood = args.DS, args.DS_ood

    TU = not DS.startswith('ogbg-mol')

    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    path_ood = osp.join(osp.dirname(osp.realpath(__file__)), 'data', DS_ood)
    
    if TU:
        dataset_ood = TUDataset(path_ood, name=DS_ood)
        dataset = TUDataset(path, name=DS, transform=(Constant(1, cat=False)))
    else:
        dataset = PygGraphPropPredDataset(name=DS, root=path,aug=args.aug)
        dataset.data.x = dataset.data.x.type(torch.float32)
        dataset.data.edge_index = dataset.data.edge_index.to(torch.long)

        dataset_ood = (PygGraphPropPredDataset(name=DS_ood, root=path_ood,aug=args.aug))
        dataset_ood.data.x = dataset_ood.data.x.type(torch.float32)
        dataset_ood.data.edge_index = dataset_ood.data.edge_index.type(torch.long)
    
    #change here
    if args.device >= 0:
        device = torch.device("cuda:" + str(args.device))

    num_sample = len(dataset)
    num_train = int(num_sample * train_per)
    num_test = int(num_sample * 0.1)
    num_valid = int(num_sample * 0.1)
    
    indices = torch.randperm(num_sample)
    idx_train = torch.sort(indices[:num_train])[0]
    idx_test = torch.sort(indices[num_train: num_train+num_test])[0]
    idx_valid = torch.sort(indices[num_train+num_test:])[0]

    dataset_train = dataset[idx_train]
    #训练集多大，测试集多大
    dataset_valid = dataset[idx_valid]
    dataset_test = dataset[idx_test]
    dataset_valid_ood = dataset_ood[len(dataset_test): len(dataset_valid) + len(dataset_test)]
    dataset_ood = dataset_ood[: len(dataset_test)]
    
    dataset_num_features = dataset[0][0].num_node_features
    dataset_num_features_ood = dataset_ood[0][0].num_node_features
    assert dataset_num_features == dataset_num_features_ood
    
    data_list_train = []
    data_list_all = []
    idx = 0
    # for data in dataset_train:
    #     # data.y = 0
    
    #     data[0]['idx'] = idx
    #     data[1]['idx'] = idx
    #     # data[0].edge_index = data[0].edge_index.to(torch.long)
    #     # data[1].edge_index = data[1].edge_index.to(torch.long)
    #     idx += 1
    #     data_list_train.append(data)
    #     data_list_all.append(data)
    print('here1111111111-------------------------')

    # #change here
    data_list_test = []
    data_list_nontrain = []
    data_list_valid = []
    
    y_dim1 = len(dataset_test[0][0].y)
    y_dim2 = len(dataset_test[0][0].y[0])
    for data in dataset_valid:
        data[0].edge_attr = None
        data[1].edge_attr = None
        data[0].edge_index = data[0].edge_index.to(torch.long)
        data[1].edge_index = data[1].edge_index.to(torch.long)
        data_list_valid.append(data)
        data_list_all.append(data)
        data_list_nontrain.append(data)

    #change here


    ood_y_dim1 = len(dataset_ood[0][0].y)
    ood_y_dim2 = len(dataset_ood[0][0].y[0])
    print(ood_y_dim1,ood_y_dim2)
    reshape = 0
    if y_dim1 != ood_y_dim1 or ood_y_dim2 != y_dim2:
        reshape = 1
    #change here
    for data in dataset_valid_ood:
        data[0].edge_attr = None
        data[1].edge_attr = None
        data[0].edge_index = data[0].edge_index.to(torch.long)
        data[1].edge_index = data[1].edge_index.to(torch.long)
        if reshape:
            y_filled = torch.zeros((y_dim1,y_dim2))
            # use mean to fill the y in ood data
            for i in range(y_dim1):
                for j in range(y_dim2):
                    y_filled[i][j] = torch.mean(torch.tensor(data[0].y,dtype=torch.float64))
            for i in range(min(y_dim1,ood_y_dim1)):
                for j in range(min(y_dim2,ood_y_dim2)):
                    y_filled[i][j] = data[0].y[i][j]
            data[0].y = y_filled   
            data[1].y = y_filled
        data_list_valid.append(data) 
        data_list_all.append(data)
        data_list_nontrain.append(data)


    for data in dataset_test:
        data[0].edge_attr = None
        data[1].edge_attr = None
        data[0].edge_index = data[0].edge_index.to(torch.long)
        data[1].edge_index = data[1].edge_index.to(torch.long)
        data_list_test.append(data)
        data_list_all.append(data)
        data_list_nontrain.append(data)
    #change here

    ood_y_dim1 = len(dataset_ood[0][0].y)
    ood_y_dim2 = len(dataset_ood[0][0].y[0])
    print(ood_y_dim1,ood_y_dim2)
    reshape = 0
    if y_dim1 != ood_y_dim1 or ood_y_dim2 != y_dim2:
        reshape = 1
    #change here
    for data in dataset_ood:
        data[0].edge_attr = None
        data[1].edge_attr = None
        data[0].edge_index = data[0].edge_index.to(torch.long)
        data[1].edge_index = data[1].edge_index.to(torch.long)
        if reshape:
            y_filled = torch.zeros((y_dim1,y_dim2))
            # use mean to fill the y in ood data
            for i in range(y_dim1):
                for j in range(y_dim2):
                    y_filled[i][j] = torch.mean(torch.tensor(data[0].y,dtype=torch.float64))
            for i in range(min(y_dim1,ood_y_dim1)):
                for j in range(min(y_dim2,ood_y_dim2)):
                    y_filled[i][j] = data[0].y[i][j]
            data[0].y = y_filled
            data[1].y = y_filled
        
        data_list_test.append(data)
        data_list_all.append(data)
        data_list_nontrain.append(data)
    print('here4444444444444444444444444-------------------------')
    return dataset_train, data_list_valid, data_list_test, data_list_all, data_list_nontrain, dataset_num_features


def get_ad_split_TU(args, fold=5):
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)

    dataset = TUDataset(path, name=DS)
    data_list = []
    label_list = []

    for data in dataset:
        data_list.append(data)
        label_list.append(data.y.item())

    kfd = StratifiedKFold(n_splits=fold, random_state=0, shuffle=True)

    splits = []
    for k, (train_index, test_index) in enumerate(kfd.split(data_list, label_list)):
        splits.append((train_index, test_index))

    return splits


def get_ad_dataset_TU(args, split, need_str_enc=True):
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)

    if DS in ['IMDB-BINARY', 'REDDIT-BINARY', 'COLLAB']:
        dataset = TUDataset(path, name=DS, transform=(Constant(1, cat=False)))
    else:
        dataset = TUDataset(path, name=DS)

    dataset_num_features = dataset.num_node_features

    data_list = []
    label_list = []

    for data in dataset:
        data.edge_attr = None
        data_list.append(data)
        label_list.append(data.y.item())

    if need_str_enc:
        data_list = init_structural_encoding(data_list, rw_dim=args.rw_dim, dg_dim=args.dg_dim)

    (train_index, test_index) = split
    data_train_ = [data_list[i] for i in train_index]
    data_test = [data_list[i] for i in test_index]

    data_train = []
    for data in data_train_:
        if data.y != 0:
            data_train.append(data)

    idx = 0
    for data in data_train:
        data.y = 0
        data['idx'] = idx
        idx += 1

    for data in data_test:
        data.y = 1 if data.y == 0 else 0

    dataloader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size_test, shuffle=True)
    meta = {'num_feat':dataset_num_features, 'num_train':len(data_train)}

    return dataloader, dataloader_test, dataset_num_features


def get_ad_dataset_Tox21(args, need_str_enc=True):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data')

    data_train_ = read_graph_file(args.DS + '_training', path)
    data_test = read_graph_file(args.DS + '_testing', path)

    dataset_num_features = data_train_[0].num_features

    data_train = []
    for data in data_train_:
        if data.y == 1:
            data_train.append(data)

    idx = 0
    for data in data_train:
        data.y = 0
        data['idx'] = idx
        idx += 1

    for data in data_test:
        data.y = 1 if data.y == 1 else 0

    if need_str_enc:
        data_train = init_structural_encoding(data_train, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
        data_test = init_structural_encoding(data_test, rw_dim=args.rw_dim, dg_dim=args.dg_dim)

    dataloader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size_test, shuffle=True)
    meta = {'num_feat':dataset_num_features, 'num_train':len(data_train)}

    return dataloader, dataloader_test, dataset_num_features