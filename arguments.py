import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--DS', dest='DS', help='Dataset')
    parser.add_argument('--local', dest='local', action='store_const', 
            const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const', 
            const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const', 
            const=True, default=False)

    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
            help='')

    parser.add_argument('--aug', type=str, default='dnodes')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--amplr', type=float, default=2e-2)
    parser.add_argument('--bal1', type=float, default=10)
    parser.add_argument('--bal2', type=float, default=10)
    parser.add_argument('--ratio', type=float, default=0.4)
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    parser.add_argument('--lofk', type=int, default=5)
    
    parser.add_argument('-DS_pair', default=None)
    parser.add_argument('-rw_dim', type=int, default=16)
    parser.add_argument('-dg_dim', type=int, default=16)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-batch_size_test', type=int, default=9999)

    return parser.parse_args()

