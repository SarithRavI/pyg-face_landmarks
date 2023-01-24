import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from conv import GNN_node, GNN_node_Virtualnode

from torch_scatter import scatter_mean

class GNN(torch.nn.Module):

    def __init__(self, num_landmarks, num_layer = 5,num_fnn_layers =1, input_node_dim=1,input_edge_dim = 1, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_landmarks = num_landmarks
        self.num_fnn_layers = num_fnn_layers
        self.graph_pooling = graph_pooling
        self.input_node_dim = input_node_dim
        self.input_edge_dim = input_edge_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer,input_node_dim=self.input_node_dim,input_edge_dim = self.input_edge_dim, emb_dim=emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        self.preprocess_linear_list = torch.nn.ModuleList()

        self.graph_pred_linear_list = torch.nn.ModuleList()

        # if graph_pooling == "set2set":
        #     for i in range(max_seq_len):
        #          self.graph_pred_linear_list.append(torch.nn.Linear(2*emb_dim, self.num_vocab))

        if graph_pooling == "set2set":   
            self.graph_pred_linear_list.append(torch.nn.Linear(2*emb_dim, self.num_landmarks))   
        
        # else:
        #     for i in range(max_seq_len):
        #          self.graph_pred_linear_list.append(torch.nn.Linear(emb_dim, self.num_vocab))
        else:
            for i in range(num_fnn_layers-1):
                self.graph_pred_linear_list.append(torch.nn.Linear(emb_dim, emb_dim))
            self.graph_pred_linear_list.append(torch.nn.Linear(emb_dim, self.num_landmarks))
                


    def forward(self, batched_data):
        '''
            Return:
                A list of predictions.
                i-th element represents prediction at i-th position of the sequence.
        '''
        # input_dim=batched_data.x.size(0)
        
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        output = h_graph # initial input is set to the output of the GNN 
        for fnn_inx in range(self.num_fnn_layers):
            output = self.graph_pred_linear_list[fnn_inx](output)
            
        # pred_list = self.graph_pred_linear_list[i](h_graph)
        # for i in range(self.max_seq_len):
        #     pred_list.append(self.graph_pred_linear_list[i](h_graph))
        # return pred_list

        return output 

if __name__ == '__main__':
    pass