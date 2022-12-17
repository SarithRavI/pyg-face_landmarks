import os.path as osp
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_undirected, add_self_loops,from_scipy_sparse_matrix
from torch_sparse import coalesce
from torch_geometric.io import read_txt_array

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.image import img_to_graph

def split(data, batch):
    """
    PyG util code to create graph batches
    """
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    # Edge indices should start at zero for every graph.
    data.__num_nodes__ = torch.bincount(batch).tolist()

    slices = {'edge_index': []}
    if data.x is not None:
        slices['x'] = node_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    return data, slices

def read_graph_data(folder):
    # import images  

    # get the format of the file inside the folder
    # as a 'convention' .pkl file contains a list of images -> (# of images, height, width, channels)
    # and .npz contains a map with key 'face_images' -> (height,width,channels, # images)
    # do not contain files with both format types

    fmt = [file[-3:] for file in os.listdir(folder) if (file[-3:]=='pkl' or file[-3:]=='npz')]
    if fmt is 'npz':
        images = np.load(os.path.join(folder,"face_images.npz"))['face_images']
        tot_shape =images.shape
        if  len(tot_shape) > 3: # this is for RGB
            num_node_attr = tot_shape[2]
        else:
            num_node_attr = 1
        num_images = len(images)
    elif fmt is 'pkl':
        with open(os.path.join(folder,"face_images.pkl"), "rb") as fp:   # Unpickling
            images = pickle.load(fp)
        tot_shape =images.shape
        if  len(tot_shape) > 3: # this is for RGB
            num_node_attr = tot_shape[3]
        else:
            num_node_attr = 1
        num_images = tot_shape[-1]
    
    # import graph labels
    graph_labels = pd.read_csv(os.path.join(folder,"facial_keypoints.csv")).to_numpy()

    # by default number of edge attributes are 1
    num_edge_attr = 1

    num_labels = graph_labels.shape[-1]
    x = np.empty([0,num_node_attr])
    edge_index = np.empty([2,0])
    edge_attr = np.empty([0,num_edge_attr],dtype=np.float32)
    node_graph_id=np.array([],dtype=np.int32)
    edge_slice = np.array([0])
    y = np.empty([0,num_labels],dtype=np.float32)

    for img_inx in tqdm(range(num_images)):
        if fmt is 'pkl':
            img_matrix = images[img_inx]
        elif fmt is 'npz':
            if len(tot_shape)>3:
                img_matrix = images[:,:,:,img_inx]
            else:
                img_matrix = images[:,:,img_inx]

        img_graph = img_to_graph(img = img_matrix)
        img_to_pyg = from_scipy_sparse_matrix(img_graph)

        img_edge_index = img_to_pyg[0]
        # since we are getting the edge attributes from pyg 
        # we set the # of edge attributes to 1 by default
        img_edge_attr = img_to_pyg[1].reshape(1,-1)
            
        img_node_attr = img_matrix.reshape(-1,num_node_attr)
        x = np.vstack((x,img_node_attr))
        num_nodes = img_node_attr.shape[0] # number of nodes 
        # append node_graph_id 
        graph_inx = img_inx
        node_graph_id = np.append(node_graph_id,[graph_inx]*num_nodes)
        # add image edge_index
        edge_index = np.hstack((edge_index,img_edge_index))
        # add image edge attr
        edge_attr = np.vstack((edge_attr,img_edge_attr))
                
        # edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)
            
        edge_slice = np.append(edge_slice,edge_slice[-1]+img_edge_index[0].shape[-1])
        y = np.vstack((y,graph_labels[graph_inx].reshape(1,-1)))

    # converting to torch array 
    edge_index = torch.from_numpy(edge_index).to(torch.int64)
    edge_attr = torch.from_numpy(edge_attr).to(torch.float32)
    x = torch.from_numpy(x).to(torch.float32)
    y = torch.from_numpy(y).to(torch.float32)
    # print(node_graph_id)
    node_graph_id = torch.from_numpy(node_graph_id)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data, slices = split(data, node_graph_id)

    slices['edge_index']=torch.from_numpy(edge_slice).to(torch.int34)
    if data.edge_attr is not None:
        slices['edge_attr'] = torch.from_numpy(edge_slice).to(torch.int34)
    
    return data, slices

class PygGraphPropPredDataset(InMemoryDataset):

    def __init__(self,root,name,transform=None, pre_transform=None,pre_filter=None):
        self.name = name
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        super(PygGraphPropPredDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    def download(self):
        pass
    @property
    def raw_dir(self):
        name = "raw/"
        return osp.join(self.root,self.name,name)

    @property
    def processed_dir(self):
        name = 'processed/'
        return osp.join(self.root,self.name,name)

    @property
    def num_node_attributes(self):
        pass 
    
    
    def num_landmarks(self):
        return self.data.y.size(1)

    @property
    def raw_file_names(self):
        names = ["face_images.npz","facial_keypoints.csv"]
        return names

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):

        self.data,self.slices = read_graph_data(self.raw_dir)
        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)
            
        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        print('Saving...')
        torch.save((self.data, self.slices), self.processed_paths[0])

