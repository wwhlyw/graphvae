from torch.utils.data import DataLoader, Dataset
import numpy as np
import os



class Data(Dataset):
    def __init__(self, file_paths):
        self.node_feats_files = file_paths[0]
        self.edge1_feats_files = file_paths[1]
        self.edge2_feats_files = file_paths[2]
        self.edge3_feats_files = file_paths[3]
        self.node_feats = self._read_data(self.node_feats_files)
        self.edge1_feats = self._read_data(self.edge1_feats_files)
        self.edge2_feats = self._read_data(self.edge2_feats_files)
        self.edge3_feats = self._read_data(self.edge3_feats_files)
        self.n_node = self.node_feats.shape[1]
        self.node_features_dim = self.node_feats.shape[2]

    def _read_data(self, files):
        datas = []
        for file in os.listdir(files):
            datas.append(np.load(os.path.join(files, file)))
        datas = np.concatenate(datas, axis=0)
        datas = (datas - np.min(datas)) / (np.max(datas) - np.min(datas))
        
        return datas

    def __getitem__(self, idx):
        # return self.node_feats, np.concatenate(self.edge1_feats[idx], self.edge2_feats[idx], self.edge3_feats[idx], -1)
        return self.node_feats[idx].astype(np.float32), \
            np.concatenate([
                self.edge1_feats[idx], 
                self.edge2_feats[idx],
                self.edge3_feats[idx]], axis=-1).astype(np.float32), \
            self.node_feats[idx+1].astype(np.float32),\
            np.concatenate([
                self.edge1_feats[idx+1],
                self.edge2_feats[idx+1],
                self.edge3_feats[idx+1]],axis=-1).astype(np.float32)
    
    def __len__(self):
        return self.edge3_feats.shape[0] - 1

