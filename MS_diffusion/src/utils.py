import os
import pathlib
import torch_geometric.utils
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
import omegaconf
import wandb


def build_load_subdata_dir_name(ms_data_path, embeddings_type, splitting_path):
    ms_path = pathlib.Path(ms_data_path)
    dataset_name = ms_path.parent.name if ms_path.parent.name else ms_path.stem
    
    if splitting_path:
        split_path = pathlib.Path(splitting_path)
        split_filename = split_path.stem
        if "_" in split_filename:
            splitting_type = split_filename.rsplit("_", 1)[-1]
        else:
            splitting_type = split_filename
    else:
        splitting_type = "none"
    
    embedding_type = embeddings_type if embeddings_type else "null"
    
    return f"{dataset_name}_{embedding_type}_{splitting_type}"


def auto_generate_general_name(cfg, overrides=None):
    if cfg.general.name is not None and cfg.general.name != "":
        return None
    
    name_parts = []
    if cfg.general.test_only is not None and cfg.general.test_only != "":
        name_parts.append("test")
    else:
        name_parts.append("train")
    
    embeddings_type = None
    ms_data_path = None
    splitting_path = None
    resume = None
    
    if overrides:
        for override in overrides:
            if '=' in override:
                key, value = override.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                
                if key == 'conditioning.embeddings_type':
                    embeddings_type = value
                elif key == 'conditioning.ms_data_path':
                    ms_data_path = value
                elif key == 'conditioning.splitting_path':
                    splitting_path = value
                elif key == 'general.resume':
                    resume = value
    
    if embeddings_type is None:
        try:
            embeddings_type = cfg.conditioning.embeddings_type
        except:
            embeddings_type = None
    
    if ms_data_path is None:
        try:
            ms_data_path = cfg.conditioning.ms_data_path
        except:
            ms_data_path = None
    
    if splitting_path is None:
        try:
            splitting_path = cfg.conditioning.splitting_path
        except:
            splitting_path = None
    
    if resume is None:
        try:
            resume = cfg.general.resume
        except:
            resume = None
    
    if ms_data_path:
        ms_path = pathlib.Path(ms_data_path)
        dataset_name = ms_path.parent.name if ms_path.parent.name else ms_path.stem
        name_parts.append(dataset_name)
    else:
        name_parts.append("null")
    
    if embeddings_type:
        name_parts.append(str(embeddings_type))
    else:
        name_parts.append("null")
    
    if splitting_path:
        split_path = pathlib.Path(splitting_path)
        split_filename = split_path.stem
        if "_" in split_filename:
            splitting_type = split_filename.rsplit("_", 1)[-1]
        else:
            splitting_type = split_filename
        name_parts.append(splitting_type)
    else:
        name_parts.append("none")
    
    if resume is not None and str(resume).strip() != "" and str(resume).lower() != "null":
        name_parts.append("resume")
    
    generated_name = "_".join(name_parts)
    with open_dict(cfg.general):
        cfg.general.name = generated_name
    
    return generated_name


def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs('graphs')
        os.makedirs('chains')
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs('graphs/' + args.general.name)
        os.makedirs('chains/' + args.general.name)
    except OSError:
        pass


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def to_dense(x, edge_index, edge_attr, batch, atom_attr = None):

    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)
    
    if atom_attr == None:
        return PlaceHolder(X=X, E=E, y=None), node_mask
    
    else:
        atom_attr, _ = to_dense_batch(x=atom_attr, batch=batch)
        return PlaceHolder(X=X, atom_attr=atom_attr, E=E, y=None), node_mask



def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg


class PlaceHolder:
    def __init__(self, X, E, y, atom_attr=None, smiles = None):
        self.X = X
        self.E = E
        self.y = y
        self.atom_attr = atom_attr
        self.smiles = smiles

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        if self.atom_attr != None:
            self.atom_attr = self.atom_attr.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        
        if self.atom_attr != None:
            if collapse:
                self.charges = torch.argmax(self.atom_attr[:, :, :9], dim=-1)
                self.charges[node_mask == 0] = - 1
                self.Hs = torch.argmax(self.atom_attr[:, :, 9:], dim=-1)
                self.Hs[node_mask == 0] = - 1
            
            else:
                self.charges = self.atom_attr[:, :, :9] * x_mask
                self.Hs = self.atom_attr[:, :, 9:] * x_mask
        
        return self


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': cfg.general.wandb_project, 'config': config_dict,
              'reinit': True, 'mode': cfg.general.wandb}
    wandb.login(key = cfg.general.wandb_api_key)
    wandb.init(**kwargs)
    wandb.save('*.txt')