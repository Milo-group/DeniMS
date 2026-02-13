import graph_tool as gt
import os
import pathlib
import warnings
import sys

import time
import torch

# Add parent directory to sys.path to allow 'src' module imports when loading checkpoints
_ms_diffusion_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ms_diffusion_dir not in sys.path:
    sys.path.insert(0, _ms_diffusion_dir)

# Patch torch.load to use weights_only=False by default for PyTorch 2.6+ compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

import utils
from diffusion_model_ms import DiscreteEdgesDenoisingDiffusion
from diffusion.extra_features import ExtraFeatures
from diffusion.extra_features_molecular import ExtraMolecularFeatures

from datasets import ms_dataset
from metrics.molecular_metrics import SamplingMolecularMetricsEdges 
from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscreteEdges
from analysis.visualization import MolecularVisualization

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def _generate_names(cfg: DictConfig, overrides=None):
    load_subdata_dir = getattr(cfg.conditioning, 'load_subdata_dir', None)
    if load_subdata_dir is None or load_subdata_dir == "" or str(load_subdata_dir).lower() == "null":
        ms_data_path = getattr(cfg.conditioning, 'ms_data_path', None)
        embeddings_type = getattr(cfg.conditioning, 'embeddings_type', None)
        splitting_path = getattr(cfg.conditioning, 'splitting_path', None)
        
        if ms_data_path and embeddings_type:
            generated_name = utils.build_load_subdata_dir_name(
                ms_data_path, embeddings_type, splitting_path
            )
            with open_dict(cfg.conditioning):
                cfg.conditioning.load_subdata_dir = generated_name
            print(f"Auto-generated load_subdata_dir: {generated_name}")
    
    generated_name = utils.auto_generate_general_name(cfg, overrides=overrides)
    if generated_name is not None:
        print(f"Auto-generated general.name: {generated_name}")


def _main_impl(cfg: DictConfig):
    
    dataset_config = cfg["dataset"]

    datamodule = ms_dataset.MSDataModule_lmdb(cfg)
    
    dataset_infos = ms_dataset.MSinfos(datamodule=datamodule, cfg=cfg)
    train_smiles = ms_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                        dataset_infos=dataset_infos, evaluate_dataset=False, source = True)

    extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)

    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos, embeddings = True)
    
    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                            domain_features=domain_features, embeddings = True)

    train_metrics = TrainMolecularMetricsDiscreteEdges(dataset_infos)

    sampling_metrics = SamplingMolecularMetricsEdges(dataset_infos, train_smiles)
    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features}

    utils.create_folders(cfg)
    model = DiscreteEdgesDenoisingDiffusion(cfg=cfg, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/E_logp',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="ddp_find_unused_parameters_true",
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=False,
                      callbacks=callbacks,
                      log_every_n_steps=50 if cfg.general.name != 'debug' else 1,
                      num_sanity_val_steps=-1,
                      limit_val_batches=4,
                      logger = [])

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)

    else:
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':

    config_path = os.path.join(os.path.dirname(__file__), '../configs')
    config_name = 'config'
    
    overrides = sys.argv[1:] if len(sys.argv) > 1 else []
    
    with initialize_config_dir(config_dir=config_path, version_base='1.3'):
        cfg = compose(config_name=config_name, overrides=overrides)
        _generate_names(cfg)
        
        new_overrides = overrides.copy()
        
        load_subdata_dir = getattr(cfg.conditioning, 'load_subdata_dir', None)
        if load_subdata_dir and not any('conditioning.load_subdata_dir' in o for o in overrides):
            new_overrides.append(f'conditioning.load_subdata_dir="{load_subdata_dir}"')
        
        general_name = getattr(cfg.general, 'name', None)
        if general_name and not any('general.name' in o for o in overrides):
            new_overrides.append(f'general.name="{general_name}"')
    
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]] + new_overrides
    
    main = hydra.main(version_base='1.3', config_path=config_path, config_name=config_name)(_main_impl)
    main()
