# 1. Imports and constants
import argparse
import os
from typing import Dict, List, Tuple

import pandas as pd
import torch
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.strategies import DDPStrategy
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from graphnet.data.dataloader import DataLoader
from graphnet.data.datamodule import GraphNeTDataModule
from graphnet.data.dataset import SQLiteDataset
from graphnet.data.utilities.sqlite_utilities import query_database
from graphnet.datasets.nubench_datasets import TRUTH_NUBENCH
from graphnet.models import StandardModel
from graphnet.models.data_representation import (
    GraphDefinition,
    PercentileClusters,
)
from graphnet.models.data_representation.graphs import KNNEdges
from graphnet.models.detector.nubench import FlowerL
from graphnet.models.gnn import DynEdge
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.training.loss_functions import LogCoshLoss

#constants
FEATURES = ['sensor_pos_x', 'sensor_pos_y', 'sensor_pos_z', 'charge', 't']
TRUTH = TRUTH_NUBENCH
PULSEMAP_TRAIN = "merged_photons"
PULSEMAP_TEST = "pulses_no_noise"
TRUTH_TABLE = "mc_truth"
STAGES = {
    "fast": {"n_events": 1_000_000, "scheduler_patience": 2, "early_stopping_patience": 10},
    "full": {"n_events": None, "scheduler_patience": 6, "early_stopping_patience": 20},
    }
TEST_BATCH_SIZE = 500


def make_balanced_selection(data_dir: str, seed: int) -> pd.DataFrame:
    """Return a table of event numbers with equally many tracks and cascades.

    Args:
        data_dir: Path to the directory containing the SQLite files.
        seed: Random seed for reproducibility.
    Returns:
        A pandas DataFrame containing the balanced selection of events.
    """
    output_path = os.path.join(data_dir, "selections", f"training_subselection_seed{seed}.parquet")
    # If the balanced selection already exists, load it from disk
    if os.path.isfile(output_path):
        return pd.read_parquet(output_path)

    # select events from the training selection
    train_selection = pd.read_parquet(os.path.join(data_dir, "selections", "train_selection.parquet"))
    train_events = train_selection["event_no"].values
    db_path = os.path.join(data_dir, "merged", "merged.db")
    truth = query_database(db_path, f"SELECT event_no, interaction, initial_state_type FROM {TRUTH_TABLE}")
    truth = truth[truth["event_no"].isin(train_events)]

    # splitting track and cascade events based on interaction and initial_state_type
    is_track = (truth["interaction"] == 1) & (truth["initial_state_type"].abs() == 14)
    tracks = truth[is_track]
    cascades = truth[~is_track]

    # balance number of track and cascade events
    n = min(len(tracks), len(cascades))
    selection = pd.concat([
        tracks.sample(n=n, random_state=seed),
        cascades.sample(n=n, random_state=seed)]).sample(
            frac=1, random_state=seed).reset_index(drop=True)
    selection.to_parquet(output_path, index=False)
    return selection

def build_graph(augment: bool) -> GraphDefinition:
    node_definition = PercentileClusters(
        cluster_on=["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"],
        percentiles=[0, 10, 50, 90, 100],
        input_feature_names=FEATURES)
    edge_definition = KNNEdges(nb_nearest_neighbours=8)
    graph_definition = GraphDefinition(
        detector=FlowerL(),
        node_definition=node_definition,
        edge_definition=edge_definition,
        input_feature_names=FEATURES,
        perturbation_dict = {'t': 1, 'charge': 0.25} if augment else None,
    )
    return graph_definition

def build_train_val_loaders(graph_definition: GraphDefinition, event_list: list, db_path: str,
                            batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    dataset_reference = SQLiteDataset
    dataset_args = {"truth_table": TRUTH_TABLE,
                    "pulsemaps": PULSEMAP_TRAIN,
                    "truth": TRUTH,
                    "features": FEATURES,
                    "path": db_path,
                    "data_representation": graph_definition}
    train_dataloader_kwargs = {"batch_size": batch_size,
                               "num_workers": num_workers,
                               "shuffle": True,
                               "persistent_workers": True,
                               "pin_memory": True,
                               }
    dm = GraphNeTDataModule(dataset_reference=dataset_reference,
                                     dataset_args=dataset_args,
                                     selection=event_list,
                                     train_dataloader_kwargs=train_dataloader_kwargs)
    return dm.train_dataloader, dm.val_dataloader


def build_test_loader(graph_definition: GraphDefinition, event_list: list, db_path: str,
                      batch_size: int, num_workers: int) -> DataLoader:
    dataset_reference = SQLiteDataset
    dataset_args = {"truth_table": TRUTH_TABLE,
                    "pulsemaps": PULSEMAP_TEST,
                    "truth": TRUTH,
                    "features": FEATURES,
                    "path": db_path,
                    "data_representation": graph_definition}
    test_dataloader_kwargs = {"batch_size": batch_size,
                              "num_workers": num_workers,
                              "shuffle": False,
                              "pin_memory": True,
                              }
    dm = GraphNeTDataModule(dataset_reference=dataset_reference,
                                     dataset_args=dataset_args,
                                     test_selection=event_list,
                                     test_dataloader_kwargs=test_dataloader_kwargs)
    return dm.test_dataloader

def build_model(graph: GraphDefinition, scheduler_patience: int) -> StandardModel:
    backbone = DynEdge(
        nb_inputs=graph.nb_outputs,
        global_pooling_schemes=["min", "max","mean"], # Why no sum?
        )
    task = EnergyReconstruction(
        hidden_size=backbone.nb_outputs,
        target_labels="initial_state_energy",
        loss_function=LogCoshLoss(),
        transform_prediction_and_target=lambda x: torch.log10(x),
        transform_inference=lambda x: torch.pow(10, x),
        )
    model = StandardModel(
        data_representation=graph,
        backbone=backbone,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-3,
                           "eps": 1e-8},
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={
            "patience": scheduler_patience,
        },
        scheduler_config={
            "frequency": 1,
            "monitor": "val_loss",
        }
    )
    return model

def main(args: argparse.Namespace) -> None:
    # Settings and paths
    settings = STAGES[args.stage]
    db_path = os.path.join(args.data_dir, "merged", "merged.db")
    run_dir = os.path.join(args.output_dir, args.stage, "nubench_arca_dynedge")
    os.makedirs(run_dir, exist_ok=True)

    # event lists
    n_events = args.n_events if args.n_events is not None else settings["n_events"]
    selection = make_balanced_selection(args.data_dir, args.seed)
    train_events = selection["event_no"].tolist()[:n_events]
    test_selection = pd.read_parquet(os.path.join(args.data_dir, "selections", "test_selection.parquet"))
    test_events = test_selection["event_no"].tolist()

    # build graph and dataloaders for training and testing
    train_graph = build_graph(augment=True)
    train_loader, val_loader = build_train_val_loaders(train_graph, train_events, db_path,
                                                      args.batch_size, args.num_workers)
    test_graph = build_graph(augment=False)
    test_loader = build_test_loader(test_graph, test_events, db_path, TEST_BATCH_SIZE, args.num_workers)

    # build model and output names
    model = build_model(train_graph, scheduler_patience=settings["scheduler_patience"])
    prefix = "DynEdge_training_arca_{stage}_initial_state_energy"
    name = prefix.format(stage=args.stage)

    # weights for "full" from fast
    if args.stage == "full":
        fast_dir = os.path.join(args.output_dir, "fast", "nubench_arca_dynedge")
        fast_weights = os.path.join(fast_dir, prefix.format(stage="fast") + "_state_dict.pth")
        model.load_state_dict(fast_weights)

    # training and saving of model
    model.fit(
        train_loader,
        val_loader,
        max_epochs=args.max_epochs,
        early_stopping_patience=settings["early_stopping_patience"],
        gpus=[args.gpu],
        distribution_strategy=DDPStrategy(process_group_backend="gloo"),
        logger=CSVLogger(save_dir=os.path.join(run_dir, "logs")),
        plugins=[LightningEnvironment()],
        ckpt_path=args.ckpt_path,
        )
    model.save_state_dict(os.path.join(run_dir, f"{name}_state_dict.pth"))
    model.save_config(os.path.join(run_dir, f"{name}_model_config.yml"))
    results = model.predict_as_dataframe(
        test_loader,
        additional_attributes=TRUTH + ["event_no"],
        gpus =[args.gpu],
        plugins =[LightningEnvironment()],
    ).set_index("event_no")
    results.to_parquet(os.path.join(run_dir, f"{name}_test_results.parquet"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain DynEdge for energy on Flower L.")
    parser.add_argument("--stage", choices=list(STAGES), required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1500)
    parser.add_argument("--n-events", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt-path", default=None)
    main(parser.parse_args())