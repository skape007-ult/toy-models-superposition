"""Reproduction of Elhage et al. 2022, "Toy Models of Superposition"."""
from .model import Config, ToyModel, generate_batch, weighted_mse
from .train import train, get_device
