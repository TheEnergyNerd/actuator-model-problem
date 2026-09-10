"""Installable Isaac Lab extension: pip install -e isaac/exts/atlas_actuators
(from the motordesign repo checkout — the envelope tier loads its canonical
implementation from isaac/atlas_actuator.py by relative path)."""
from setuptools import setup, find_packages

setup(
    name="atlas-actuators",
    version="0.1.0",
    description="Physics-grounded actuator models (envelope + FOC tiers) for "
                "NVIDIA Isaac Lab, parameterized from Atlas motor designs.",
    author="Pranav Myana + Claude",
    packages=find_packages(),
    install_requires=[],          # torch/isaaclab come from the Isaac Lab env
    python_requires=">=3.10",
    classifiers=["Programming Language :: Python :: 3.10"],
)
