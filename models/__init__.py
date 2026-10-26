"""Paper-aligned model components for cross-domain graph adaptation."""
from models.backbones import BaseGCN, infer_backbone_dimensions, load_pretrained_backbone, load_pretrained_gcn
from models.graphmae_gat_backbone import GraphMAEGATBackbone
from models.dual_branch_framework import DualBranchFramework
from models.prompt_graph_construction import PromptGraphConstruction
from models.prompting_adaptation import PromptingAdaptation
from models.prompting_weight_learning import PromptingWeightLearning
from models.prompt_representation_residual import PromptRepresentationResidual
from models.prompt_pool import PromptPoolSelector
from models.representation_adaptation import ResidualRepresentationAdapter

__all__ = ['BaseGCN', 'GraphMAEGATBackbone', 'DualBranchFramework', 'PromptGraphConstruction',
           'PromptingAdaptation', 'PromptingWeightLearning', 'PromptRepresentationResidual',
           'PromptPoolSelector', 'ResidualRepresentationAdapter', 'infer_backbone_dimensions',
           'load_pretrained_backbone', 'load_pretrained_gcn']
