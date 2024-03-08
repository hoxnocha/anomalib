from typing import Any, List, Sequence, Callable, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
#from lightning.pytorch import LightningModule
from torchmetrics import MeanMetric, Accuracy, AUROC, MetricCollection
from torchmetrics.functional import f1_score

#from clip_model.data.airogs import AirogsDataModule
import open_clip
from open_clip import create_model_and_transforms,get_tokenizer
import tqdm
import json
#import logging
from .airogs_winclip_prompts import (
     TEMPLATE_LEVEL_PROMPTS,
     STATE_LEVEL_NORMAL_PROMPTS,
     STATE_LEVEL_ABNORMAL_PROMPTS,
)
# debug
from omegaconf import DictConfig, ListConfig
from .torch_model import CLIPModel
import ipdb


PATH_TO_PROMPTS = "/home/students/tyang/clip_model/prompts/airogs_gpt_prompts.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from anomalib.models.components import AnomalyModule
from anomalib.models.clip.torch_model import CLIPModel

class Clip(AnomalyModule):
    def __init__(
            self,
            backbone: str = "ViT-B-16-plus-240",
            pretrained: str = "laion400m_e32",
            object: str = "retina",
            zero_shot: bool = False,
            text_prompt_type: str = "gpt",
            k_shot: int = 0,
            classifier_method: str = "PCA",

            
            sampling_ratio: float = 0.2,
    ):
        
        super().__init__()
        self.model = CLIPModel(

            model_name=backbone,
            pretrained=pretrained,
            object=object,
            zero_shot=zero_shot,
            text_prompt_type=text_prompt_type,
            classifier_method=classifier_method,

        )

        self.k_shot = k_shot
        self.reference_images : List[torch.Tensor] = []
        self.ref_image_embeddings: List[torch.Tensor] = []
        self.sampling_ratio = sampling_ratio

    def configure_optimizers(self,):
   
            return None
        
    
    def training_step(self, batch: dict[str, str| torch.Tensor], *args, **kwargs):

            del args, kwargs

            image_embeddings = self.model(batch["image"])
            self.ref_image_embeddings.append(image_embeddings)


    

    def on_validation_start(self) -> None:
         
         ref_image_embedding_tensor = torch.cat(self.ref_image_embeddings, dim=0)
         if self.k_shot != 0:
                ref_image_embedding_tensor = ref_image_embedding_tensor[torch.randperm(ref_image_embedding_tensor.size(0))[ : self.k_shot]]

         self.model.subsample_embedding(ref_image_embedding_tensor, sampling_ratio=self.sampling_ratio)
         

    def validation_step(self, batch: dict[str, str| torch.Tensor], *args, **kwargs):
        del args, kwargs
        
        output = self.model(batch["image"])
        batch["pred_scores"] = output["pred_scores"]

        return batch

  

class ClipLightning(Clip):

    def __init__(self, hparams):
        super().__init__(

            backbone=hparams.model.backbone,
            pretrained=hparams.model.pretrained,
            object=hparams.model.object,
            zero_shot=hparams.model.zero_shot,
            text_prompt_type=hparams.model.text_prompt_type,
            k_shot=hparams.model.k_shot,
            sampling_ratio=hparams.model.sampling_ratio,
            classifier_method=hparams.model.classifier_method,

        )

        self.hparams: DictConfig | ListConfig
        self.save_hyperparameters(hparams)
    
        


        

    




                              






