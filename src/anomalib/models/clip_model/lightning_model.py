from typing import Any, List, Sequence, Callable, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from lightning.pytorch import LightningModule
from torchmetrics import MeanMetric, Accuracy, AUROC, MetricCollection
from torchmetrics.functional import f1_score

from clip_model.data.airogs import AirogsDataModule
import open_clip
from open_clip import create_model_and_transforms,get_tokenizer
import tqdm
import json
from .airogs_winclip_prompts import (
     TEMPLATE_LEVEL_PROMPTS,
     STATE_LEVEL_NORMAL_PROMPTS,
     STATE_LEVEL_ABNORMAL_PROMPTS,
)
# debug
import ipdb


PATH_TO_PROMPTS = "/home/students/tyang/clip_model/prompts/airogs_gpt_prompts.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from anomalib.models.components import AnomalyModule
from anomalib.models.clip_model.torch_model import CLIPModel

class Clip(AnomalyModule):
    def __init__(
            self,
            backbone: str = "ViT-B-16-plus-240",
            pretrained: str = "laion400m_e32",
            object: str = "retina",
            zero_shot: bool = False,
            text_prompt_type: str = "gpt",
            k_shot: int = 5,
    ):
        
        super().__init__()
        self.model = CLIPModel(

            model_name=backbone,
            pretrained=pretrained,
            object=object,
            zero_shot=zero_shot,
            text_prompt_type=text_prompt_type,
            k_shot=k_shot,
        )

        self.k_shot = k_shot
        self.reference_images : List[torch.Tensor] = []

    def configure_optimizers(self,):
            #optimizer = torch.optim.Adam(self.model.parameters(), lr=4e-6)
            return None
        
    
    def training_step(self, batch: dict[str, str| torch.Tensor], *args, **kwargs):

            del args, kwargs
            x = batch["image"]
            self.reference_images.append(x)


    

    def on_validation_start(self) -> None:
         ref_img_tensor = torch.cat(self.reference_images, dim=0)
         ref_img_tensor = ref_img_tensor[torch.randperm(ref_img_tensor.size(0))[ : self.k_shot]]
         

       
            



class ClipLightning(Clip):

    def __init__(self, hparams):
        super().__init__(

        )
    
        


        

    




                              






