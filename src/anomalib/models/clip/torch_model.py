from typing import Any, List, Sequence, Callable, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchmetrics import MeanMetric, Accuracy, AUROC, MetricCollection
from torchmetrics.functional import f1_score
import numpy as np

#from clip_model.data.airogs import AirogsDataModule
import open_clip
from open_clip import create_model_and_transforms,get_tokenizer
import faiss
#import tqdm
import json
from .airogs_winclip_prompts import (
     TEMPLATE_LEVEL_PROMPTS,
     STATE_LEVEL_NORMAL_PROMPTS,
     STATE_LEVEL_ABNORMAL_PROMPTS,
)
# debug
import ipdb

from anomalib.models.components import  DynamicBufferModule, KCenterGreedy



PATH_TO_PROMPTS = "/home/students/tyang/anomalib/src/anomalib/models/clip/airogs_gpt_prompts.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CLIPModel(DynamicBufferModule, nn.Module):
    def __init__(self, 
                 model_name: str = "ViT-B-16-plus-240",
                 pretrained: str = "laion400m_e32",
                 
                 object: str = "retina", # text descriptor for the object to be classified, "fundus", for 2 stage classification consider "optic disc"
                 
                 zero_shot: bool = False,
                 text_prompt_type: str = "gpt", # "standard" or "gpt". gpt is for gpt4 generated prompts
                 #k_shot: int = 5,
                 #classifier_method: str = "PCA", # "PCA" or "min_max"
                 
                 ):
        super().__init__()
        
       
        self.clip_model, _, _ = create_model_and_transforms(model_name, pretrained)
        self.tokenizer = get_tokenizer(model_name)
        

        self.object = object
  
        self.zero_shot = zero_shot
        self.text_prompt_type = text_prompt_type
        #self.k_shot = k_shot
        #self.classifier_method = classifier_method
        self.register_buffer("memory_bank", torch.Tensor())
        self.memory_bank: torch.Tensor


    def forward(self, image: torch.Tensor):
        """return embedding during training or similiarity score during testing"""
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        batch_size,_ = image_features.shape

        if self.training:
            output = image_features
        
        else:
            if self.zero_shot: # zero shot learning, calculate similarity score with text features
                text_features = self.build_text_classifier()
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                output = {"pred_scores": text_probs[:, 0]}
 
            
            else :
                pred_scores = self.computing_anomaly_score(image_features, self.memory_bank, )
                output = {"pred_scores": pred_scores}
            
            




       
        return output

                

    
    def subsample_embedding(self, embedding: torch.Tensor, sampling_ratio: float, ) -> None:
        """Subsample embedding based on coreset sampling and store to memory.

        Args:
            embedding (np.ndarray): Embedding tensor from the CNN
            sampling_ratio (float): Coreset sampling ratio
        """
        # Coreset Subsampling
          # methode == "min_max":
        
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
        coreset = sampler.sample_coreset()
        self.memory_bank = coreset
        
        #else: # PCA
            
    
    
    def computing_anomaly_score(self, embedding: torch.Tensor, memory_bank: torch.Tensor) :    
        """Nearest Neighbours using brute force method and cosine norm.

        Args:
            embedding (Tensor): Features to compare the distance with the memory bank.
            

        Returns:
            Tensor: anomaly scores.
            
        """
        input_norm = F.normalize(embedding, dim=-1)
        memory_norm = F.normalize(memory_bank, dim=-1)

        cos_similarities = input_norm @ memory_norm.T
        max_cos_similarities = torch.max(cos_similarities, dim=-1).values

        anomoly_scores = 1 - max_cos_similarities

        return anomoly_scores


    
    @torch.no_grad()
    def build_text_classifier(self,):
        
     
        def _process_template(state_level_templates):
            text = []
            for template in TEMPLATE_LEVEL_PROMPTS:
                for state_template in state_level_templates:
                    text.append(template(state_template(self.object)))
            
            texts = self.tokenizer(text).to(device=device)
            class_embeddings = self.clip_model.encode_text(texts)
            mean_class_embeddings = torch.mean(class_embeddings, dim=0, keepdim=True)
            mean_class_embeddings = F.normalize(mean_class_embeddings, dim=-1)
            return mean_class_embeddings
        
        if self.text_prompt_type == "standard":


            RG_text_embedding = _process_template(STATE_LEVEL_ABNORMAL_PROMPTS)
            NRG_text_embedding = _process_template(STATE_LEVEL_NORMAL_PROMPTS)

            
            return torch.cat([RG_text_embedding, NRG_text_embedding], dim=0)
        
        else:# gpt
            with open(PATH_TO_PROMPTS, "r") as f:
                gpt_prompts = json.load(f)

                text_weights = []

                for class_name in gpt_prompts.keys():
                    texts = []
                    for prompt in gpt_prompts[class_name]:
                        texts.append(prompt)
                    
                    texts = self.tokenizer(texts).to(device=device)
                    class_embeddings = self.clip_model.encode_text(texts)
                    mean_class_embeddings = torch.mean(class_embeddings, dim=0, keepdim=True)
                    mean_class_embeddings = F.normalize(mean_class_embeddings, dim=-1)
                    text_weights.append(mean_class_embeddings)
                
                return torch.cat(text_weights, dim=0)
            
#



            

        
            
        

        
            
""" @torch.no_grad()
    def image_encoder(self, x, normalize=True):
        
        image encoder, last layer of vision transformer resblocks will be replaced
        by CSA (Correlative Self-Attention) model
        
        vit = self.clip_model.visual.eval()
        w, h = x.shape[-2:]
        x = SCLIPADModel.get_conv1(
            vit=vit, patch_size=self.patch_size, stride=self.patch_size)(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        # class embedding and positional embeddings
        x = torch.cat(
            [vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        if x.shape[1] != vit.positional_embedding.shape[0]:
            x = x + self.interpolate_pos_encoding(x, w, h)
        else:
            x = x + vit.positional_embedding.to(x.dtype)
        x = vit.patch_dropout(x)
        x = vit.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        resblocks = vit.transformer.resblocks

        for block in resblocks[:self.feature_layer]:
            x = block(x) 

         Applying Self Attention Module in the last attention block 
        for block in resblocks[self.feature_layer:]:
            if self.isCSA:
                csa_x = x + csa_attn(
                    block, 
                    block.ln_1(x), 
                    istimm=False, 
                    attn_logit_scale=self.attn_logit_scale,
                )
                csa_x = csa_x + block.mlp(block.ln_2(csa_x))

                if hasattr(self, "split_layers") and "visual_11" in self.split_layers:
                    split_block = self.split_layers["visual_11"]
                    split_x = x + csa_attn(
                        split_block, 
                        split_block.ln_1(x), 
                        istimm=self.istimm, 
                        attn_logit_scale=self.attn_logit_scale
                    ) 
                    split_x = split_x + split_block.mlp(split_block.ln_2(split_x))
                if self.clsCSA:
                    x[0:1, ...] = csa_x[0:1, ...]
                else:
                    
                    x[0:1, ...] = block(x)[0:1, ...]

                if hasattr(self, "split_layers") and "visual_11" in self.split_layers:
                    x[1:, ...] = split_x[1:, ...]
                else:
                    x[1:, ...] = csa_x[1:, ...]
            else:
                x = block(x)

        x = x.permute(1, 0, 2)
        x = vit.ln_post(x) @ vit.proj

        
        if normalize:
            x = F.normalize(x, dim=-1, p=2)
        return {
            "cls": x[:, 0, :],
            "tokens": x[:, 1:, :]
        }"""

    



        
        








                

        
