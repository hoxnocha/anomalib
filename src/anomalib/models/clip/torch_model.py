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
from sklearn.decomposition import PCA
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
                 
                 classifier_method: str = "PCA", # "PCA" or "vanilla"
                 
                 ):
        super().__init__()
        
       
        self.clip_model, _, _ = create_model_and_transforms(model_name, pretrained)
        self.tokenizer = get_tokenizer(model_name)
        

        self.object = object
  
        self.zero_shot = zero_shot
        self.text_prompt_type = text_prompt_type
        
        self.classifier_method = classifier_method
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
                pred_scores = self.computing_anomaly_score(image_features, self.memory_bank, self.classifier_method )
                output = {"pred_scores": pred_scores}
            
            
       
        return output

                

    
    def subsample_embedding(self, embedding: torch.Tensor, sampling_ratio: float, ) -> None:
        """Subsample embedding based on coreset sampling and store to memory.

        Args:
            embedding (np.ndarray): Embedding tensor from the CNN
            sampling_ratio (float): Coreset sampling ratio
        """
        
        if sampling_ratio == 1.0:
            self.memory_bank = embedding

        else:
        
         sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
         coreset = sampler.sample_coreset()
         self.memory_bank = coreset
        
        
            
    
    
    def computing_anomaly_score(self, embedding: torch.Tensor, memory_bank: torch.Tensor, classifier_method: str) -> torch.Tensor:    
        """Nearest Neighbours using brute force method and cosine norm.

        Args:
            embedding (Tensor): Features to compare the distance with the memory bank.
            

        Returns:
            Tensor: anomaly scores.
            
        """
        input_norm = F.normalize(embedding, dim=-1)
        memory_norm = F.normalize(memory_bank, dim=-1)

        if classifier_method == "PCA":
            pca = PCA(n_components=0.95)
            
            input_np = input_norm.cpu().numpy()
            memory_np = memory_norm.cpu().numpy()

            input_pca = pca.fit_transform(input_np)
            memory_pca = pca.transform(memory_np)

            input_norm = torch.tensor(input_pca).to(device=device)
            memory_norm = torch.tensor(memory_pca).to(device=device)

            cos_similarities = input_norm @ memory_norm.T

            max_cos_similarities = torch.max(cos_similarities, dim=-1).values

            anomoly_scores = 1 - max_cos_similarities

        else:
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
            



        
        








                

        
