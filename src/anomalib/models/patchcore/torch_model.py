"""PyTorch model for the PatchCore model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import antialiased_cnns
import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from antialiased_cnns.blurpool import BlurPool
from timm.models.registry import list_models as list_models_timm
from torchvision.models import get_model_builder, get_model_weights
from torchvision.models import list_models as list_models_torch
from torchvision.models.feature_extraction import create_feature_extractor
from anomalib.models.components.filters import GaussianBlur2d
import torchvision.transforms as transforms

from anomalib.models.components import DynamicBufferModule, FeatureExtractor, KCenterGreedy

from anomalib.models.patchcore.anomaly_map import AnomalyMapGenerator
from anomalib.pre_processing import Tiler


MODELS_TIMM = list_models_timm()
MODELS_TORCH = list_models_torch()


class PatchcoreModel(DynamicBufferModule, nn.Module):
    """Patchcore Module."""

    def __init__(
        self,
        input_size: tuple[int, int],
        layers: list[str],
        backbone: str = "wide_resnet50_2",
        pre_trained: bool = True,
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        self.backbone = backbone
        self.layers = layers
        self.input_size = input_size
        self.num_neighbors = num_neighbors
        
        if self.backbone in MODELS_TORCH:
            print(f"Loading model {self.backbone} from torchvision")
            _model_builder = get_model_builder(self.backbone)
            _model_weights = get_model_weights(self.backbone)
            model = _model_builder(weights=_model_weights.DEFAULT)
        elif self.backbone in MODELS_TIMM:
            print(f"Loading model {self.backbone} from timm")
            model = timm.create_model(self.backbone, pretrained=True)

        # loading anti-aliased models from antialiased-cnns
        elif self.backbone == "antialiased_wide_resnet50_2" or backbone == "antialiased_wide_resnet50_2_384":
            print(f"Loading model {self.backbone} from antialias_cnn")
            model = antialiased_cnns.wide_resnet50_2(pretrained=True)
        elif self.backbone == "antialiased_wide_resnet101_2":
            print(f"Loading model {self.backbone} from antialias_cnn")
            model = antialiased_cnns.wide_resnet101_2(pretrained=True)
        elif self.backbone == "antialiased_resnet18":
            print(f"Loading model {self.backbone} from antialias_cnn")
            model = antialiased_cnns.resnet18(pretrained=True)
        elif self.backbone == "antialiased_resnet50":
            print(f"Loading model {self.backbone} from antialias_cnn")
            model = antialiased_cnns.resnet50(pretrained=True)
        else:
            raise ValueError(f"Model {self.backbone} not found")
        
        self.feature_extractor = create_feature_extractor(
            model=model,
            return_nodes={layer: layer for layer in self.layers},
            tracer_kwargs={"leaf_modules": [BlurPool]},  # for models comes from antialias
        )
        self.feature_extractor.eval()
        
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

        self.register_buffer("memory_bank", Tensor())
        self.memory_bank: Tensor

    def forward(self, input_tensor: Tensor) -> Tensor | dict[str, Tensor]:
        """Return Embedding during training, or a tuple of anomaly map and anomaly score during testing.

        Steps performed:
        1. Get features from a CNN.
        2. Generate embedding based on the features.
        3. Compute anomaly map in test mode.

        Args:
            input_tensor (Tensor): Input tensor

        Returns:
            Tensor | dict[str, Tensor]: Embedding for training,
                anomaly map and anomaly score for testing.
        """
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        embedding = self.generate_embedding(features)

        if self.tiler:
            embedding = self.tiler.untile(embedding)

        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)

        if self.training:
            output = embedding
        else:
            # apply nearest neighbor search
            patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1)
            # reshape to batch dimension
            patch_scores = patch_scores.reshape((batch_size, -1))
            locations = locations.reshape((batch_size, -1))
            # compute anomaly score
            pred_score = self.compute_anomaly_score(patch_scores, locations, embedding)
            # reshape to w, h
            patch_scores = patch_scores.reshape((batch_size, 1, width, height))
            # get anomaly map
            anomaly_map = self.anomaly_map_generator(patch_scores)

            output = {"anomaly_map": anomaly_map, "pred_score": pred_score}

        return output

    def generate_embedding(self, features: dict[str, Tensor]) -> Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features: Hierarchical feature map from a CNN (ResNet18 or WideResnet)
            features: dict[str:Tensor]:

        Returns:
            Embedding vector
        """

        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="bilinear")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        return embeddings

    @staticmethod
    def reshape_embedding(embedding: Tensor) -> Tensor:
        """Reshape Embedding.

        Reshapes Embedding to the following format:
        [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

        Args:
            embedding (Tensor): Embedding tensor extracted from CNN features.

        Returns:
            Tensor: Reshaped embedding tensor.
        """
        embedding_size = embedding.size(1)
        embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)
        return embedding
        
    def subsample_embedding(self, embedding: Tensor, sampling_ratio: float) -> None:
        """Subsample embedding based on coreset sampling and store to memory.

        Args:
            embedding (np.ndarray): Embedding tensor from the CNN
            sampling_ratio (float): Coreset sampling ratio
        """
        # Coreset Subsampling
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
        coreset = sampler.sample_coreset()
        self.memory_bank = coreset
    #def kl_divergence_filter(self, feature, batch):
    # feature_tensor = feature["adaptive_avg_pool2d"]
    # tensor_size = feature_tensor.shape[0]
    # kl_divergences = torch.zeros((tensor_size, tensor_size))
    ## for i in range(tensor_size):
     # for j in range(i+1, tensor_size):
      #  feature_i = feature_tensor[i].view(-1)
      #  feature_j = feature_tensor[j].view(-1)

      ##  kl_divergences[i,j] = F.kl_div(F.softmax(feature_i,dim=0).log(), F.softmax(feature_j,dim=0))
       # kl_divergences[j,i] = F.kl_div(F.softmax(feature_j,dim=0).log(), F.softmax(feature_i,dim=0))
    
    # average_kl_divergences = torch.mean(kl_divergences, dim=1)
    # most_unique_idx = torch.argmax(average_kl_divergences)
    # most_unique_data = batch[most_unique_idx]

     #return most_unique_data
    
    #def kl_divengence_subsample(self, embedding: Tensor):
        #import ipdb; ipdb.set_trace()
       # tensor_size = embedding.shape[0]
       # kl_divergences = torch.zeros((tensor_size, tensor_size))
       # for i in range(tensor_size):
         ## for j in range(i+1, tensor_size):
          ## feature_i = embedding[i].view(-1)
          # feature_j = embedding[j].view(-1)

          # kl_divergences[i,j] = F.kl_div(F.softmax(feature_i,dim=0).log(), F.softmax(feature_j,dim=0))
          # kl_divergences[j,i] = F.kl_div(F.softmax(feature_j,dim=0).log(), F.softmax(feature_i,dim=0))
    
        #average_kl_divergences = torch.mean(kl_divergences, dim=1)
       # most_unique_idx = torch.argmax(average_kl_divergences)
       # self.memory_bank = embedding[0][most_unique_idx]

    
    #def kmeans_embedding(self, embedding, num_clusters: int) :
       # """Subsample embedding based on coreset sampling and store to memory.

       # Args:
            #embedding (np.ndarray): Embedding tensor from the CNN
            #sampling_ratio (float): Coreset sampling ratio
        #"""
        # Coreset Subsampling
        #kmeans = KMeans(n_clusters=num_clusters)
        #kmeans.fit(embedding)
        #import ipdb; ipdb.set_trace()
        #clusters_centers = kmeans.cluster_centers_
        #labels = kmeans.labels_
        #distances = [torch.cdist(embedding[i].unsqueeze(1), clusters_centers[labels[i]].unsqueeze(1)) for i in range(len(embedding))] 
        #unique_embedding = [embedding[i] for i in range(len(embedding)) if distances[i] > 0.5]
        #return unique_embedding

    #def find_least_similiar_embeddings(self, embedding, num_bins:int) -> list[Tensor]:
        #"""Find the most special embeddings in the memory bank.

        #Args:
            #embedding (list[Tensor]): Embedding tensor from the CNN
            #num_bins (int): Number of bins to divide the embedding space into

        #Returns:
           # list[Tensor]: Embeddings that are most special
       # """
        #embedding = torch.vstack(embedding)
        #hist, bin_edges = torch.histc(embedding, bins=num_bins)
        #least_similiar_bins = torch.argmin(hist)
        #least_similiar_embedding = []
        #for i in range(len(self.embedding)):
            #if bin_edges[i] <= least_similiar_bins[i] <= bin_edges[i+1]:    
                #least_similiar_embedding.append(self.embedding[i])
        
        #least_similiar_embedding = torch.stack(least_similiar_embedding)
        #return least_similiar_embedding
    
    #def get_unique_embedding(self, embedding) :
        
        #distances = torch.mm(embedding, embedding.t())
        #unique_embedding = embedding[distances.max(dim=1) == distances.max(dim=1)[0]]
        #return unique_embedding

    #def get_unfimiliar_embedding(self, embedding) :
        #import ipdb; ipdb.set_trace()
        #dist_matrix = torch.mm(embedding ** 2, torch.ones(embedding.shape[1], embedding.shape[0])) + torch.mm(embedding ** 2, torch.ones(embedding.shape[1], embedding.shape[0])).t() - 2 * torch.mm(embedding, embedding.t())
        #max_idx = torch.argmax(dist_matrix, dim=1)  
        #unfimiliar_embedding = embedding[max_idx]
        #return unfimiliar_embedding
    def crop_disc(self, input_tensor: Tensor, ):
        """Crop the disc from the input tensor.

        Args:
            input_tensor (Tensor): Input tensor

        Returns:
            Tensor: Cropped disc

        """
        #import ipdb; ipdb.set_trace()
        batch_size, channels, height, width = input_tensor.shape
        crop_size = 240
        gray = transforms.Grayscale(num_output_channels=1)(input_tensor)
        gaussian_gray = transforms.functional.gaussian_blur(gray, kernel_size=23, sigma=1.5)
        
        max_values,max_indices = torch.max(gaussian_gray.view(gaussian_gray.shape[0],-1),dim=1)

        max_coordinates = torch.stack([max_indices % gaussian_gray.shape[-1], max_indices // gaussian_gray.shape[-1]], dim=1)
        cropped_images = []
        for i in range(batch_size):
          start_y = max(0, int(max_coordinates[i, 1]) - crop_size // 2)
          end_y = min(int(max_coordinates[i, 1]) + crop_size // 2, height)
          start_x = max(0, int(max_coordinates[i, 0]) - crop_size // 2)
          end_x = min(int(max_coordinates[i, 0]) + crop_size // 2, width)
    
          cropped_image = input_tensor[i, : ,start_y:end_y ,start_x:end_x,]
          cropped_images.append(cropped_image)
        

        max_height = max(t.shape[1] for t in cropped_images)
        max_width = max(t.shape[2] for t in cropped_images)
        padded_tensors = [F.pad(t, (0, max_width - t.shape[2], 0, max_height - t.shape[1])) for t in cropped_images]
        cropped_images_t = torch.stack(padded_tensors)
        cropped_images_t = cropped_images_t.to(dtype=torch.float32)
        return cropped_images_t
        
    def crop_disc_2(self, input_tensor: Tensor, ):
        """Crop the disc from the input tensor.

        Args:
            input_tensor (Tensor): Input tensor

        Returns:
            Tensor: Cropped disc

        """
        
        #import ipdb; ipdb.set_trace()

        batch_size, channels, height, width = input_tensor.shape
        crop_size = 240
        gray = transforms.Grayscale(num_output_channels=1)(input_tensor)
        blurred = transforms.functional.gaussian_blur(gray, kernel_size=23, sigma=1.5)
        
        # Use adaptive thresholding to create a binary mask
        thresholded = torch.where(blurred > blurred.mean(), torch.tensor(1.0), torch.tensor(0.0))

         # Find contours using Sobel operator
        sobel_x = F.conv2d(thresholded, torch.cuda.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3), padding=1)
        sobel_y = F.conv2d(thresholded, torch.cuda.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3), padding=1)
        gradient_magnitude = torch.sqrt(sobel_x**2 + sobel_y**2)

          # Use non-maximum suppression to get thin edges
        non_max_suppressed = gradient_magnitude * (gradient_magnitude == F.max_pool2d(gradient_magnitude, kernel_size=3, stride=1, padding=1))

       # Apply a threshold to get a binary edge map
        edge_map = torch.where(non_max_suppressed > non_max_suppressed.mean(), torch.tensor(1.0), torch.tensor(0.0))

        result_tensor = blurred * (1 - edge_map)

        
        max_values,max_indices = torch.max(result_tensor.view(result_tensor.shape[0],-1),dim=1)

        max_coordinates = torch.stack([max_indices % result_tensor.shape[-1], max_indices // result_tensor.shape[-1]], dim=1)
        cropped_images = []
        for i in range(batch_size):
          start_y = max(0, int(max_coordinates[i, 1]) - crop_size // 2)
          end_y = min(int(max_coordinates[i, 1]) + crop_size // 2, height)
          start_x = max(0, int(max_coordinates[i, 0]) - crop_size // 2)
          end_x = min(int(max_coordinates[i, 0]) + crop_size // 2, width)
    
          cropped_image = input_tensor[i, : ,start_y:end_y ,start_x:end_x,]
          cropped_images.append(cropped_image)
        

        filtered_tensors = [tensor for tensor in cropped_images if tensor.shape[1] > crop_size -90 and tensor.shape[2] > crop_size - 90]
        max_height = max(t.shape[1] for t in cropped_images)
        max_width = max(t.shape[2] for t in cropped_images)
        padded_tensors = [F.pad(t, (0, max_width - t.shape[2], 0, max_height - t.shape[1])) for t in filtered_tensors]
        stacked_tensor = torch.stack(padded_tensors)
        return stacked_tensor
        
        

    @staticmethod
    def euclidean_dist(x: Tensor, y: Tensor) -> Tensor:
        """
        Calculates pair-wise distance between row vectors in x and those in y.

        Replaces torch cdist with p=2, as cdist is not properly exported to onnx and openvino format.
        Resulting matrix is indexed by x vectors in rows and y vectors in columns.

        Args:
            x: input tensor 1
            y: input tensor 2

        Returns:
            Matrix of distances between row vectors in x and y.
        """
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)  # |x|
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)  # |y|
        # row distance can be rewritten as sqrt(|x| - 2 * x @ y.T + |y|.T)
        res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
        res = res.clamp_min_(0).sqrt_()
        return res

    def nearest_neighbors(self, embedding: Tensor, n_neighbors: int) -> tuple[Tensor, Tensor]:
        """Nearest Neighbours using brute force method and euclidean norm.

        Args:
            embedding (Tensor): Features to compare the distance with the memory bank.
            n_neighbors (int): Number of neighbors to look at

        Returns:
            Tensor: Patch scores.
            Tensor: Locations of the nearest neighbor(s).
        """
        distances = self.euclidean_dist(embedding, self.memory_bank)
        if n_neighbors == 1:
            # when n_neighbors is 1, speed up computation by using min instead of topk
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores, locations

    def compute_anomaly_score(self, patch_scores: Tensor, locations: Tensor, embedding: Tensor) -> Tensor:
        """Compute Image-Level Anomaly Score.

        Args:
            patch_scores (Tensor): Patch-level anomaly scores
            locations: Memory bank locations of the nearest neighbor for each patch location
            embedding: The feature embeddings that generated the patch scores
        Returns:
            Tensor: Image-level anomaly scores
        """

        # Don't need to compute weights if num_neighbors is 1
        if self.num_neighbors == 1:
            return patch_scores.amax(1)
        batch_size, num_patches = patch_scores.shape
        # 1. Find the patch with the largest distance to it's nearest neighbor in each image
        max_patches = torch.argmax(patch_scores, dim=1)  # indices of m^test,* in the paper
        # m^test,* in the paper
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]
        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        score = patch_scores[torch.arange(batch_size), max_patches]  # s^* in the paper
        nn_index = locations[torch.arange(batch_size), max_patches]  # indices of m^* in the paper
        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = self.memory_bank[nn_index, :]  # m^* in the paper
        # indices of N_b(m^*) in the paper
        memory_bank_effective_size = self.memory_bank.shape[0]  # edge case when memory bank is too small
        _, support_samples = self.nearest_neighbors(
            nn_sample, n_neighbors=min(self.num_neighbors, memory_bank_effective_size)
        )
        # 4. Find the distance of the patch features to each of the support samples
        distances = self.euclidean_dist(max_patches_features.unsqueeze(1), self.memory_bank[support_samples])
        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        # 6. Apply the weight factor to the score
        score = weights * score  # s in the paper
        return score
