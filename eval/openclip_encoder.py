#!/usr/bin/env python
import torch
import torchvision
import open_clip
import torch.nn.functional as F
import torch.nn as nn

class SimpleFeatureEnhancer:
    def __init__(self, feature_dim):
        self.weight = torch.nn.Parameter(torch.randn(feature_dim, feature_dim).cuda())
        self.bias = torch.nn.Parameter(torch.randn(feature_dim).cuda())

    def enhance(self, features):
        # 简单的线性变换和非线性激活
        enhanced_features = F.relu(torch.matmul(features, self.weight) + self.bias)
        return enhanced_features
        
class OpenCLIPNetwork:
    def __init__(self, device):
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.clip_model_type = "ViT-B-16"
        self.clip_model_pretrained = 'laion2b_s34b_b88k'
        self.clip_n_dims = 512
        model, _, _ = open_clip.create_model_and_transforms(
            self.clip_model_type,
            pretrained=self.clip_model_pretrained,
            precision="fp16",
        )
        model.eval()
        
        self.tokenizer = open_clip.get_tokenizer(self.clip_model_type)
        self.model = model.to(device)
        
        # object
        self.negatives = ("object", "things", "stuff", "texture")
        self.positives = (" ",)
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to(device) # tokenlization
            self.pos_embeds = model.encode_text(tok_phrases)    # convert it into embeddings
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to(device)
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)
        
    @torch.no_grad()
    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        # embed shape: (HxW, 512)
        # embed: 32768x512
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0) #  combine pos and neg embeddings
        p = phrases_embeds.to(embed.dtype)
        output = torch.mm(embed, p.T) # matrix mul
        positive_vals = output[..., positive_id : positive_id + 1] # select the corresponding positive output
        negative_vals = output[..., len(self.positives) :] # all remainings are negative output
        repeated_pos = positive_vals.repeat(1, len(self.negatives)) # why repeat here? #QUESTION #FIXME

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)
        softmax = torch.softmax(10 * sims, dim=-1) # sharpen the results
        best_id = softmax[..., 0].argmin(dim=1)
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]
    
    # @torch.no_grad()
    # # this is with negative embeddings
    # def get_3d_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
    #     """
    #     Compute the relevancy of the embedding to the positive and negative text embeddings.
    #     :param embed: Tensor of shape (N, 512), where N is the number of Gaussian points.
    #     :param positive_id: Index of the positive embedding to compare.
    #     :return: Relevancy tensor of shape (N, 2).
    #     """
    #     phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
    #     p = phrases_embeds.to(embed.dtype)
    #     # print("phrases_embeds shape", phrases_embeds.shape)

    #     embed_norm = embed / (embed.norm(dim=-1, keepdim=True) + 1e-9)

    #     # feature_enhancer = SimpleFeatureEnhancer(512)
    #     # embed_norm = feature_enhancer.enhance(embed_norm)
    #     # Compute similarities
    #     output = torch.mm(embed_norm, p.T)
         
    #     # Extract positive and negative similarities
    #     positive_vals = output[..., positive_id : positive_id + 1]
    #     negative_vals = output[..., len(self.positives) :]
        
    #     # Stack positive and negative similarities for softmax calculation
    #     sims = torch.cat((positive_vals, negative_vals), dim=-1)
        
    #     repeated_pos = positive_vals.repeat(1, len(self.negatives)) # why repeat here? #QUESTION #FIXME

    #     sims = torch.stack((repeated_pos, negative_vals), dim=-1)
    #     softmax = torch.softmax(10*sims, dim=-1) # sharpen the results
    #     best_id = softmax[..., 0].argmin(dim=1)
        
    #     # Gather the results based on the best_id
    #     # return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives) + 1, 2))[:, 0, :]
    #             # Print intermediate results for debugging
    #     # print("Best ID shape:", best_id.shape)
    #     # print("Softmax shape:", softmax.shape)
    #     # print("Best ID expanded shape:", best_id[..., None, None].expand(best_id.shape[0], len(self.neg_embeds) + 1, 2).shape)
        
    #     # # Gather the results based on the best_id
    #     # result = torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.neg_embeds) + 1, 2))[:, 0, :]
    #     # print("Result shape:", result.shape)
        
    #     # return result
        
    #             # Apply softmax to sharpen the results
    #     # softmax = torch.softmax(sims, dim=-1) # sharpen the results
    #     # best_id = softmax[..., 0].argmin(dim=1)
        
    #     # Gather the results based on the best_id
    #     # return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives) + 1, 2))[:, 0, :]
    #             # Print intermediate results for debugging
    #     #print("Best ID shape:", best_id.shape)
    #     #print("Softmax shape:", softmax.shape)
    #     #print("Best ID expanded shape:", best_id[..., None, None].expand(best_id.shape[0], len(self.neg_embeds) + 1, 2).shape)
        
    #     # Gather the results based on the best_id
    #     result = torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.neg_embeds) + 1, 2))[:, 0, :]
    #     #print("Result shape:", result.shape)
        
    #     return result
    
    


    

    
    @torch.no_grad()
    def get_3d_relevancy(self, embed: torch.Tensor, positive_id: int, query_dim) -> torch.Tensor:
        """
        Compute the relevancy of the embedding to the positive text embeddings.
        :param embed: Tensor of shape (N, 512), where N is the number of Gaussian points.
        :param positive_id: Index of the positive embedding to compare.
        :return: Relevancy tensor of shape (N, 1).
        """
        # Ensure the positive embeddings are in the same data type as the embed
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0) #  combine pos and neg embeddings
        p = phrases_embeds.to(embed.dtype)
        print("self.pos_embeds shape", self.pos_embeds.shape)
        print("p shape", p.shape)
        
        # Compute similarities
        output = torch.mm(embed, p.T)
        
        print("output", output.shape)

        if query_dim == -1: # represent the pos query sim
            positive_vals = output[..., positive_id : positive_id + 1]
            return positive_vals
        else:
            negative_vals = output[..., len(self.positives) :]
            return negative_vals[..., query_dim].unsqueeze(1)
    

    def encode_image(self, input, mask=None):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input, mask=mask)

    def encode_text(self, text_list, device):
        text = self.tokenizer(text_list).to(device)
        return self.model.encode_text(text)
    
    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.positives]
                ).to(self.neg_embeds.device)
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
    
    def set_semantics(self, text_list):
        self.semantic_labels = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.semantic_labels]).to("cuda")
            self.semantic_embeds = self.model.encode_text(tok_phrases)
        self.semantic_embeds /= self.semantic_embeds.norm(dim=-1, keepdim=True)
    
    def get_semantic_map(self, sem_map: torch.Tensor) -> torch.Tensor:
        # embed: 3xhxwx512
        n_levels, h, w, c = sem_map.shape
        pos_num = self.semantic_embeds.shape[0]
        phrases_embeds = torch.cat([self.semantic_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(sem_map.dtype)
        sem_pred = torch.zeros(n_levels, h, w)
        for i in range(n_levels):
            output = torch.mm(sem_map[i].view(-1, c), p.T)
            softmax = torch.softmax(10 * output, dim=-1)
            sem_pred[i] = torch.argmax(softmax, dim=-1).view(h, w)
            sem_pred[i][sem_pred[i] >= pos_num] = -1
        return sem_pred.long()


    def get_max_across(self, sem_map):
        # sem shape - (lvl, h, w, 512)        
        n_phrases = len(self.positives)
        n_phrases_sims = [None for _ in range(n_phrases)]
        
        n_levels, h, w, _ = sem_map.shape
        clip_output = sem_map.permute(1, 2, 0, 3).flatten(0, 1) # (HxW, 3, 512)

        n_levels_sims = [None for _ in range(n_levels)]
        for i in range(n_levels):
            for j in range(n_phrases):
                probs = self.get_relevancy(clip_output[..., i, :], j) # cal each feature level
                pos_prob = probs[..., 0:1]
                n_phrases_sims[j] = pos_prob
            n_levels_sims[i] = torch.stack(n_phrases_sims)
        
        relev_map = torch.stack(n_levels_sims).view(n_levels, n_phrases, h, w)
        return relev_map
    
    
    def query_3d_gaussian(self, gaussian_features: torch.Tensor, query_dim) -> torch.Tensor:
        """
        Compute the similarity between 3D Gaussian points and text queries and activate based on a threshold.
        :param gaussian_features: Tensor of shape (level, N, 512), where level is the number of levels, 
                                  N is the number of 3D Gaussian points, and 512 is the feature dimension.
        :return: Tensor indicating which Gaussian points are activated.
        """

        print("self.positives", self.positives)
        n_phrases = len(self.positives)  # Number of positive phrases
        n_levels = gaussian_features.shape[0]  # Number of levels
        n_points = gaussian_features.shape[1]  # Number of 3D Gaussian points

        relevancies = []
        for i in range(n_levels):
            level_relevancies = []
            for j in range(n_phrases):
                relevancy = self.get_3d_relevancy(gaussian_features[i], j, query_dim=query_dim)
                level_relevancies.append(relevancy)
            relevancies.append(torch.stack(level_relevancies, dim=1))  # Shape: (N, n_phrases, 2)
        
        relev_map = torch.stack(relevancies, dim=0)  # Shape: (level, N, n_phrases, 2)
        
        # Apply threshold to determine which Gaussian points are activated
        activated_map = relev_map[..., 0] # Shape: (level, N, n_phrases)
        
        return activated_map