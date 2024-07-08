#!/usr/bin/env python
import torch
import torchvision
import open_clip
import numpy as np

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

        self.negatives = ("object", "things", "stuff", "texture")
        self.positives = (" ",)
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to(device)
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to(device)
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def get_relevancy(self, positive_id: int, scene_index) -> torch.Tensor:
        # # embed: 32768x512
        # phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        # p = phrases_embeds.to(embed.dtype)
        # output = torch.mm(embed, p.T)
        # positive_vals = output[..., positive_id : positive_id + 1]
        # # positive_vals = np.load('/datadrive/yingwei/LangSplat_new/dataset/lerf_ovs/teatime/output/teatime_3_pos/train/stuffed bear/ours_None/renders_npy/00001.npy')
        # negative_vals = output[..., len(self.positives) :]
        # repeated_pos = positive_vals.repeat(1, len(self.negatives))

        # sims = torch.stack((repeated_pos, negative_vals), dim=-1)
        # print("this is the sim shape", sims.shape)
        # softmax = torch.softmax(10 * sims, dim=-1)
        
        # print("this is the softmax shape", softmax.shape)
        # best_id = softmax[..., 0].argmin(dim=1)
        # return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
        #     :, 0, :
        # ]
        """
        Compute the relevancy of the embedding to the positive text embeddings.
        :param embed: Tensor of shape (N, 512), where N is the number of Gaussian points.
        :param positive_id: Index of the positive embedding to compare.
        :return: Relevancy tensor of shape (N, 1).
        """
        # ***** prev version *****
        # scene_lst_int = [1,24,42,106,128,139]
        # scene_lst = ['00001', '00024','00042','00106','00128','00139']
        
        # positives_lst = [
        #     ['stuffed bear', 'coffee mug', 'bag of cookies', 'sheep', 'apple', 'paper napkin', 'plate', 'tea in a glass', 'bear nose', 'three cookies', 'coffee'],
        #     ['stuffed bear', 'sheep', 'bag of cookies', 'tea in a glass', 'coffee mug', 'plate', 'three cookies', 'hooves', 'paper napkin', 'coffee', 'bear nose'],
        #     ['tea in a glass', 'hooves', 'stuffed bear', 'bag of cookies', 'paper napkin', 'plate', 'apple', 'coffee mug', 'coffee', 'three cookies'],
        #     ['stuffed bear', 'sheep', 'apple', 'bag of cookies', 'coffee mug', 'tea in a glass', 'bear nose', 'plate', 'three cookies', 'dall-e brand', 'paper napkin'],
        #     ['tea in a glass', 'apple', 'yellow pouf', 'sheep', 'three cookies', 'plate', 'dall-e brand'],
        #     ['tea in a glass', 'paper napkin', 'apple', 'stuffed bear', 'bag of cookies', 'plate', 'coffee mug', 'coffee', 'three cookies']
        # ]
        # ***** pre version *****
        
        scene_lst_int = [5, 23, 59, 64, 80, 118, 127]
        scene_lst = [f'{num:05d}' for num in scene_lst_int]
        positives_lst = [
['chopsticks', 'egg', 'nori', 'bowl', 'napkin', 'sake cup', 'wavy noodles', 'kamaboko', 'plate', 'onion segments'],
['bowl', 'chopsticks', 'egg', 'nori', 'wavy noodles', 'kamaboko', 'onion segments', 'corn'],
['chopsticks', 'egg', 'sake cup', 'napkin', 'wavy noodles', 'kamaboko', 'corn', 'onion segments'],
['bowl', 'egg', 'chopsticks', 'sake cup', 'wavy noodles', 'nori', 'napkin', 'kamaboko', 'plate', 'corn', 'onion segments'],
['bowl', 'chopsticks', 'sake cup', 'nori', 'egg', 'wavy noodles', 'glass of water', 'kamaboko', 'spoon', 'napkin', 'plate', 'onion segments'],
['nori', 'egg', 'sake cup', 'chopsticks', 'wavy noodles', 'kamaboko', 'corn', 'onion segments'],
['glass of water', 'nori', 'egg', 'sake cup', 'bowl', 'chopsticks', 'wavy noodles', 'spoon', 'corn', 'onion segments', 'hand', 'plate', 'kamaboko', 'napkin']]


                    
        # Map positive_id to the corresponding file path
        level = 1
        case_name = "ramen"
        positive_name = positives_lst[scene_index][positive_id]
        # positive_path = f"/datadrive/yingwei/LangSplat_prev_generate_maps/dataset/lerf_ovs/teatime/output/teatime_{level}_pos/train/scene_{scene_lst_int[scene_index]+1}/{positive_name}/ours_None/renders_npy/{scene_lst[scene_index]}.npy"
        positive_path = f"/datadrive/yingwei/LangSplat_prev_generate_maps/dataset/lerf_ovs/{case_name}/output/{case_name}_{level}/train/scene_{scene_lst_int[scene_index]}/{positive_name}/ours_None/renders_npy/{scene_lst[scene_index]}.npy"
    
        all_values = np.load(positive_path)
        
        # print("positive_vals shape", positive_vals.shape)
        # assert False

        # Extract positive values (first dimension) and negative values (remaining dimensions)
        positive_vals = all_values[:, :,0:1].reshape(-1, 1)
        negative_vals = all_values[:, :,1:].reshape(-1, all_values.shape[-1] - 1)
        

        # Repeat positive values for comparison
        repeated_pos = np.repeat(positive_vals, negative_vals.shape[1], axis=1)
        
        # Convert numpy arrays to PyTorch tensors
        repeated_pos_tensor = torch.tensor(repeated_pos, dtype=torch.float32)
        negative_vals_tensor = torch.tensor(negative_vals, dtype=torch.float32)

        # ****** recover ******
        # negative_paths = {
        #     "object": f"/datadrive/yingwei/LangSplat_prev_generate_maps/dataset/lerf_ovs/teatime/output/teatime_{level}_negative/object/renders_npy/{scene_lst[scene_index]}.npy",
        #     "things": f"/datadrive/yingwei/LangSplat_prev_generate_maps/dataset/lerf_ovs/teatime/output/teatime_{level}_negative/things/renders_npy/{scene_lst[scene_index]}.npy",
        #     "stuff": f"/datadrive/yingwei/LangSplat_prev_generate_maps/dataset/lerf_ovs/teatime/output/teatime_{level}_negative/stuff/renders_npy/{scene_lst[scene_index]}.npy",
        #     "texture": f"/datadrive/yingwei/LangSplat_prev_generate_maps/dataset/lerf_ovs/teatime/output/teatime_{level}_negative/texture/renders_npy/{scene_lst[scene_index]}.npy"
        # }
        
        # negative_vals = []
        # for key in negative_paths:
        #     negative_val = np.load(negative_paths[key])
        #     extracted_val = negative_val[:, :, 0:1]
        #     negative_vals.append(extracted_val)
        
        # concatenated_negatives = np.concatenate(negative_vals, axis=-1)
        
        # positive_vals = positive_vals[:, :, 0:1].reshape(-1, 1)
        # negative_vals = concatenated_negatives.reshape(-1, len(negative_paths))
        
        # repeated_pos = np.repeat(positive_vals, len(negative_paths), axis=1)
        
        # repeated_pos_tensor = torch.tensor(repeated_pos, dtype=torch.float32)
        # negative_vals_tensor = torch.tensor(negative_vals, dtype=torch.float32)

        # sims = torch.stack((repeated_pos_tensor, negative_vals_tensor), dim=-1)
        # ****** recover ******
        sims = torch.stack((repeated_pos_tensor, negative_vals_tensor), dim=-1)
        softmax = torch.softmax(10 * sims, dim=-1)  # sharpen the results
        best_id = softmax[..., 0].argmin(dim=1)
        
        # Gather the results based on the best_id
        result = torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]
        
        return result


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

    def get_max_across(self, scene_index):
        # n_phrases = len(self.positives)
        # n_phrases_sims = [None for _ in range(n_phrases)]
        
        # n_levels, h, w, _ = sem_map.shape
        # clip_output = sem_map.permute(1, 2, 0, 3).flatten(0, 1)

        # n_levels_sims = [None for _ in range(n_levels)]
        # for i in range(n_levels):
        #     for j in range(n_phrases):
        #         probs = self.get_relevancy(clip_output[..., i, :], j)
        #         pos_prob = probs[..., 0:1]
        #         n_phrases_sims[j] = pos_prob
        #     n_levels_sims[i] = torch.stack(n_phrases_sims)
        
        # relev_map = torch.stack(n_levels_sims).view(n_levels, n_phrases, h, w)
        # return relev_map
        
        
        n_phrases = len(self.positives)
        n_phrases_sims = [None for _ in range(n_phrases)]
        
        n_levels, h, w = 1, 731, 988

        n_levels_sims = [None for _ in range(n_levels)]
        for i in range(n_levels):
            for j in range(n_phrases):
                probs = self.get_relevancy(j, scene_index) # cal each feature level
                pos_prob = probs[..., 0:1]
                # print("pos_prob", pos_prob.shape)
                n_phrases_sims[j] = pos_prob
                pos_prob = pos_prob.view(h, w)
                
                # greyscale_image = (pos_prob.numpy() * 255).astype(np.uint8)

                # # Create a PIL image from the NumPy array
                # image = Image.fromarray(greyscale_image, mode='L')

                # # Save the image
                # image.save('greyscale_image.png')

                # torch.save(pos_prob, 'pos_prob_bear.pt')
                # assert False
                # print(pos_prob)
            n_levels_sims[i] = torch.stack(n_phrases_sims)
        relev_map = torch.stack(n_levels_sims).view(n_levels, n_phrases, h, w)
        print("relev_map", relev_map.shape)

        return relev_map
    
    