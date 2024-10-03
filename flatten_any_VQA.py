import json
from torch.utils.data import Dataset
from typing import List
import os

def inconvertible_grounding_visual(text):
    if "{" in text:
        return True
    if "}" in text:
        return True
    if "|" in text:
        return True
    if "<p>" in text:
        return True
    if "," in text:
        return True
    if "." in text:
        return True
    return False


def inconvertible_grounding_description(text):
    if "{" in text:
        return True
    if "}" in text:
        return True
    if "|" in text:
        return True
    if "<p>" in text:
        return True
    if "</p>" in text:
        return True
    return False

class flatten_dataset(Dataset):
    r"""
    Goal: Bring many QA-pairs inside 'conversations' per VQA sample -> (1 QA-pair, 1 image) per VQA sample
    Input: VQA dataset, which is usually formatted like [sample_1 Dict, sample_2 Dict,..., sample_n Dict].
        [
            sample_1 Dict{
            'image': relative/path/to/image
            'conversations': contain many QA-pairs
                [
                    {
                        'from': 'human'
                        'value': "Describe this image"
                    }, 
                    {
                        'from': 'gpt'
                        'value': "It has Formula Ford cars?"
                    }, 
                    {
                        'from': 'human'
                        'value': "What is at this region <loc_20><loc_49><loc_142><loc_121>?"
                    }, 
                    {
                        'from': 'gpt'
                        'value': "Nam's favourite night club"
                    },      
                ]
            },
            
            sample_2 Dict{
                ....
            },
            ... and so on ...
        ]
        
    Output: flattened VQA dataset, which is now formatted like [sample_1 Tuple, sample_2 Tuple,..., sample_n Tuple]
        [
            (relative/path/to/image, "Describe this image", "It has Formula Ford cars?"),
            (relative/path/to/image, "What is at this region <loc_20><loc_49><loc_142><loc_121>?", "Nam's favourite night club"),
            (relative/path/to/image, sample_2 1st question, sample_2 1st answer),
            (relative/path/to/image, sample_2 2nd question, sample_2 2nd answer),
            ... and so on ...
        ]
        
    """
    def __init__(self, data_path: str):
        super(flatten_dataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.list_data_dict = list_data_dict
        self.all_pairs = []
        
        for VQA_sample in self.list_data_dict:            
            conversations = VQA_sample['conversations']
            for i in range(0, len(conversations), 2): 
                if not (conversations[i]['from'] == 'human' and conversations[i+1]['from'] == 'gpt'):
                    continue
                if "<DENSE_REGION_CAPTION>" in conversations[i + 1]['value']:
                    if inconvertible_grounding_visual(conversations[i + 1]['value']):
                        continue
                if inconvertible_grounding_description(conversations[i]['value']) or inconvertible_grounding_description(conversations[i + 1]['value']):
                    continue
                
                self.all_pairs.append((VQA_sample['image'], conversations[i]['value'], conversations[i + 1]['value']))
         
    def _get_data_dict(self) -> List:
        print(f"Initial VQA size is {len(self.list_data_dict)}")
        print(f"Flattened VQA size is {len(self.all_pairs)}")
        return self.all_pairs

source_dir = "/home/ubuntu/Documents/nam/GeoChat_images"

relative_data_path = "Florence_Instruct_grounding_only.json"
absolute_data_path = os.path.join(source_dir, relative_data_path)

VQA_dataset = flatten_dataset(absolute_data_path)
VQA_allPairs = VQA_dataset._get_data_dict()

relative_flatten_data_path = "flatten_Florence_Instruct_grounding_only.json"
absolute_flatten_data_path = os.path.join(source_dir, relative_flatten_data_path)
with open(absolute_flatten_data_path, "w") as file:
    json.dump(VQA_allPairs, file, indent=4)