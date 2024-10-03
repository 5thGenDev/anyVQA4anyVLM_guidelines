import json
from torch.utils.data import Dataset
import torch
import gc
from typing import List
from PIL import Image
import os
import re

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    
    
def is_image_valid(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError, FileNotFoundError) as e:
        return False
    
    
def align_image_ext(image_path: str):
    r"""
    Goal: 
        no .tif because Image.open() command hates it, 
        some dataset like floodnet has .jpeg extension specifically
        
    Input: initial relative image path
    
    output: reformatted relative image path
    """
    image_path = image_path
    folder_name, file_name = os.path.split(image_path)
    name, ext = os.path.splitext(file_name)
    if ext == '.tif':
        image_path = os.path.join(folder_name, f"{name}.png")
    elif folder_name == "floodnet":
        image_path = os.path.join(folder_name, f"{name}.jpg")
    return image_path


class CoordinateValueError(Exception):
    """Coordinate values exceed normalisation range of original dataset."""
    def __init__(self, coord, max_value):
        self.coord = coord
        self.max_value = max_value
        super().__init__(f"Coordinate {coord} exceeds the maximum allowed value of {max_value}")


class PhraseError(Exception):
    def __init__(self, phrase):
        self.phrase = phrase
        super().__init__(f"This GT prompt wrong labelled 'important-phrase': {phrase}")


def inconvertible_phrase(text):
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
    if "," in text:
        return True
    if "." in text:
        return True
    return False


def geochat_to_florence2_bbox(box):
    """
    top left(x0, y0)
    bottom right(x1, y1)
    box = [x0, y0, x1, y1]
    """
    geochat_bbox = 100
    florence2_bbox = 1000
    florence_box = []
    for coord in box:
        float_coord = float(coord)
        if float_coord > geochat_bbox:
            raise CoordinateValueError(float_coord, geochat_bbox)
        scaled_float_coord = round(float_coord/ geochat_bbox * florence2_bbox)
        scaled_str_coord = str(scaled_float_coord)
        florence_box.append(scaled_str_coord) 
    return florence_box


def remove_image_tag(value):
    return value.replace("<image>", "").strip()


def format_grounding_visual(answer):
    formatted_question = "<DENSE_REGION_CAPTION>"
    delimiter_pattern = r"<delim>"
    answer = re.sub(delimiter_pattern, "", answer)
    
    phrase_pattern = r"<p>(.*?)</p>"
    old_hbbox_pattern = r"\{(<\d+>)(<\d+>)(<\d+>)(<\d+>)\|<(\d+)>\}"
    
    phrases = [(match.group(1), match.start(), match.end()) for match in re.finditer(phrase_pattern, answer)]
    
    formatted_answer = ""
    for i, (phrase, start_idx, end_idx) in enumerate(phrases):
        if re.search(old_hbbox_pattern, phrase) or inconvertible_phrase(phrase):
            raise PhraseError(phrase)
        new_sentence = f"{phrase}"
        
        search_start = end_idx
        # phrases[i+1][1] = start_idx of next phrase
        search_end = phrases[i + 1][1] if i + 1 < len(phrases) else len(answer)
        sub_text = answer[search_start:search_end]
        matched_old_hbbox_pattern = list(re.finditer(old_hbbox_pattern, sub_text))
        hbbox_pattern = ""
        for match in matched_old_hbbox_pattern:
            try:
                locs = [match.group(i).strip('<>') for i in range(1, 5)]
                locs = geochat_to_florence2_bbox(locs)
                hbbox_pattern += "".join([f"<loc_{loc}>" for loc in locs])
            except CoordinateValueError:
                raise
            angle = match.group(5)
            hbbox_pattern += f"<angle_{angle}>"
        new_sentence += f"{hbbox_pattern}"
        formatted_answer += new_sentence

    return formatted_question, formatted_answer


def format_grounding_description(answer):
    formatted_question = "<MORE_DETAILED_CAPTION>"
    delimiter_pattern = r"<delim>"
    old_hbbox_pattern = r"\{(<\d+>)(<\d+>)(<\d+>)(<\d+>)\|<(\d+)>\}"
    phrase_pattern = r"<p>(.*?)</p> "
    
    formatted_answer = re.sub(delimiter_pattern, "", answer)
    
    phrase_pattern = r"<p>(.*?)</p> "
    phrases =  [match.group(1) for match in re.finditer(phrase_pattern, formatted_answer)]
    for phrase in phrases:
        if re.search(old_hbbox_pattern, phrase) or inconvertible_phrase(phrase):
            raise PhraseError(phrase)
        formatted_answer = re.sub(phrase_pattern, phrase.strip(), formatted_answer)
        
    matched_old_hbbox_pattern = list(re.finditer(old_hbbox_pattern, formatted_answer))
    hbbox_pattern = ""
    for match in matched_old_hbbox_pattern:
        try:
            locs = [match.group(i).strip('<>') for i in range(1, 5)]
            locs = geochat_to_florence2_bbox(locs)
            hbbox_pattern += "".join([f"<loc_{loc}>" for loc in locs])
        except CoordinateValueError:
            raise
        angle = match.group(5)
        hbbox_pattern += f"<angle_{angle}>" 
        formatted_answer = re.sub(old_hbbox_pattern, hbbox_pattern , formatted_answer)
           
    return formatted_question, formatted_answer


def format_conversations(conversations, image_path):
    formatted_image_path = image_path
    formatted_conversations = []
    
    conversations[0]['value'] = remove_image_tag(conversations[0]['value'])
    question = conversations[0]['value']
    if '[grounding]' in question:
        for i in range(0, len(conversations), 2):
            try:
                formatted_question, formatted_answer = format_grounding_visual(conversations[i+1]['value'])
                formatted_conversations.append({'from': 'human', 'value': formatted_question})
                formatted_conversations.append({'from': 'gpt', 'value': formatted_answer})
                formatted_question, formatted_answer = format_grounding_description(conversations[i+1]['value'])
                formatted_conversations.append({'from': 'human', 'value': formatted_question})
                formatted_conversations.append({'from': 'gpt', 'value': formatted_answer})
            except(CoordinateValueError, PhraseError):
                raise
            
    return formatted_conversations, image_path
            
            
class formatVQA(Dataset):
    # class CustomDataset is first instantiated, before being wrapped with DistributedSampler
    def __init__(self, data_path: str):
        super(formatVQA, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.list_data_dict = list_data_dict
        del list_data_dict
        flush()

    def _get_data_dict(self) -> List:
        print(f"VQA size is {len(self.list_data_dict)}")
        return self.list_data_dict
         
VQA_dataset = formatVQA("GeoChat_Instruct.json")
list_data_dict = VQA_dataset._get_data_dict()         

source_dir = "/home/ubuntu/Documents/nam/GeoChat_images/share/softwares/kartik/GeoChat_finetuning/final_images_llava"
grounding_excl_dict = []
for sample in list_data_dict:
    conversations = sample['conversations']
    question = conversations[0]['value']
    if not '[grounding]' in question:
        continue
    
    relative_image_path = sample['image']
    absolute_image_path = os.path.join(source_dir, relative_image_path)
    if not is_image_valid(absolute_image_path):
        continue
    
    try:
        sample['conversations'], sample['image'] = format_conversations(sample['conversations'], sample['image'])
    except(CoordinateValueError, PhraseError):
        continue
    grounding_excl_dict.append(sample)

print(f"New VQA size is {len(grounding_excl_dict)}")
with open("Florence_Instruct_grounding_only.json", "w") as file:
    json.dump(grounding_excl_dict , file, indent=4)