from torch.utils.data import Dataset
from typing import List
import json
import re
import os
from PIL import Image

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


def format_refer(question, answer):
    phrase_pattern = r"<p>(.*?)</p>"
    old_hbbox_pattern = r"\{(<\d+>)(<\d+>)(<\d+>)(<\d+>)\|<(\d+)>\}"
    
    phrase = re.search(phrase_pattern, question).group(1) # expect only 1 phrase <p>...</p> per question or sentence in answer.
    if re.search(old_hbbox_pattern, phrase) or inconvertible_phrase(phrase):
        raise PhraseError(phrase)
    formatted_question = f"<CAPTION_TO_PHRASE_GROUNDING>{phrase}"
    
    formatted_answer = f"{phrase}"
    matched_old_hbbox_pattern = list(re.finditer(old_hbbox_pattern, answer))
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
        
        if match == matched_old_hbbox_pattern[-1]:
            exit
    formatted_answer += f"{hbbox_pattern}"
    return formatted_question, formatted_answer


def format_scene_classify(question):
    question = question.strip()
    formatted_question = "<SCENE_CLASSIFY>"
    sentences = question.split('.')
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    for idx, sentence in enumerate(sentences):
        if idx == 0:
            formatted_question += f"{sentence}"
            continue
        
        if sentence.startswith("Classes:"):
            phrase = sentence.split("Classes")[-1].strip() 
            sentence = f"{phrase}"
        formatted_question += f"{sentence}. "
    
    return formatted_question.strip()


def format_vqa(question):
    sentences = question.split('.')
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    formatted_question = "<VQA>"
    for sentence in sentences:
        formatted_question += sentence.replace(" \n", " ", 1)

    return formatted_question


def format_conversations(conversations, image_path):
    formatted_image_path = image_path
    formatted_conversations = []
    
    conversations[0]['value'] = remove_image_tag(conversations[0]['value'])
    question = conversations[0]['value']
    if '[refer]' in question:
        for i in range(0, len(conversations), 2):
            try:
                formatted_question, formatted_answer = format_refer(conversations[i]['value'], conversations[i+1]['value'])
                formatted_conversations.append({'from': 'human', 'value': formatted_question})
                formatted_conversations.append({'from': 'gpt', 'value': formatted_answer})
            except(CoordinateValueError, PhraseError):
                raise
                
    elif question.startswith('Classify'):
        for i in range(0, len(conversations), 2):
            formatted_question = format_scene_classify(conversations[i]['value'])
            formatted_conversations.append({'from': 'human', 'value': formatted_question})
            formatted_conversations.append({'from': 'gpt', 'value': conversations[i+1]['value']})
            
    else:
        for i in range(0, len(conversations), 2):
            formatted_question = format_vqa(conversations[i]['value'])
            formatted_conversations.append({'from': 'human', 'value': formatted_question})
            formatted_conversations.append({'from': 'gpt', 'value': conversations[i+1]['value']})

            
    return formatted_conversations, formatted_image_path


class formatVQA(Dataset):
    def __init__(self, data_path: str):
        super(formatVQA, self).__init__()
        self.list_data_dict = json.load(open(data_path, "r"))
        
    def _get_data_dict(self) -> List:
        print(f"VQA size is {len(self.list_data_dict)}")
        return self.list_data_dict
         

VQA_dataset = formatVQA("/absolute/path/to/GeoChat_Instruct.json")
list_data_dict = VQA_dataset._get_data_dict()

source_dir = "/absolute/path/to/share/softwares/kartik/GeoChat_finetuning/final_images_llava"
new_list_data_dict = []
for sample in list_data_dict:
    relative_image_path = sample['image']
    absolute_image_path = os.path.join(source_dir, relative_image_path)
    if not is_image_valid(absolute_image_path):
        continue
    
    try:
        sample['conversations'], sample['image'] = format_conversations(sample['conversations'], sample['image'])
    except(CoordinateValueError, PhraseError):
        continue
    new_list_data_dict.append(sample)
    
print(f"New VQA size is {len(new_list_data_dict)}")
with open("Florence_Instruct.json", "w") as file:
    json.dump(new_list_data_dict, file, indent=4)
