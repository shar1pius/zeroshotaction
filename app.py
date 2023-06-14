import data
import torch
import gradio as gr
from models import imagebind_model
from models.imagebind_model import ModalityType
from PIL import Image, ImageDraw, ImageFont
import glob
import os
from tqdm import tqdm
import argparse
import numpy as np
from skimage.io import imsave
import pdb
import imageio
from collections import deque

parser = argparse.ArgumentParser(description="")
parser.add_argument("--input_video_file", default='./.assets/ImpressiveJapaneseKatsuCurry.mp4', type=str)
parser.add_argument("--output_video_file", default='./.assets/ImpressiveJapaneseKatsuCurry_out_v2.mp4', type=str)
parser.add_argument("--text", default='stirring using spatula|adding ingredients', type=str, help='add text with | seperator')
args = parser.parse_args()

video_dir = args.input_video_file.split('.mp4')[0]; os.makedirs(video_dir, exist_ok = True)
video_image_dir = os.path.join(video_dir, 'images'); os.makedirs(video_image_dir, exist_ok = True)
# # extract video frames and save them into a directory
print('Reading and extracting video frames......')
reader = imageio.get_reader(args.input_video_file, 'ffmpeg')
fps = reader.get_meta_data()['fps']
for num, image in enumerate(reader):
    save_img_file = os.path.join(video_image_dir, str(num).zfill(8)+'.jpg')
    imsave(save_img_file, image)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

## Prompt
#text = "stirring using spatula|adding ingredients|cleaning hands|putting gloves|cutting|person standing"
text = "stirring using spatula|adding ingredients|cleaning hands|putting gloves"

def image_text_zeroshot(image, text_list):
    image_paths = [image]
    labels = [label.strip(" ") for label in text_list.strip(" ").split("|")]
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(labels, device),
        ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    scores = (
        torch.softmax(
            embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1
        )
        .squeeze(0)
        .tolist()
    )

    score_dict = {label: score for label, score in zip(labels, scores)}

    return score_dict

# running window
last_five_strings = deque(maxlen=3)

# stitch prediction into a video
print('stitch prediction into a video......')
writer = imageio.get_writer(args.output_video_file, fps = fps)
for img_file in tqdm(sorted(glob.glob(video_image_dir + '/*'))):
    fname = os.path.basename(img_file).split('.')[0]
    print(text)
    scores =  image_text_zeroshot(img_file, text)
    # string = str(scores)
    val = list(scores.values())
    # threshold based on vusial feedback.
    if val[0] > 0.75:
        string = 'S T I R R I N G'
    elif val[1] > 0.82:
        string = 'A D D I N G   I N G R E D I E N T S'
    else:
        string = 'N O N E'

    last_five_strings.append(string)
    most_common_string = max(set(last_five_strings), key=last_five_strings.count)
    #print('last 5 strings - ',last_five_strings, 'most common string = ',most_common_string)

    img = Image.open(img_file)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('./.assets/Lato-Regular.ttf' ,28)
    draw.text((40, 40), most_common_string, font=font, fill='BLACK')
    writer.append_data(np.array(img))


writer.close()
