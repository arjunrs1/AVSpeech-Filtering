import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn import Softmax, Sigmoid
from torchvision import io
from transformers import ViltProcessor, ViltForQuestionAnswering
from pathlib import Path
from numpy import unravel_index
import tqdm

# img_folder_path: folder path for images
# questions_file: txt file with questions on separate line
class ImageTextQuery(Dataset):
    def __init__(self, img_folder_path, questions_file):

        #read image urls from folder and store in an array
        images = []
        folder_dir = Path(img_folder_path)
        for img in os.listdir(folder_dir):

            if (img.endswith(".png")):
                images.append(img)

        # read questions from questions.txt and store as integers in an array 
        
        # store question in an array
        questions = []
        with open (questions_file) as input:
            for line in input:
                if line.strip():
                    questions.append(line.rstrip())
        

        self.df = init_df()
        self.folder = folder_dir
        self.questions = np.array(questions)
        self.images = np.array(images)
                                                
    
    def __len__(self):
        return len(self.images) * len(self.questions)
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        idx_components = unravel_index(idx, (len(self.images), len(self.questions)))

        # prepare images - load from memory and store in array
        img_name = self.images[idx_components[0]]
        path = self.folder / img_name
        image = io.read_image(str(path))

        question = self.questions[idx_components[1]]

        return image, img_name, question

# question strings are not in df, but are indexed starting zero in the order from the text file
def init_df():
    cols = ["image_filename", "question", "answer", "confidence_score"]
    df = pd.DataFrame(columns=cols)
    return df

# updates image row in df for a qiven question, adds the answer and corresponding accuracy
def update_df(data, img, question, answer, accuracy):

    # get question numbers for column indexing
    sorter = np.argsort(data.questions)
    idx = sorter[np.searchsorted(data.questions, question, sorter=sorter)]
    
    # append df with new question image pairs from this batch
    temp = pd.DataFrame({"image_filename":img, "question":idx, 
                        "answer":np.vectorize(convert_to_label)(answer.detach().numpy()), 
                        "confidence_score":accuracy.detach().numpy()})
    data.df = pd.concat([data.df, temp])

# since we only care about yes and no values, we don't need full id2label dictionary
def convert_to_label(num):
    if num == 3:
        return "yes"
    elif num == 9:
        return "no"
    else:
        return "N/A"

def main():
    # dataset and loader
    vqa_data = ImageTextQuery("/vision/vision_data/sound-spaces/data/vam/data/acoustic_avspeech/img_subset/", "questions.txt")
    vqa_dataloader = DataLoader(vqa_data, batch_size=64)

    # processor and model 
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to("cuda")
    model.eval()

    # iterate over batches
    with torch.no_grad():
        for imgs, img_paths, questions in tqdm.tqdm(vqa_dataloader):
            encoding = processor (imgs, questions, return_tensors="pt", padding="max_length").to("cuda")
            outputs = model(**encoding)

            logits = outputs.logits
            ans = logits.argmax(1)

            # softmax = Softmax(dim=1)
            # softmax_vals = softmax(logits)
            # softmax_scores = softmax_vals[range(len(logits)), ans]

            sigmoid = Sigmoid()
            sigmoid_vals = sigmoid(logits)
            sigmoid_scores = sigmoid_vals[range(len(logits)), ans]

            update_df(vqa_data, np.array(img_paths), np.array(questions), ans.detach().cpu(), sigmoid_scores.detach().cpu())

    vqa_data.df = vqa_data.df.pivot(index='image_filename', columns='question', values=['answer', 'confidence_score'])
    # df columns  = [image_filename, answer_0, confidence_score_0, answer_1, ...]
    vqa_data.df.columns = [f"answer_{i}" for i in range(len(vqa_data.questions))] + [f"confidence_score_{i}" for i in range(len(vqa_data.questions))] 
    vqa_data.df.to_csv("./output.csv")


if __name__ == "__main__":
    main()