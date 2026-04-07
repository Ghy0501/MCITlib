import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import torch
import pickle
from transformers import AutoTokenizer, AutoModel
from typing import List


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # [batch, seq_len, hidden]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )


def sentence_bert(sentences: List[str]):
    tokenizer = AutoTokenizer.from_pretrained('/your_path/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('/your_path/all-MiniLM-L6-v2')

    encoded_input = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Instruction embedding generation')
    parser.add_argument(
        '--model_path',
        default="/mnt/ShareDB-3TB/syc/model/all-MiniLM-L6-v2",
        type=str,
        help='model_path'
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path)
    # list_instruction_imagenetr = [
    #                 "What is the object in the image?\nAnswer the question using a single word or phrase.",
    #                 "What is the object in the image?\nAnswer the question using a single word or phrase.",
    #                 "What is the object in the image?\nAnswer the question using a single word or phrase.",
    #                 "What is the object in the image?\nAnswer the question using a single word or phrase.",
    #                 "What is the object in the image?\nAnswer the question using a single word or phrase.",
    #             ]
    # list_instruction_arxivqa = ["At approximately what value of k [Mpc^-1] does the slope of the primordial power spectrum, as indicated by the dashed blue line, become nearly vertical?\n\\( 10^{-2} \\) Mpc^-1\n\\( 10^{-1} \\) Mpc^-1\n\\( 10^{0} \\) Mpc^-1\n\\( 10^{1} \\) Mpc^-1\nRationale:The slope of the primordial power spectrum is represented by the dashed blue line. This line becomes nearly vertical around \\( 10^{-1} \\) Mpc^-1, as seen in the graph where the line sharply transitions from a negative slope to a nearly infinite slope.\nAnswer with the option's letter from the given choices directly.",
    #                 "What follows the creation of a \"bag-of-words\" representation in this process?\nA) Overlaying SIFT descriptors\nB) Clustering descriptors into \"visual words\"\nC) Applying semantic term classifiers\nD) Creating semantic representation\nRationale:The figure indicates that after the \"bag-of-words\" representation is created (part D), the next step is to apply semantic term classifiers (part E), which is a precursor to creating the final semantic representation (part F).\nAnswer with the option's letter from the given choices directly.",
    #                 "Based on the pie charts in panel b, which parameter has the greatest relative importance for balance recovery in the robotic model?\nJoint position\nBase velocity v\nJoint torque\nGravity vector\nRationale:The pie chart for balance recovery in panel b shows the largest segment associated with 'Joint position', which indicates that it has the greatest relative importance, taking up 44% of the pie.\nAnswer with the option's letter from the given choices directly.",
    #                 "What does the graph indicate about the relationship between energy (E) and the flux of ultra-high-energy (UHE) photons?\nThe flux of UHE photons increases exponentially with energy.\nThe flux of UHE photons remains constant across different energies.\nThe flux of UHE photons decreases as the energy increases.\nThe graph shows no discernible pattern between energy and the flux of UHE photons.\nRationale:The graph illustrates a negative correlation between the energy (log(E/eV)) and the flux (E^3dN/dE (m^-2 s^-1 sr^-1)), as the energy increases, the flux represented by the blue stars shows a decreasing trend.\nAnswer with the option's letter from the given choices directly.",
    #                 "Based on figure b), which node number is the immediate successor of the root node for the most number of branches?\nA) 1\nB) 2\nC) 3\nD) 4\nRationale:The root node in figure b) is connected to multiple nodes directly. Node number 2 has the most branches descending from it, indicating it is the immediate successor with the most number of branches.\nAnswer with the option's letter from the given choices directly."]
    # list_instruction_vizwiz = ["What is happening in the image?\nGenerate a brief caption for the image.",
    #                 "What is happening in the image?\nGenerate a brief caption for the image.",
    #                 "What is happening in the image?\nGenerate a brief caption for the image.",
    #                 "What is happening in the image?\nGenerate a brief caption for the image.",
    #                 "What is happening in the image?\nGenerate a brief caption for the image."]
    # list_instruction_iconqa = [
    #                 "How many fish are there?\nAnswer the question using a single word or phrase.",
    #                 "How many squares are there?\n0. 72\n1. 71\n2. 63\nAnswer with the option's letter from the given choices directly.",
    #                 "How many pigs are there?\n0. 2\n1. 3\n2. 1\n3. 4\n4. 5\nAnswer with the option's letter from the given choices directly.",
    #                 "How many fish are there?\nAnswer the question using a single word or phrase.",
    #                 "Does this shape have a circle as a face?\n0. no\n1. yes\nAnswer with the option's letter from the given choices directly.",
    #             ]
    # list_instruction_clevr = [
    #                 "Subtract all spheres. How many objects are left?\nAnswer the question using a single word or phrase.",
    #                 "Subtract all blocks. How many objects are left?\nAnswer the question using a single word or phrase.",
    #                 "Subtract 2 yellow balls. How many objects are left?\nAnswer the question using a single word or phrase.",
    #                 "Add 2 big metallic cubes. How many objects exist?\nAnswer the question using a single word or phrase.",
    #                 "Add 7 matte cylinders. How many matte cylinders exist?\nAnswer the question using a single word or phrase."
    #             ]

    # list_instruction_flickr30k = [
    #                 "What is happening in the image?\nGenerate a brief caption for the image.",
    #                 "What is happening in the image?\nGenerate a brief caption for the image.",
    #                 "What is happening in the image?\nGenerate a brief caption for the image.",
    #                 "What is happening in the image?\nGenerate a brief caption for the image.",
    #                 "What is happening in the image?\nGenerate a brief caption for the image.",
    #             ]

    # list_instruction_rs = [
    #                 "Is there a vineyard? \nAnswer the question using a single word or phrase.",
    #                 "Are there less residential buildings than grass areas? \nAnswer the question using a single word or phrase.",
    #                 "What is the area covered by buildings? \nAnswer the question using a single word or phrase.",
    #                 "Is a commercial building present? \nAnswer the question using a single word or phrase.",
    #                 "What is the area covered by industrial areas on the right of the  road? \nAnswer the question using a single word or phrase.",
    #             ]

    # list_instruction_med = [
    #                 "does the renal glomerulus show markedly thickend glomerular basement membrane in a diabetic?",
    #                 "are old intracortical infarcts seen as areas of tissue loss and residual gliosis?",
    #                 "is mass of intestines and mesenteric nodes showing lesions that look more like carcinoma but are in fact tuberculosis?",
    #                 "does this image show cut surface with multiple small infiltrates that simulate granulomata diagnosed as reticulum cell sarcoma?",
    #                 "is rheumatoid nodule composed of central necrosis rimmed by palisaded histiocytes?",
    #             ]

    # list_instruction_ad = [
    #                 "What is the status of the pedestrians that are to the front right of the ego car? Objects are encoded using <c,CAM,[cx,cy]>, where c is the identifier, CAM indicates the camera where the object\u2019s center point is situated, and x, y represent the horizontal and vertical coordinates of the center point of the 2D bounding box.",
    #                 "What are objects to the back of the ego car? Objects are encoded using <c,CAM,[cx,cy]>, where c is the identifier, CAM indicates the camera where the object\u2019s center point is situated, and x, y represent the horizontal and vertical coordinates of the center point of the 2D bounding box.",
    #                 "Is <c2,CAM_BACK_RIGHT,[925, 840]> an object that the ego vehicle should consider in the current scene? Objects are encoded using <c,CAM,[cx,cy]>, where c is the identifier, CAM indicates the camera where the object\u2019s center point is situated, and x, y represent the horizontal and vertical coordinates of the center point of the 2D bounding box.",
    #                 "Is <c2,CAM_BACK,[511, 786]> an object that the ego vehicle should consider in the current scene? Objects are encoded using <c,CAM,[cx,cy]>, where c is the identifier, CAM indicates the camera where the object\u2019s center point is situated, and x, y represent the horizontal and vertical coordinates of the center point of the 2D bounding box.",
    #                 "What is the status of the construction vehicle that is to the back left of the ego car? Objects are encoded using <c,CAM,[cx,cy]>, where c is the identifier, CAM indicates the camera where the object\u2019s center point is situated, and x, y represent the horizontal and vertical coordinates of the center point of the 2D bounding box.",
    #             ]

    # list_instruction_sci = [
    #                 "Which bird utilizes a grasping hold?\nA.Eagle\nB.Sparrow\nC.Heron\nD.Woodpecker\nAnswer with the option's letter from the given choices directly.",
    #                 "Which is a producer?\nA.chimpanzee\nB.anaconda\nC.frog\nD.fruits\nAnswer with the option's letter from the given choices directly.",
    #                 "The herb Cilantro is also known as\nA.Mint\nB.Thyme\nC.Coriander\nD.Basil\nAnswer with the option's letter from the given choices directly.",
    #                 "What letter corresponds to 0 km/hr?\nA.C\nB.B\nC.A\nD.D\nAnswer with the option's letter from the given choices directly.",
    #                 "What is the highest value in states that border Iowa?\nBe succinct.",
    #             ]

    # list_instruction_fin = [
    #                 "Based on the candlestick chart, what can be observed about the price movement between April 11 and April 16? A. The price experienced a significant increase followed by a slight decrease. B. The price showed a continuous decline throughout this period. C. The price fluctuated wildly with no discernible trend. D. The price remained relatively stable with minor fluctuations.",
    #                 "Based on the candlestick chart, which of the following statements best describes the trend of the stock price from February 14th to May 11th? A. The stock price demonstrated a clear upward trend with occasional corrections. B. The stock price remained relatively stable with minor fluctuations. C. The stock price experienced significant volatility with no clear trend. D. The stock price showed a consistent downward trend throughout the period.",
    #                 "Based on the candlestick chart, which of the following best describes the overall trend of the stock price from March 21 to May 30? A. The stock price fluctuated wildly without any discernible trend. B. The stock price experienced a significant increase followed by a steady decline. C. The stock price remained relatively stable throughout the period. D. The stock price showed a consistent upward trend.",
    #                 "Based on the candlestick chart, which of the following statements best describes the trading volume during the period when the stock price was declining? A. Trading volume increased significantly as the stock price declined. B. Trading volume remained consistently low throughout the decline. C. Trading volume fluctuated unpredictably with no clear pattern. D. Trading volume decreased as the stock price declined.",
    #                 "Was there a significant increase in volume during the rising trend?",
    #             ]

    list_instruction_ocr = [
                    "In what year was a peak in sales of the Ford Fiesta observed? Answer: ",
                    "What is the title of this book? Answer: ",
                    "What is written in biggest size font in the top left corner rectangle box in the map ? Answer: ",
                    "Generate the detailed caption in English: ",
                    "By 2019, how many road traffic fatalities were there in the Czech Republic? Answer: ",
                ]

    list_instruction_math = [
                    "Hint: Please answer the question and provide the final answer at the end.\nQuestion: What is the minimum value of Tomato on the graph?",
                    "According to the question shown in the image, please first conduct reasoning, and then answer the question and provide the final value, e.g., The answer is xxx\nQuestion: Can you tell me the length of arc CE in sector CBE?",
                    "Hint: Please answer the question and provide the final answer at the end.\nQuestion: What is the first item on the agenda?",
                    "Hint: Please answer the question and provide the final answer at the end.\nQuestion: How many activities have the code 'OK' mentioned under the 'CODE' column?",
                    "Hint: Please answer the question and provide the final answer at the end.\nQuestion: What is the measure of angle APB?",
                ]

    list_instruction_vp = [
                    "There is a small metal object to the right of the shiny cube that is left of the gray object; what is its shape?",
                    "How many animals are pictured?",
                    "What number of metallic cylinders are on the right side of the tiny thing to the right of the brown shiny thing?",
                    "There is a cylinder on the left side of the blue metallic thing; what is its material?",
                    "How many refrigerators are in this picture?",
                ]

    list_instruction_app = [
                    "Pinpoint the bounding box coordinates of the clickable area needed to execute the instruction: \"Enter your name\". The coordinates should be specified as four float numbers between 0 and 1, i.e., [left, top, right, bottom].",
                    "What is the command to display the banlist?",
                    "What is the status of \"Enable PIN security\"?",
                    "You are given a phone UI screen. Describe the screen in one sentence.",
                    "Please identify and generate the text content of the webpage's main heading.",
                ]
    
    list_multi = [list_instruction_ocr, list_instruction_math, list_instruction_vp, list_instruction_app]
    
    
    ########################## single指令embedding保存 ###########################
    # instruction_emb = sentence_bert(list_instruction)
    # print(instruction_emb.size())

    ########################## multi指令embedding保存 ###########################
    instruction_emb = sentence_bert(list_multi[0])
    instruction_emb = torch.mean(instruction_emb, dim=0, keepdim=True)
    # print(instruction_emb)
    for i in range(1,4):
        now_emb = sentence_bert(list_multi[i])
        now_emb = torch.mean(now_emb, dim=0, keepdim=True)
        instruction_emb = torch.cat([instruction_emb, now_emb], dim=0)


    with open('./ins_emb_multi_acl.pkl', 'wb') as f:
        pickle.dump(instruction_emb, f)
        print(instruction_emb)
        print(instruction_emb.shape)