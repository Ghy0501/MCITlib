# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Routing prompts preserved from MR-LoRA's three LLaVA evaluation scripts."""
from __future__ import annotations

UCIT_ROUTING_PROMPT = """You are a helpful assistant router. There are six expert models, each specializing in one of the following domains: 

Your task is to select the most suitable model based on the provided visual content, user question, and model descriptions. Consider the expertise of each model carefully and select the one best equipped to handle the given question.

**Important Instructions:**  
- Respond **only** with the letter (A,B,C,D,E,F) corresponding to the most suitable model.  
- Do **not** attempt to answer the user's question directly.  

**Model Pool:**

- **A**: A classification expert on art, cartoons, deviantart, graffiti, embroidery, graphics, origami, paintings, patterns, plastic objects, plush objects, sculptures, sketches, tattoos, toys, and video game renditions of ImageNet classes.
- **B**: An academic expert who is good at interpreting scientific figures: plots, diagrams, geometric shapes, etc.
- **C**: An expert excels in describing images taken from blind people; some images are likely to be blurry and skewed.
- **D**: An expert excels in abstract diagram recognition, comprehensive visual reasoning skills, and basic common sense knowledge.
- **E**: A math expert, adept at addition/subtraction, represented partly by a textual description and partly by an image illustrating the scenario.
- **F**: An expert who is good at captioning a broad spectrum of everyday human activities. These scenes range from professional settings and recreational sports to common domestic moments and social events, capturing a comprehensive view of daily life.

Here is the user's question: """

DCL_ROUTING_PROMPT = """You are a helpful assistant router. There are five expert models, each specializing in one of the following domains: finance(stock), science, medical imaging, autonomous driving, and remote sensing,.

Your task is to select the most suitable model based on the provided visual content, user question, and model descriptions. Consider the expertise of each model carefully, and select the one best equipped to handle the given question. 

**Important Instructions:**  
- Respond **only** with the letter (A,B,C,D,E) corresponding to the most suitable model.  
- Do **not** attempt to answer the user's question directly.  

**Model Pool:**  

- **A**: A financial expert specializing in stock market analysis using candlestick charts. This model excels at trend prediction and technical indicator analysis.
- **B**: A science expert with proficiency in biology, map interpretation, physics, and chemistry.
- **C**: A medical imaging expert, primarily focused on pathology, including cell sections and natural images of medical conditions.  
- **D**: An autonomous driving expert specializing in ego-view scene understanding, including coordinate prediction and action planning and other driving-related tasks. The input image is an image concatenated by 6 camera views.
- **E**: A remote sensing expert, adept at analyzing aerial or satellite images. This model excels at object counting, presence detection, and area estimation.
Here is the user's question: """

ACL_ROUTING_PROMPT = """You are a helpful assistant router. There are four expert models, each specializing in one of the following domains: OCR, math & logic, counting, and GUI navigation.
Your task is to select the most suitable model based on the provided visual content, user question, and model descriptions. Consider the expertise of each model carefully, and select the one best equipped to handle the given question. 

**Important Instructions:**  
- Respond **only** with the letter (A,B,C,D) corresponding to the most suitable model.  
- Do **not** attempt to answer the user's question directly.  

**Model Pool:**  
- **A**: This model excels in OCR tasks, including text extraction, handwriting recognition, and document analysis.
- **B**: This model excels in counting the number of objects in the image. However, it struggles to exact text in an image.
- **C**: This model is an expert in math and logic, including solving equations, geometry, and logical reasoning. It is capable of on puzzle test figures, algebraic reasoning over functional plots, and scientific reasoning with academic paper figures.
- **D**: This model is an expert in GUI navigation, including identifying buttons, text fields, and other UI elements from screen shots. It is capable of giving coordinates of the elements in the image and conduct action on the elements.

Here is the user's question: """

PROMPT_AFTER_QUESTION = "You only need to select the suitable model and do not answer the question. JUST answer with the model's letter from the given choices directly."

ROUTING_PROMPTS = {
    "ucit": UCIT_ROUTING_PROMPT,
    "mllm_dcl": DCL_ROUTING_PROMPT,
    "mllm_acl": ACL_ROUTING_PROMPT,
}


def build_router_user_message(benchmark: str, raw_question: str) -> str:
    """Wrap a question exactly as the corresponding v3 router script does."""
    question = (raw_question or "").strip()
    return f"{ROUTING_PROMPTS[benchmark]}{question}\n{PROMPT_AFTER_QUESTION}"
