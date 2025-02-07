# foundation_models

This repository is a course project on Foundation Models. Our domain is scientific question answering, based on [Science QA](https://scienceqa.github.io/) dataset. 

### 📄 Data 📄
The dataset contains middle to high school level tasks in natural sciences and social sciences. Task **input** is always a short question and 2-5 answer choices, optionally with _lecture_ material, _hint_, and a relevant _image_ (the image can be part of the question or an additional illustration). The **output** should be the correct _answer_ + a plausible _explanation_ of the answer choice. 

We work with four experimental settings for **input**: 

- Question - Task - Choices - Hint (QTCH)
- Question - Task - Choices - Hint - Lecture (QTCHL)
- Question - Task - Choices - Hint - Lecture - Solution (QTCHLS)
- Question - Task - Choices - Hint - Solution (QTCHS)

Image is always attached to the input if it is available; otherwise a dummy placeholder blank image is created. 

### 🚀 Approach 🚀
We benchmark several large multimodal LLMs to assess: 
- their capabilities of answering scientific questions from existing knowledge and common sense;
- their ability to retrieve information from context (i.e. choose the correct answer if the solution is given);
- their reasoning abilities (i.e. coming to a correct conclusion provided lecture material).

Our metrics are: accuracy for answer choice; average of several normalised textual similarity metrics: BLEU-1, BLEU-4, ROUGE-L, METEOR, cosine similarity (BERT embeddings). 

We further attempt:
- adapter tuning to adapt existing small language models to the dataset, as well as
- learning from a teacher model's outputs (knowledge distillation scenario).

### 💡 Research Questions & Analysis 💡
**1. How well do front-tier LLMs perform on Scientific QA & Scientific Reasoning?**
  - They can achieve high accuracy in answers to scientific questions from multimodal inputs _(avg. accuracy 80.5125 across 4 experimental settings and 6 models)_;
  - Big LLMs can benefit from more context for better reasoning _(settings with more context have consistently better scores both in accuracy and semantic similarity than task-only setting)_;
  - Extracting information (from solutions) is easier than producing new ideas (from lectures), as shown by accuracies and textual similarity scores.

**2. How would learning an adapter for correct solution generation affect generations of smaller multimodal LLMs for this task?**
  - Learning an adapter affects the GenLM's probability to generate explanations that deviate from golden ones
  - In an experimental setting which required the model to produce the exact golden explanation from the training dataset, the adapters silenced model's generations, resulting in empty strings or answers with unknown rarely seen tokens.
  - We suspect this is due to an overly strict error function, which supposedly creates an adapter-brake that discourages the model from making any error 

**3. How does learning from golden data compare to learning from teacher model's outputs?**
  - Learning from humanly prepared data (original dataset) consistently outperforms learning from the generations of the teacher model on the train partition, as shown by textual similarity metrics of the explanations.
  - The error function on validation in KD-setting is close to the error function on validation in golden-setting; the error on train in KD-setting is consistently higher then in golden setting. 
