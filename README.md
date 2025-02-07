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
1. How well do front-tier LLMs perform on Scientific QA & Scientific Reasoning?
2. How would learning an adapter for correct solution generation affect generations of smaller multimodal LLMs for this task?
3. How does learning from golden data compare to learning from teacher model's outputs?
