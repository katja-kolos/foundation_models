# foundation_models

This repo is a course project in Foundation Models. Our domain is scientific question answering, based on [Science QA](https://scienceqa.github.io/) dataset. 

### 💡 Hypotheses 💡
1. **agentic approach simulation**: a very **small LM** might be able to perform this task in a tool calling scenario: SML can 'decide' to invoke (a) 🌄 tool for image captioning (b) 📚 tool for information retrieval (both tools simulated by the dataset's provided image caption data and lecture data). We construct a framework with SML that is way less computationally expensive than inferencing an LLM, and compare the performance to existing good multimodal models.
  tl;dr: SML + lecture vs BigLLM; zero-shot on validation. 
2. **better performance 📈 **: we can improve performance of existing good multimodal models through soft-prompting.
3. **knowledge distillation from reasoning paths experiment 🧪**:
   
  * 3.1. learning from answer and solution of good model as opposed to learning from raw data lowers performance significantly (!test!)
  * 3.2. learning from answer and solution of good model AND answer and solution from training data also lowers performance significantly because of conflicts (!test!)
  * 3.3. learning from good CoT of good model may lead to better performance than learning from the outputs (answer and solution) of the same good model (!test!)
  * 3.4. learning from answer and solution of good model and retrieved augmented context (in our simulation: lecture) can boost performance (!test!)
  * 3.5. learning from mistakes of big model (answer and solution of train + answer and solution of big model + retrieved augmented context) can boost performance even further (!test!)
  * 3.6. fine-tuning on reasoning paths of domain task (with or without "retrieved" context (=lecture) may be more efficient than fine-tuning on whole domain texts (compare performance of our best method to zero-shot inference of science model NOT tuned on this dataset, also compare with leaderboard. Also report differences in training and inference costs).



### 👣 Steps 👣
1. Benchmarking existing good multimodal models (and small text-only models as weak baseline) - done.
We have a champion: Gemini is the best for the role of big teacher LLM. Its outputs will be used as training CoT reasoning paths.
2. Define the student model!
3. How are we going to train the student? Write script for LoRA/soft prompting. 
