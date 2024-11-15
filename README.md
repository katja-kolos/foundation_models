# foundation_models

This repo is a course project in Foundation Models. Our domain is scientific question answering, based on [Science QA](https://scienceqa.github.io/) dataset. 

### 💡 Hypotheses 💡
1. **agentic approach simulation**: a very **small LM** might be able to perform this task in a tool calling scenario: SML can 'decide' to invoke (a) 🌄 tool for image captioning (b) 📚 tool for information retrieval (both tools simulated by the dataset's provided image caption data and lecture data). We construct a framework with SML that is way less computationally expensive than inferencing an LLM, and compare the performance to existing good multimodal models.
3. **better performance 📈 **: we can improve performance of existing good multimodal models through soft-prompting.
4. **knowledge distillation from reasoning paths experiment 🧪**: fine-tuning on reasoning paths of domain task (with retrieved context) may be more efficient than fine-tuning on whole domain texts (this is fine-tuning on very concentrated data, like stackoverflow answers vs whole domain books). 

### 👣 Steps 👣
1. Benchmarking existing good multimodal models (and small text-only models as weak baseline) - in progress
2. ...
