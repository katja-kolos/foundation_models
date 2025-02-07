Our evaluation scripts are a based of those defined in the repository of the [ScienceQA dataset](https://github.com/lupantech/ScienceQA/tree/main/tools).

We preserve the original naming conventions, as well as the choice of the libraries implementing the metrics:

- BLEU-1, BLEU-4: [nltk.translate.bleu_score.sentence_bleu](https://www.nltk.org/api/nltk.translate.bleu_score.html)
- ROUGE-L: [rouge](https://pypi.org/project/rouge/)
(Upon careful investigation we noticed that the `rouge-l` were not faithful to the definitions in the [ROUGE paper](https://aclanthology.org/W04-1013.pdf); rouge-score[https://pypi.org/project/rouge-score/] would have been a better choice). 
- METEOR: [huggingface evaluate](https://huggingface.co/spaces/evaluate-metric/meteor)
- cosine similarity with a sentence transformer model: `sentence-transformers/all-MiniLM-L6-v2`.
