# generative-belief
Using generative models for belief/event factuality prediction, beating the previous state-of-the-art from Murzaku et al. 2022 (https://aclanthology.org/2022.coling-1.66). This repo uses the same archtiecture and is a general follow-up and extension Murzaku et al. 2023 (https://aclanthology.org/2023.findings-acl.44/).

To use our model for factuality/beleif generation, please use our inference code here: https://github.com/yurpl/generative-belief/blob/main/checkpoints/load.py. This should work on most datasets and return (fully end-to-end) the head words and their associated factuality values.

The checkpoints are here (too large to upload on Github): https://drive.google.com/drive/folders/1ZNvQOstSZkOdcPD-wVybGV6xKqtEQk4B?usp=sharing

Potential applications of using our factuality models: 
- Social media analysis
- Belief level in text or documents
- Combination with sentiment to mine user's cognitive states


If you find our work useful, please cite us at:
```
@inproceedings{murzaku-etal-2023-towards,
    title = "Towards Generative Event Factuality Prediction",
    author = "Murzaku, John  and
      Osborne, Tyler  and
      Aviram, Amittai  and
      Rambow, Owen",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.44",
    doi = "10.18653/v1/2023.findings-acl.44",
    pages = "701--715",
    abstract = "We present a novel end-to-end generative task and system for predicting event factuality holders, targets, and their associated factuality values. We perform the first experiments using all sources and targets of factuality statements from the FactBank corpus. We perform multi-task learning with other tasks and event-factuality corpora to improve on the FactBank source and target task. We argue that careful domain specific target text output format in generative systems is important and verify this with multiple experiments on target text output structure. We redo previous state-of-the-art author-only event factuality experiments and also offer insights towards a generative paradigm for the author-only event factuality prediction task.",
}
```
