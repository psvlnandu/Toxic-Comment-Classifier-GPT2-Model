# Toxic Comment Classifier using Fine-Tuned GPT-2

This project is a multi-label text classification model designed to detect and categorize six different types of toxicity in online comments. It utilizes a pre-trained `gpt2-medium` model, which was fine-tuned on the Jigsaw Toxic Comment Classification Challenge dataset using the Hugging Face `transformers` library.

## Dataset

[Jigsaw dataset from Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
- It's a collection of thousands of comments from Wikipedia talk pages.
- Each comment is given a score from 0 to 1 for six different types of toxicity: toxic, severe_toxic, obscene, threat, insult, and identity_hate. The more details can be found on kaggle so i am skipping here

## Training: A Step-by-Step Guide
**Step-1: Setup ( The tools)**

- **`preprocess_text`**: A function to clean the raw text.
- **`compute_metrics`**: A function that tells the `Trainer` how to calculate F1, ROC AUC, and accuracy during evaluation.
- **`MultiLabelTrainer`**: Your custom trainer that handles the specific loss calculation for a multi-label problem.

**Step-2:  The `main()` Function (The Core Logic)**
- Split the data into train & eval
- Load model & tokenizer: You load the pre-trained `gpt2-medium` model and its corresponding tokenizer, making sure to configure them for your specific task (e.g., setting the padding token and dropout).
- Create an instance on TrainingArgs
- You create MultiLabelTrainer giving its model, data, tokenize and the Training Arguments
- You call trainer.train()

**Step-3: deep dive into Training Args**
- `output_dir` is folder where all the Ops will be saved including checkpoint & final model
- `num_train_epochs`:2
- `learning_rate` Controls how big of a "step" the model takes when learning. It's one of the most important hyperparameters.
Too high, and the model might learn too fast and overshoot the best solution. Too low, and it might take forever to train. Your value of `2e-5` is a great, standard starting point for fine-tuning.
- **`per_device_train_batch_size` & `per_device_eval_batch_size`**:How many examples the model looks at in a single step (one batch) during training and evaluation. 
A larger batch size can speed up training but uses significantly more GPU memory. Your setting of `16` is a good choice for a T4 GPU.
- **`weight_decay`**: a regularization technique to prevent overfitting. `0.01` is a common and effective value.
- **`eval_strategy` & `save_strategy`**: 
Tells the `Trainer` *when* to run an evaluation and *when* to save a checkpoint. & `"epoch"` means it will do both at the end of every epoch.
- `load_best_model_at_end & metric_for_best_model`: `"f1"` is an excellent choice for this task as it balances precision and recall.
- `report_to`
reports the weights to wandb
- `fp16`
enables the mixed precision training. uses combination of 16bit and 32bit floating point numbers to calculations much faster on modern GPUs (like the T4) and use less memory, without sacrificing much precision.

**Evaluation Metrics**
- eval_f1
- eval_roc_auc
- eval_accuracy
- eval_loss

## How to Use

You can easily test this model using the `pipeline` function from the `transformers` library.

```python
from transformers import pipeline

# Load my model from the Hugging Face Hub
classifier = pipeline("text-classification", model="raavip/gpt2-toxic-comment-classifier")

# Test with a comment
comment = "This is a horrible and insulting comment."
results = classifier(comment)
print(results)
```
## Performance

The final model achieved the following performance on the evaluation set:

| Metric    | Score    |
|-----------|----------|
| F1 Score  | 0.795    |
| ROC AUC   | 0.918    |
| Accuracy  | 0.923    |
| Eval Loss | 0.040    |

**Model**

  The model is public available on Hugging face mode card [here](https://huggingface.co/raavip/gpt2-toxic-comment-classifier)
  
  Checkout my [Spaces for full fine tuning code](https://huggingface.co/spaces/raavip/FineTuningModels)
  
