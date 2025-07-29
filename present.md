Ok, so here comes the part for finetuned llm. So this workflow is just the lower part of the whole one

 

Page18

So for our finetuned LLM methodology, this ipage shows our main architecture:

So we pick up an llm such as LLmam 3 8b as our model backbone, and basically apply 2 changes. 

 

Firstly we freeze LLM backbone and apply Lora to increase finetune efficiency. To be short Lora is a technique to inject small modules to llm attention and other layer to replace original trainable parameters, replace them with low rank parameter matrix, to reduce the total cost of finetuning. 

 

Secondly,

You can refer to the model component on the right. Initial input is our feature data, we utilize them to generate prompt, combined with context and convert them as chat format, for LLM input. 

 

On the top we can see how LLM output, originally LLM utilized lm head to generate logit of next token, and recursively generate tokens onwards. because it’s no longer a language model task, we replace the language model head with regression head, which is MLP layers, to utilize information from last hidden state of last token, and output a numerical value

 

Page19

This page is some of finetuned LLM Configurations

We choose 500 stocks with top liquidity as dataset, and sixteen features which were chosen from caishen feature engineering. For prompts, Finetuned llm shares the same prompt generator with Hoste, the main different is we only consider regression at this time, so we only keep regression in task prompt and data definition. We try to simplify the prompt to keep it efficient for finetune, so we only apply zero shot template, and discard other complicated logic. 

 

And another useful step in data processing is that we do resampling to balance dataset. This kind of imbalance comes from too many zero sample and extreme large value, so according to our experience in caishen project, this kind of resample is necessary. So as shown here, we bucket data point by different criterion and apply weights to each bucket.

 

And we search for the optimal hyperparameters in order to get best model performance. The table display most important ones, but we also did a lot search on others

  

 

 

 

 

 

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method for large language models (LLMs). It freezes the original model weights and injects trainable low-rank matrices into key layers (e.g., attention), reducing trainable parameters by ~10,000x. LoRA maintains performance while drastically lowering memory usage, enabling fine-tuning on consumer GPUs. It supports multi-task learning via task-specific adapters and is widely used in models like LLaMA and GPT. LoRA is compatible with quantization for further efficiency gains.