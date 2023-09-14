# Considerations for choosing a model
Once you have scoped out your use case, and determined how you'll need the LLM to work within your application, your next step is to select a model to work with. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/287dabc1-9dd7-4379-8c3e-695c4c8ee03b)

Your first choice will be to either work with an existing model, or train your own from scratch. There are specific circumstances where training your own model from scratch might be advantageous.

In general, however, you'll begin the process of developing your application using an existing foundation model. Many open-source models are available for members of the AI community like you to use in your application. The developers of some of the major frameworks for building generative AI applications like Hugging Face and PyTorch, have curated hubs where you can browse these models.

A really useful feature of these hubs is the inclusion of model cards, that describe important details including the best use cases for each model, how it was trained, and known limitations.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/089f5650-d2f5-4e01-b643-f5c4d2c51a67)

With this knowledge in hand, you'll find it easier to navigate the model hubs and find the best model for your use case.

Let's take a high-level look at the initial training process for LLMs. This phase is often referred to as pre-training. LLMs encode a deep statistical representation of language. This understanding is developed during the models pre-training phase when the model learns from vast amounts of unstructured textual data. This can be gigabytes, terabytes, and even petabytes of text. This data is pulled from many sources, including scrapes off the Internet and corpora of texts that have been assembled specifically for training language models.

In this self-supervised learning step, the model internalizes the patterns and structures present in the language. These patterns then enable the model to complete its training objective, which depends on the architecture of the model, as you'll see shortly. During pre-training, the model weights get updated to minimize the loss of the training objective. 

The encoder generates an embedding or vector representation for each token. Pre-training also requires a large amount of compute and the use of GPUs. Note, when you scrape training data from public sites such as the Internet, you often need to process the data to increase quality, address bias, and remove other harmful content. As a result of this data quality curation, often only 1-3% of tokens are used for pre-training. You should consider this when you estimate how much data you need to collect if you decide to pre-train your own model.

We saw earlier that there were three variance of the transformer model; 
- encoder-only
- encoder-decoder models,
- and decode-only.

Each of these is trained on a different objective, and so learns how to carry out different tasks. 

Encoder-only models are also known as **Autoencoding models**, and they are pre-trained using **masked language modeling**. Here, tokens in the input sequence or randomly mask, and the training objective is to predict the mask tokens in order to reconstruct the original sentence. This is also called a **denoising objective**. 

Autoencoding models spilled bi-directional representations of the input sequence, meaning that the model has an understanding of the full context of a token and not just of the words that come before. 

Encoder-only models are ideally suited to task that benefit from this bi-directional contexts. You can use them to carry out sentence classification tasks, for example, sentiment analysis or token-level tasks like named entity recognition or word classification. Some well-known examples of an autoencoder model are **BERT** and **RoBERTa**. 

Now, let's take a look at decoder-only or **autoregressive models**, which are pre-trained using **causal language modeling**. Here, the training objective is to predict the next token based on the previous sequence of tokens. Predicting the next token is sometimes called **full language modeling** by researchers. Decoder-based autoregressive models, mask the input sequence and can only see the input tokens leading up to the token in question. The model has no knowledge of the end of the sentence. The model then iterates over the input sequence one by one to predict the following token. 

In contrast to the encoder architecture, this means that the context is unidirectional. By learning to predict the next token from a vast number of examples, the model builds up a statistical representation of language. Models of this type make use of the decoder component off the original architecture without the encoder. **Decoder-only models are often used for text generation**, although larger decoder-only models show strong zero-shot inference abilities, and can often perform a range of tasks well. Well known examples of decoder-based autoregressive models are **GPT** and **BLOOM**.

The final variation of the transformer model is the **sequence-to-sequence model** that uses both the encoder and decoder parts off the original transformer architecture. The exact details of the pre-training objective vary from model to model. A popular sequence-to-sequence model **T5**, pre-trains the encoder using span corruption, which masks random sequences of input tokens. Those mass sequences are then replaced with a unique Sentinel token, shown here as x. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/21542639-99f7-4304-acff-7f44421c8c78)

Sentinel tokens are special tokens added to the vocabulary, but do not correspond to any actual word from the input text. The decoder is then tasked with reconstructing the mask token sequences auto-regressively. The output is the Sentinel token followed by the predicted tokens. 

You can use sequence-to-sequence models for 
- Translation
- Summarization
- Question-answering.

They are generally useful in cases where you have a body of texts as both input and output. Besides **T5**, which you'll use in the labs in this course, another well-known encoder-decoder model is **BART**, not bird. 

# Summary
Here's a quick comparison of the different model architectures and the targets off the pre-training objectives. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/160eb709-43dd-4c13-8ab6-0e9505a12a83)

Autoencoding models are pre-trained using masked language modeling. They correspond to the encoder part of the original transformer architecture, and are often used with:
- Sentence classification
- Token classification

Autoregressive models are pre-trained using causal language modeling. Models of this type make use of the decoder component of the original transformer architecture, and often used for 
- text generation

Sequence-to-sequence models use both the encoder and decoder part off the original transformer architecture. The exact details of the pre-training objective vary from model to model. The T5 model is pre-trained using span corruption. Sequence-to-sequence models are often used for 
- Translation
- Summarization
- Question-answering.

# Model Size vs Time
![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/1606a149-88ec-4465-9141-958ba60313c1)

Can we just keep adding parameters to increase performance and make models smarter? Where could this model growth lead? While this may sound great, it turns out that training these enormous models is difficult and very expensive, so much so that it may be infeasible to continuously train larger and larger models.
