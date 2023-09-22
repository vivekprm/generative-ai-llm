# Model optimizations for deployment
Let's talk about the things you'll have to consider to integrate your model into applications. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/a64c88d9-636d-4b78-b84c-9562307f2096)

There are a number of important questions to ask at this stage. The first set is related to how your LLM will function in deployment. So how fast do you need your model to generate completions? What compute budget do you have available? And are you willing to trade off model performance for improved inference speed or lower storage? 

The second set of questions is tied to additional resources that your model may need. Do you intend for your model to interact with external data or other applications? And if so, how will you connect to those resources? 

Lastly, there's the question of how your model will be consumed. What will the intended application or API interface that your model will be consumed through look like? Let's start by exploring a few methods that can be used to optimize your model before deploying it for inference. 

While we could dedicate several lessons to this topic, the aim of this section is to offer you an introduction to the most important optimization techniques. 

Large language models present inference challenges in terms of computing and storage requirements, as well as ensuring low latency for consuming applications. These challenges persist whether you're deploying on premises or to the cloud, and become even more of an issue when deploying to edge devices. One of the primary ways to improve application performance is to reduce the size of the LLM. This can allow for quicker loading of the model, which reduces inference latency.

However, the challenge is to reduce the size of the model while still maintaining model performance. Some techniques work better than others for generative models, and there are tradeoffs between accuracy and performance. You'll learn about three techniques in this section. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/1c157e38-d8e8-4883-a835-dfca396badf8)

Distillation uses a larger model, the teacher model, to train a smaller model, the student model. You then use the smaller model for inference to lower your storage and compute budget. 

Similar to quantization aware training, post training quantization transforms a model's weights to a lower precision representation, such as a 16- bit floating point or eight bit integer. As you learned in week one of the course, this reduces the memory footprint of your model. 

The third technique, Model Pruning, removes redundant model parameters that contribute little to the model's performance. Let's talk through each of these options in more detail.

## Model Distillation
Model Distillation is a technique that focuses on having a **larger teacher model train a smaller student model**. The student model learns to statistically mimic the behavior of the teacher model, either just in the final prediction layer or in the model's hidden layers as well. You'll focus on the first option here. 

You start with your fine tune LLM as your teacher model and create a smaller LLM for your student model. You freeze the teacher model's weights and use it to generate completions for your training data. At the same time, you generate completions for the training data using your student model. The knowledge distillation between teacher and student model is achieved by minimizing a loss function called the **distillation loss**. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/3d1100c0-2be3-458a-b93a-ce7a43ed69f7)

To calculate this loss, distillation uses the probability distribution over tokens that is produced by the **teacher model's softmax layer**. Now, the teacher model is already fine tuned on the training data. So the probability distribution likely closely matches the ground truth data and won't have much variation in tokens. That's why Distillation applies a little trick adding a temperature parameter to the softmax function. 

As we learned in lesson one, a **higher temperature increases the creativity of the language the model generates**. With a temperature parameter greater than one, the probability distribution becomes broader and less strongly peaked. This softer distribution provides you with a set of tokens that are similar to the ground truth tokens. In the context of Distillation, the teacher model's output is often referred to as soft labels and the student model's predictions as soft predictions. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/ee51b185-b41d-4d0e-8253-d6716356776d)

In parallel, you train the student model to generate the correct predictions based on your ground truth training data. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/12094253-2ec1-499b-ae14-156874787bff)

Here, you don't vary the temperature setting and instead use the standard softmax function. Distillation refers to the student model outputs as the hard predictions and hard labels. 

The loss between these two is the student loss. The combined distillation and student losses are used to update the weights of the student model via back propagation. The key benefit of distillation methods is that the smaller student model can be used for inference in deployment instead of the teacher model. 

In practice, distillation is not as effective for generative decoder models. It's **typically more effective for encoder only models**, such as **Burt** that have a lot of representation redundancy. 

**Note that with Distillation, you're training a second, smaller model to use during inference**. You aren't reducing the model size of the initial LLM in any way. 

Let's have a look at the next model optimization technique that actually reduces the size of your LLM. 

## Post-Training Quantization
You were introduced to the second method, quantization, back in week one in the context of training. Specifically Quantization Aware Training, or QAT for short.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/5a6ed97e-8f87-4f4a-b4ef-fb8f0a10d9ba)

However, after a model is trained, you can perform post training quantization, or PTQ for short to optimize it for deployment. PTQ transforms a model's weights to a lower precision representation, such as 16-bit floating point or 8-bit integer. To reduce the model size and memory footprint, as well as the compute resources needed for model serving, quantization can be applied to just the model weights or to both weights and activation layers. 

In general, quantization approaches that include the activations can have a higher impact on model performance. Quantization also requires an extra calibration step to statistically capture the dynamic range of the original parameter values. As with other methods, there are tradeoffs because sometimes quantization results in a small percentage reduction in model evaluation metrics. However, that reduction can often be worth the cost savings and performance gains. 

## Pruning
The last model optimization technique is pruning. At a high level, the goal is to reduce model size for inference by eliminating weights that are not contributing much to overall model performance. These are the weights with values very close to or equal to zero. Note that some pruning methods require full retraining of the model, while others fall into the category of parameter efficient fine tuning, such as LoRA. There are also methods that focus on post-training Pruning. 

In theory, this reduces the size of the model and improves performance. In practice, however, there may not be much impact on the size and performance if only a small percentage of the model weights are close to zero. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/0811055a-0844-4e50-9ee6-cca68b367016)

Quantization, Distillation and Pruning all aim to reduce model size to improve model performance during inference without impacting accuracy. Optimizing your model for deployment will help ensure that your application functions well and provides your users with the best possible experience sense.

# Generative AI Project Lifecycle Cheat Sheet
To help you plan out different stages of the generative AI project life cycle, this cheat sheet provide some indication of the time and effort required for each phase 
of work. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/341de39c-dd32-4b72-846d-b39fc318ef66)

As you saw earlier, pre-training a large language model can be a huge effort. This stage is the most complex you'll face because of the model architecture decisions, the large amount of training data required, and the expertise needed. Remember though, that in general, you will start your development work with an existing foundation model. You'll probably be able to skip this stage. 

If you're working with a foundation model, you'll likely start to assess the model's performance through prompt engineering, which requires less technical expertise, and no additional training of the model. If your model isn't performing as you need, you'll next think about prompt tuning and fine tuning. Depending on your use case, performance goals, and compute budget, the methods you'll try could range from full fine-tuning to parameter efficient fine tuning techniques like laura or prompt tuning. Some level of technical expertise is required for this work. But since fine-tuning can be very successful with a relatively small training dataset, this phase could potentially be completed in a single day. 

Aligning your model using reinforcement learning from human feedback can be done quickly, once you have your train reward model. You'll likely see if you can use an existing reward model for this work, as you saw in this week's lab. However, if you have to train a reward model from scratch, it could take a long time because of the effort involved to gather human feedback. 

Finally, optimization techniques you learned about in the last video, typically fall in the middle in terms of complexity and effort, but can proceed quite quickly assuming the changes to the model don't impact performance too much. After working through all of these steps, you have hopefully trained in tuned a gray LLM that is working well for your specific use case, and is optimized for deployment. 

# Using the LLM in applications
Although all the training, tuning and aligning techniques you've explored can help you build a great model for your application. There are some broader challenges with large language models that can't be solved by training alone. Let's take a look at a few examples. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/f3398b51-21d2-4724-8c92-c3b585adf549)

One issue is that the internal knowledge held by a model cuts off at the moment of pretraining. For example, if you ask a model that was trained in early 2022 who the British Prime Minister is, it will probably tell you Boris Johnson. This knowledge is out of date. The model does not know that Johnson left office in late 2022 because that event happened after its training. 

Models can also struggle with complex math. If you prompt a model to behave like a calculator, it may get the answer wrong, depending on the difficulty of the problem. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/a88c442c-c23d-40d9-8f7a-32f154352fb8)

Here, you ask the model to carry out a division problem. The model returns a number close to the correct answer, but it's incorrect. Note the LLMs do not carry out mathematical operations. They are still just trying to predict the next best token based on their training, and as a result, can easily get the answer wrong. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/50ec0d12-1630-4e31-aca0-ec2b7484b1d5)

Lastly, one of the best known problems of LLMs is their tendency to generate text even when they don't know the answer to a problem. This is often called hallucination, and here you can see the model clearly making up a description of a nonexistent plant, the Martian Dunetree. Although there is still no definitive evidence of life on Mars, the model will happily tell you otherwise. 

In this section, we'll learn about some techniques that we can use to help our LLM overcome these issues by connecting to external data sources and applications. We'll have a bit more work to do to be able to connect our LLM to these external components and fully integrate everything for deployment within our application. 

Our application must manage the passing of user input to the large language model and the return of completions. This is often done through some type of orchestration library. This layer can enable some powerful technologies that augment and enhance the performance of the LLM at runtime. By providing access to external data sources or connecting to existing APIs of other applications. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/d35c8a7a-391c-40dd-b7da-5e369c24e733)

One implementation example is Langchain, which you'll learn more about later in this lesson. Let's start by considering how to connect LLMs to external data sources.

## Retrieval Augmented Generation (RAG)
Retrieval Augmented Generation, or RAG for short, is a framework for building LLM powered systems that make use of external data sources. And applications to overcome some of the limitations of these models. RAG is a great way to overcome the knowledge cutoff issue and help the model update its understanding of the world. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/1eee31d0-826e-42b6-805f-7cf1f9da7f9d)

While you could retrain the model on new data, this would quickly become very expensive. And require repeated retraining to regularly update the model with new knowledge. A more flexible and less expensive way to overcome knowledge cutoffs is to **give your model access to additional external data at inference time**. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/ec8ee63a-a3fe-4867-afcf-1b46f3391e2a)

RAG is useful in any case where you want the language model to have access to data that it may not have seen. This could be new information documents not included in the original training data, or proprietary knowledge stored in your organization's private databases. Providing your model with external information, can improve both the relevance and accuracy of its completions. 

Let's take a closer look at how this works. Retrieval augmented generation isn't a specific set of technologies, but rather a framework for providing LLMs access to data they did not see during training. A number of different implementations exist, and the one you choose will depend on the details of your task and the format of the data you have to work with. 

Here you'll walk through the implementation discussed in one of the earliest papers on RAG by researchers at Facebook, originally published in 2020. At the heart of this implementation is a model component called the Retriever, which consists of a query encoder and an external data source.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/5a9fa525-f937-424b-8b5b-f0b7ecea869d)

The encoder takes the user's input prompt and encodes it into a form that can be used to query the data source. In the Facebook paper, the external data is a vector store, which we'll discuss in more detail shortly. But it could instead be a SQL database, CSV files, or other data storage format. These two components are trained together to find documents within the external data that are most relevant to the input query. The Retriever returns the best single or group of documents from the data source and combines the new information with the original user query. The new expanded prompt is then passed to the language model, which generates a completion that makes use of the data.

Let's take a look at a more specific example. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/f76d6531-b5c9-42be-ad52-63a4ccd78b66)

Imagine you are a lawyer using a large language model to help you in the discovery phase of a case. A RAG architecture can help you ask questions of a corpus of documents, for example, previous court filings. Here you ask the model about the plaintiff named in a specific case number.

The prompt is passed to the query encoder, which encodes the data in the same format as the external documents. And then searches for a relevant entry in the corpus of documents. Having found a piece of text that contains the requested information, the Retriever then combines the new text with the original prompt. 

The expanded prompt that now contains information about the specific case of interest is then passed to the LLM. The model uses the information in the context of the prompt to generate a completion that contains the correct answer. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/993c76ee-d393-4c98-9d7d-e47071467b86)

The use case you have seen here is quite simple and only returns a single piece of information that could be found by other means. But imagine the power of RAG to be able to generate summaries of filings or identify specific people, places and organizations within the full corpus of the legal documents. Allowing the model to access information contained in this external data set greatly increases its utility for this specific use case.

In addition to overcoming knowledge cutoffs, RAG also helps you avoid the problem of the model hallucinating when it doesn't know the answer. 

### RAG Integrates with multiple datasources
RAG architectures can be used to integrate multiple types of external information sources. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/0119ef8c-8ae9-43f7-b2b3-8698eba3f6e9)

You can augment large language models with access to local documents, including private wikis and expert systems. Rag can also enable access to the Internet to extract information posted on web pages, for example, Wikipedia. 
By encoding the user input prompt as a SQL query, RAG can also interact with databases. 
Another important data storage strategy is a Vector Store, which contains vector representations of text. This is a particularly useful data format for language models, since internally they work with vector representations of language to generate text. Vector stores enable a fast and efficient kind of relevant search based on similarity. 

## Data Preparation For Vector Store For RAG
Note that implementing RAG is a little more complicated than simply adding text into the large language model. There are a couple of key considerations to be aware of, starting with:
- The size of the context window. Most text sources are too long to fit into the limited context window of the model, which is still at most just a few thousand tokens. Instead, the external data sources are chopped up into many chunks, each of which will fit in the context window. Packages like Langchain can handle this work for you.
  
![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/eb8e8d99-503f-4ed0-b1aa-aeacd40b3d7c)
  
- Second, the data must be available in a format that allows for easy retrieval of the most relevant text. Recall that large language models don't work directly with text, but instead create vector representations of each token in an embedding space.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/c39ee9cc-82bc-4927-9a8d-8639444995db)

These embedding vectors allow the LLM to identify semantically related words through measures such as cosine similarity, which you learned about earlier.
Rag methods take the small chunks of external data and process them through the large language model, to create embedding vectors for each. These new representations of the data can be stored in structures called vector stores, which allow for fast searching of datasets and efficient identification of semantically related text.

## Vector Database Search
Vector databases are a particular implementation of a vector store where each vector is also identified by a key. This can allow, for instance, the text generated by RAG to also include a citation for the document from which it was received. So you've seen how access to external data sources can help a model overcome limits to its internal knowledge. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/c0f43116-04d1-4754-87fa-d2312735b37c)

By providing up to date relevant information and avoiding hallucinations, you can greatly improve the experience of using your application for your users. 

