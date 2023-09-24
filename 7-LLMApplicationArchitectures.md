Let's look at the building blocks for creating LLM powered applications. 

# Building Generative Applications 
You'll require several key components to create end-to-end solutions for your applications, starting with the infrastructure layer. 

This layer provides the compute, storage, and network to serve up your LLMs, as well as to host your application components. You can make use of your on-premises infrastructure for this or have it provided for you via on-demand and pay-as-you-go Cloud services. 

Next, you'll include the large language models you want to use in your application. These could include foundation models, as well as the models you have adapted to your specific task. The models are deployed on the appropriate infrastructure for your inference needs. Taking into account whether you need real-time or near-real-time interaction with the model. 

You may also have the need to retrieve information from external sources, such as those discussed in the retrieval augmented generation section. 

Your application will return the completions from your large language model to the user or consuming application. Depending on your use case, you may need to implement a mechanism to capture and store the outputs. For example, you could build the capacity to store user completions during a session to augment the fixed contexts window size of your LLM. 

You can also gather feedback from users that may be useful for additional fine-tuning, alignment, or evaluation as your application matures. Next, you may need to use additional tools and frameworks for large language models that help you easily implement some of the techniques discussed in this course. 

As an example, you can use LangChain's built-in libraries to implement techniques like PAL, REACT or chain of thought prompting. You may also utilize model hubs which allow you to centrally manage and share models for use in applications. 

In the final layer, you typically have some type of user interface that the application will be consumed through, such as a website or a rest API. This layer is where you'll also include the security components required for interacting with your application. 

At a high level, this architecture stack represents the various components to consider as part of your generative AI applications. 

Your users, whether they are human end-users or other systems that access your application through its APIs, will interact with this entire stack. 

# Summary
As you can see, the model is typically only one part of the story in building end-to-end generative AI applications. In summary:

We saw how to align your models with human preferences, such as helpfulness, harmlessness, and honesty by fine-tuning using a technique called **reinforcement learning with human feedback**, or RLHF for short. Given the popularity of RLHF, there are many existing RL reward models and human alignment datasets available, enabling you to quickly start aligning your models. 

In practice, RLHF is a very effective mechanism that you can use to improve the alignment of your models, reduce the toxicity of their responses, and let you use your models more safely in production. You also saw important techniques to optimize your model for inference by reducing the size of the model through distillation, quantization, or pruning. This minimizes the amount of hardware resources needed to serve your LLMs in production. 

Lastly, we explored ways that we can help our model perform better in deployment through structured prompts and connections to external data sources and applications. LLMs can play an amazing role as the reasoning engine in an application, exploiting their intelligence to power exciting, useful applications. Frameworks like LangChain are making it possible to quickly build, deploy, and test LLM powered applications, and it's a very exciting time for developers. 

# Amazon Sagemaker JumpStart
Amazon Sagemaker JumpStart, can help us get into production quickly and operate at scale.

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/8e846c92-d203-4471-95c1-d4626c904e0c)

Here's the application stack that we explored previously. As we saw, building an LLM-powered application requires multiple components. 

Sagemaker JumpStart is a model hub, and it allows you to quickly deploy foundation models that are available within the service, and integrate them into your own applications. The JumpStart service also provides an easy way to fine-tune and deploy models. JumpStart covers many parts of this diagram, including the infrastructure, the LLM itself, the tools and frameworks, and even an API to invoke the model. 

In contrast to the models that we worked with in the labs, JumpStart models require GPUs to fine tune and deploy. And keep in mind, these GPUs are subject to on-demand pricing and you should refer to the Sagemaker pricing page before selecting the compute you want to use. Also, please be sure to delete the Sagemaker model endpoints when not in use and follow cost monitoring best practices to optimize cost. 

Sagemaker JumpStart is accessible from the AWS console, or through Sagemaker studio. For this brief tour, I'll start in studio then choose JumpStart from the main screen. I could optionally choose JumpStart from the left-hand menu, and select models, notebooks, and solutions as well. After I click on "JumpStart", you'll see different categories that include end-to-end solutions across different use cases, as well as a number of foundation models for different modalities that you can easily deploy, as well as fine-tune, where yes is indicated under the fine-tuning option. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/1c806aa0-eaf6-4ef6-bce7-c0ba00bf61c3)

Let's look at an example we're all familiar with after working through the course, which is the Flan-T5 model. We've specifically been using the base variant in the course to minimize the resources needed by the lab environments. 

However, as you can see here, you can also utilize other variants of Flan-T5 through JumpStart depending on your needs. You'll also notice the Hugging Face logo here, which means they're actually coming directly from Hugging Face. And AWS has worked with Hugging Face to the point where you can easily, with just a few clicks, deploy, or fine-tune the model. 
