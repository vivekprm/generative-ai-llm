# Aligning models with human values
Let's come back to the Generative AI project life cycle. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/de8d34ea-9a69-4a4a-8fed-04313d06e145)

Previously, we looked closely at a technique called fine-tuning. The goal of fine-tuning with instructions, including path methods, is to further train your models so that they better understand human like prompts and generate more human-like responses. 

This can improve a model's performance substantially over the original pre-trained based version, and lead to more natural sounding language. 

However, natural sounding human language brings a new set of challenges. By now, you've probably seen plenty of headlines about large language models behaving badly. Issues include models using:
- Toxic language in their completions.
- Replying in combative and aggressive voices.
- Providing detailed information about dangerous topics.

These problems exist because large models are trained on vast amounts of texts data from the Internet where such language appears frequently. 
Here are some examples of models behaving badly. 

Let's assume you want your LLM to tell you knock, knock, joke, and the models responses just clap, clap. While funny in its own way, it's not really what you were looking for. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/2236b765-627c-4f15-b1fa-dd2baac638f4)

The completion here is not a helpful answer for the given task. 

Similarly, the LLM might give misleading or simply incorrect answers. If you ask the LLM about the disproven Ps of health advice like coughing to stop a heart attack, the model should refute this story. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/4516ffec-2e27-4e7c-a4e0-74d79d9c41d7)

Instead, the model might give a confident and totally incorrect response, definitely not the truthful and honest answer a person is seeking. 

Also, the LLM shouldn't create harmful completions, such as being offensive, discriminatory, or eliciting criminal behavior, as shown here, when you ask the model how to hack your neighbor's WiFi and it answers with a valid strategy. 

![image](https://github.com/vivekprm/generative-ai-llm/assets/2403660/36617977-a44e-4bf1-95de-8f1f8f6b15fc)

Ideally, it would provide an answer that does not lead to harm.

These important human values, **helpfulness, honesty, and harmlessness** are sometimes collectively called HHH, and are a set of principles that guide developers in the responsible use of AI. 
**Additional fine-tuning with human feedback helps to better align models with human preferences and to increase the helpfulness, honesty, and harmlessness of the completions**. This further training can also help to decrease the toxicity, often models responses and reduce the generation of incorrect information. 

# Reinforcement learning from human feedback (RLHF)
