## Project Report: Emotion-Aware Chatbot

Code available at: 

https://drive.google.com/drive/folders/1rKa9wHlLbDrMA_N4QaDylWPylipOUeT3?usp=sharing

### Introduction

The project aims to develop a chatbot that is not only capable of generating meaningful responses but is also cognizant of the emotional content within a conversation. The chatbot's emotional awareness is projected to enhance the interaction quality, making it more contextually appropriate and human-like.

The model is built upon OpenAI's GPT-2 architecture, specifically the 'DialoGPT' variant which is specifically designed for generating conversational responses. This model is fine-tuned on the DailyDialog dataset to make the chatbot more attuned to daily conversation dynamics.

A crucial feature of this chatbot is its ability to interpret emotions. An auxiliary model, a DistilBert model trained for emotion detection, is used to analyze the emotional content of the conversation, thereby rendering the chatbot 'emotion-aware'.

### Code Explanation

The code can be divided into several key sections:

#### 1. Preprocessing

In the preprocessing stage, the DailyDialog dataset is loaded and preprocessed to include emotion tags in the dialogues. This is done using the `preprocess_and_encode` function which attaches emotion tags to both the user's and the chatbot's utterances. The function takes the dataset examples, processes the dialog and emotion fields, and returns encoded inputs ready for training.

```python
def preprocess_and_encode(examples):
    # Add emotion tags to both the user's and the agent's utterances
    dialog_with_emotions = []
    for dialog, emotion in zip(examples['dialog'], examples['emotion']):
        dialog_with_emotion = []
        for i in range(len(dialog)):
            emotion_tag = emotion_dict[emotion[i]]
            dialog_with_emotion.append(f"{emotion_tag}: {dialog[i]}")

        dialog_with_emotions.append(dialog_with_emotion)

    dialog_string = [" ".join(dialog) for dialog in dialog_with_emotions]

    encoded = tokenizer(dialog_string, truncation=True, padding='max_length', max_length=128)
    encoded['labels'] = encoded['input_ids'][:]

    return encoded
```

#### 2. Model Training

The chatbot is trained using the Hugging Face's `Trainer` class. The `TrainingArguments` are defined to specify training-related parameters such as the learning rate, number of training epochs, batch size, scheduler type, etc. The `seed` is also set for reproducibility purposes.

```python
training_args = TrainingArguments(
    output_dir="./DialoGPT-chatbot-emotion-aware",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=0,
    weight_decay=0.01,
    learning_rate=5e-5,  
    lr_scheduler_type=SchedulerType.COSINE, 
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True,
    save_strategy="steps",  
    save_steps=4000,                    
    evaluation_strategy="steps",        
    eval_steps=500,                     
    load_best_model_at_end=True,       
    metric_for_best_model="loss",
    seed=42,
)
```

#### 3. Emotion Detection

An auxiliary model, a DistilBert model, is used for emotion detection. The DistilBert model was fine-tuned on the DailyDialog dataset, exhibiting impressive performance with an accuracy of 0.928 on the validation set, and a slightly lower yet still robust accuracy of 0.854 on the test set, demonstrating its effectiveness in detecting emotions from dialogues. The weights are loaded into the model using the `load_state_dict` function. This model is used in the chat loop to interpret the emotional content of the conversation. 

```python
emotion_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=7)

# Load the model weights from a file
state_dict = torch.load('emotion_detection_model.pt')

# Apply the weights to the model
emotion_model.load_state_dict(state_dict)
```

#### 4. Chat Loop

The chat loop begins with the user input, which is encoded using the tokenizer and passed to the emotion model for emotion detection. Based on the detected emotion, an emotion tag is assigned to the user's message. The chatbot then generates a response that is decoded and printed out.

```python
for step in range(5):
    user_message = input(">> User:")

    # Prepare the message for input to the emotion model
    encoded_input = emotion_tokenizer(user_message, return_tensors='pt').to(device)
    
    # Get the detected emotion
    detected_emotion = emotion_model(encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'])
    emotion = torch.argmax(detected_emotion.logits)
    
    # Add emotion tag to the user's message
    new_user_input_ids = tokenizer.encode(emotion_dict[emotion.item()] + ": " + user_message + tokenizer.eos_token, return_tensors='pt')

    # Generate a response
    chat_history_ids = model.generate(bot_input_ids.to(device), max_length=50, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.7)
    
    # Decode the response
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_token=True)))
```

### Results 

To assess the effectiveness of chatbot in incorporating emotions, a model which does not incorporate emotion was also created, and a similar set of questions were asked from both models. Response A and Response B shown below are the baseline and emotion-aware DialoGPT models respectively.

Result A:
* User:You look very nice tonight!
* DialoGPT: It ’ s so nice and bright!
* User:What's your name?
* DialoGPT: My name is John.
* User:Where are you from?
* DialoGPT: We are from England.
* User:You like sushis?
* DialoGPT: Yes.
* User:Let's go out!
* DialoGPT: Let's go for a walk. 

Result B:

* User:You look very nice tonight!
DialoGPT: Thank you
* User:What's your name?
DialoGPT: It's Jean-Shann
* User:Where are you from?
DialoGPT: We're from China
* User:You like sushis?
DialoGPT: Yes, I do
* User:Let's go out!
DialoGPT: Yes, let's


As noticed, the emotion-aware model evidently improved the conversational flow and the appropriateness of responses when compared to the standard fine-tuned DialoGPT model. Result A, without emotion incorporation, revealed a couple of misplaced responses. For instance, in response to "You look very nice tonight!", the chatbot's answer, "It's so nice and bright!", seemed disconnected and irrelevant.

Result B, using the emotion-aware model, exhibited a noticeable enhancement in the conversational quality and relevance of the responses. For instance, in response to "You look very nice tonight!", the chatbot appropriately responded with "Thank you". The responses in Result B, in general, were more connected to the user's inputs and exhibited a clearer sense of context, highlighting the benefits of the emotion-aware approach. Therefore, the emotion-aware model seems to demonstrate better conversational aptitude, offering more engaging and contextually apt interactions.

### Conclusion

The emotion-aware chatbot is a novel way of incorporating emotional context into machine-generated responses. Through training on conversationally rich datasets and incorporating an emotion detection model, the chatbot can generate responses that not only answer the user's queries but also reflect the emotional content in the conversation, providing a more contextually enriched interaction.
