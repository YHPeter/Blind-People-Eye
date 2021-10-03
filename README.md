<!-- # Train and Deploy Image Caption via Tensorflow Serving with Flask on CentOS 7 -->
<!-- # Train and Deploy Image Caption via Tensorflow Serving with Flask on CentOS 7 -->
# Blind People's Eyes
*"What's happened?" the blind people think*

*Then he takes a photo and clicks the screen*

*"**A man flying a kite in a park!**" said by phone.*

*"OK Got it!"*

# AI to help blind people to see the world!
## How we built it

### 1. Model Training: (Python)

- Training Environment: CentOS 7 with one NVIDIA-V100
- Model Constructed: **Transformer Picture Encoder + Transformer Text Decoder**
- Dataset: COCO 2014

### 2. Front-end: (Java)

- Andriod (TexttoSpeech, Vibrate... )

### 3. Back-end: (Python)

- RESTful API with Flask
- Tensorflow Serving Docker
- Training Environment: CentOS 7

## Challenges we ran into & What we learned

### 1. Model Training

- Positional Embedding for 1D and 2D inputs
- Design the model inputs signature for deploying with Tensorflow Serving
- Select randomly from the validation dataset to monitor the robustness of the model, whether overfit in training data

### 2. Front-end

- Use fundamental java libraries to POST files, by constricting fundamental HTTP request structure
- Order the camera within App and save images with temporary space and address
- The relationship between the multithread and Intend

### 3. Back-end

- Tensorflow Serving Deployment in two Linux systems
- Restful API Design with Flask and cooperating with TensorFlow Serving

## What's next for Blind People's Eye

### Model Side

- Expand Dataset, by covering more events pictures.
- Use Alberta and some pretrianed model for decoder, could have better text generating effect
- Use distilled models, such as electra, distilled-bert, thus the model can become smaller enough to inference on the phone
- Concentrate the picture decoder and voice decoder model, generate more fluent audio

### User Interface Side

- Need more clear instructions for people to click buttons
- Modify vibration style, give different rhythms for different situations

## Sample Image

### Test1

![Test1](server/test.jpg)

```json
API RESPONSE:
{
  "text": "a man riding skis down a snow covered bike"
}
```

### Test2

![Test1](server/test2.jpg)

```json
API RESPONSE: 
{
  "text": "a group of people walking down a street next to tall buildings"
}
```
