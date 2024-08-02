# Multimodal
## BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models (from Salesforce June 2023)
<img width="1385" alt="Screenshot 2024-08-02 at 11 31 21 AM" src="https://github.com/user-attachments/assets/c2f895eb-5ee1-46ce-b356-99ead703767a">

## Learning Transferable Visual Models From Natural Language Supervision (CLIP by OpenAI Feb 2021)

[OpenAI blog](https://openai.com/index/clip/) [Paper](https://arxiv.org/pdf/2103.00020)  [Github](https://github.com/openai/CLIP) 

<img width="1422" alt="Screenshot 2024-07-31 at 4 31 06 PM" src="https://github.com/user-attachments/assets/ac1e9d57-0ae5-451e-a1b3-540dbc69d015">

### Key Concepts of Linear Probe in CLIP
In the context of CLIP (Contrastive Language–Image Pre-training) multimodal models, a **linear probe** is a technique used to evaluate the quality of the representations learned by the model. This involves using a simple linear classifier to test how well the features extracted by a pre-trained model can be used to perform specific tasks, such as image classification or text classification, without further training the model itself.

1. **Representation Quality Evaluation**:
   - The main purpose of a linear probe is to assess the quality of the representations (features) learned by a model during pre-training. By using a linear classifier, we can determine how much useful information is contained in these features for a given task.

2. **Simple Linear Classifier**:
   - A linear probe typically involves adding a linear layer (often a single fully connected layer) on top of the frozen features output by the pre-trained model. This linear layer is then trained on a specific task, such as classifying images into different categories.

3. **Fixed Feature Extraction**:
   - The pre-trained model is not updated during the linear probing process. Only the linear classifier is trained. This approach isolates the evaluation to the effectiveness of the features themselves, rather than the capacity of the entire model to learn new tasks.

4. **Efficiency**:
   - Linear probing is computationally efficient because it only requires training a small number of parameters (those in the linear layer) compared to fine-tuning the entire model. This makes it a quick way to assess how well pre-trained representations transfer to new tasks.

5. **Performance Indicator**:
   - The accuracy of the linear probe on a specific task serves as an indicator of the richness and generalizability of the learned representations. High performance with a linear probe suggests that the features are linearly separable and informative.

### Application in CLIP

- **CLIP's Architecture**: CLIP models jointly learn image and text representations by aligning them in a shared embedding space through a contrastive learning objective. This shared space is used for zero-shot classification and other tasks.
  
- **Using Linear Probes**:
  - After training with the CLIP approach, a linear probe can be added to evaluate how well the image or text features can be used for downstream tasks, such as classifying images into predefined categories without additional task-specific training of the backbone model.

### Example Process

1. **Pre-train CLIP**: The model is pre-trained on a large dataset of image-text pairs to align image and text embeddings.

2. **Feature Extraction**: Extract features from images or text using the pre-trained CLIP model.

3. **Train Linear Probe**: Use the extracted features as input to a linear classifier. Train this classifier on a labeled dataset relevant to the task of interest.

4. **Evaluate**: Assess the performance of the linear classifier to determine the effectiveness of the CLIP features for the task.

By using linear probes, researchers and practitioners can quickly gauge the potential of pre-trained multimodal models like CLIP for a variety of tasks, offering insights into the model's transfer learning capabilities.


### Vision Transformer (ViT) Architecture
The Vision Transformer (ViT) is an **encoder-only architecture**. It adapts the transformer architecture, originally designed for natural language processing tasks, to process image data.

1. **Patch Embedding**:
   - The input image is divided into fixed-size patches (e.g., 16x16 pixels). Each patch is then flattened into a vector and linearly projected into an embedding space.
   - These patch embeddings are analogous to word embeddings in NLP, where each patch serves as a "token."

2. **Position Embedding**:
   - Since transformers do not inherently capture positional information, a position embedding is added to each patch embedding to retain spatial information about where each patch is located in the original image.

3. **Transformer Encoder Layers**:
   - The core of the ViT consists of multiple layers of transformer encoders. Each encoder layer includes:
     - **Multi-Head Self-Attention**: Allows each patch to attend to every other patch, enabling the model to learn contextual relationships across the entire image.
     - **Feedforward Neural Network**: Processes the combined information from the self-attention mechanism.
     - **Layer Normalization** and **Residual Connections**: Used to stabilize and enhance the training process.

4. **Classification Head**:
   - A special token, often referred to as the [CLS] token, is prepended to the sequence of patch embeddings. The output corresponding to this token from the final encoder layer is used for classification tasks.
   - A linear layer is typically applied to the [CLS] token’s output to produce the final predictions.

### Why ViT is Encoder-Only

- **Focus on Representation Learning**: The encoder layers in ViT are designed to learn rich representations of the input image by capturing dependencies and relationships between patches.
- **No Generation Component**: Unlike encoder-decoder architectures used in tasks like sequence-to-sequence translation (e.g., in NLP), ViT does not require a decoder because it is not generating new sequences (such as translating text or generating captions). Instead, it focuses on understanding and classifying the input image.

### Comparison with Other Architectures

- **Encoder-Decoder Architectures**: Typically used in tasks requiring the transformation of one sequence into another, such as machine translation. The encoder processes the input to a latent representation, and the decoder generates the output sequence from this representation.
- **Decoder-Only Architectures**: Commonly used in autoregressive models where the focus is on generating sequences based on prior inputs, such as language models like GPT.

### Applications of ViT

ViT has been applied successfully in various vision tasks, including:
- Image Classification
- Object Detection (with modifications or additional components)
- Segmentation (often combined with other architectures or methods)

In summary, the Vision Transformer (ViT) is an encoder-only architecture that leverages the transformer’s powerful self-attention mechanism to learn comprehensive representations of images by treating image patches as tokens.

### CLIP's Training Objective
CLIP (Contrastive Language–Image Pre-training) is a multimodal model developed by OpenAI that learns to associate images and text by aligning their representations in a shared embedding space. Its training objective is designed to leverage the natural pairing of images and text found in web data to create powerful joint representations.

The primary objective of CLIP is to maximize the similarity between paired image and text representations while minimizing the similarity between unpaired images and texts. This is achieved through a contrastive learning approach:

#### Contrastive Learning

1. **Embedding Space**:
   - CLIP uses two separate neural networks to encode images and text into a shared embedding space. The image encoder is typically based on a convolutional neural network (CNN) or a vision transformer (ViT), while the text encoder is based on a transformer architecture.

2. **Contrastive Loss Function**:
   - The core of CLIP's training objective is a contrastive loss, specifically the **InfoNCE loss** (Noise Contrastive Estimation). This loss encourages the model to produce similar embeddings for matching image-text pairs and dissimilar embeddings for non-matching pairs.
   - For each image-text pair \((I, T)\), the objective is to increase the cosine similarity between their embeddings \((E_I, E_T)\) if they are paired, and decrease the similarity if they are not.
3. **Softmax with Temperature Scaling**
<img width="705" alt="Screenshot 2024-08-02 at 11 20 07 AM" src="https://github.com/user-attachments/assets/30dbb78d-fd42-4858-afad-4c0489714514">

4. **Bidirectional Contrastive Loss**:
   - CLIP's training loss is bidirectional, meaning it simultaneously computes two contrastive objectives: one where the image is used to predict the text and another where the text is used to predict the image. This ensures that both image-to-text and text-to-image associations are learned effectively.

5. **Zero-shot Learning**:
   - By training with this objective on a large dataset of diverse image-text pairs, CLIP learns robust representations that generalize well to new tasks. It can perform zero-shot learning, meaning it can classify images into new categories without additional task-specific training, simply by associating images with descriptive text prompts.

### Benefits of CLIP's Training Objective

- **Scalability**: CLIP leverages large-scale web data without needing explicit labeling beyond natural language descriptions, making it scalable across diverse datasets.
- **Generalization**: The joint training of images and text embeddings allows CLIP to generalize well to new tasks, achieving competitive performance across a wide range of benchmarks with zero-shot learning.
- **Multimodal Understanding**: By learning from both images and text, CLIP can understand and generate content that is semantically coherent across these modalities, enhancing its ability to perform tasks that require a deeper understanding of visual and textual information together.

CLIP's training objective enables it to align visual and textual information effectively, allowing it to perform well in a variety of tasks without the need for traditional supervised fine-tuning.
