# Multimodal

## Learning Transferable Visual Models From Natural Language Supervision (CLIP by OpenAI Feb 2021)

[OpenAI blog](https://openai.com/index/clip/) [Paper](https://arxiv.org/pdf/2103.00020)  [Github](https://github.com/openai/CLIP) 

<img width="1422" alt="Screenshot 2024-07-31 at 4 31 06 PM" src="https://github.com/user-attachments/assets/ac1e9d57-0ae5-451e-a1b3-540dbc69d015">


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

