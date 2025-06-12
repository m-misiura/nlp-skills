# Understanding Attention in Transformer Models
# This notebook demonstrates how attention mechanisms work in transformer-based models like ModernBERT
# with practical visualization examples
# %%
import torch
from transformers import AutoTokenizer, AutoModel, utils
from bertviz import model_view, head_view, neuron_view
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import HTML, display
import pandas as pd
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import math

# %%
# Silence unnecessary warnings
utils.logging.set_verbosity_error()


# %%
def explain_attention_math():
    """Visualize and explain the mathematics behind self-attention"""
    print(
        """
    ## Attention Formula

    The core of the transformer architecture is the self-attention mechanism, defined as:
    
    Attention(Q, K, V) = softmax(QK^T / √d_k)V
    
    Where:
    - Q (Query), K (Key), V (Value) are matrices
    - d_k is the dimension of the key vectors (scaling factor)
    - QK^T computes token similarity scores
    - softmax normalizes these scores into a probability distribution
    - Final multiplication with V produces the weighted value vectors
    """
    )

    # Create a simple example to demonstrate attention calculation
    seq_len = 3  # Very small sequence for simplicity
    d_model = 4  # Small embedding size for demonstration

    # Create sample embeddings for a 3-token sequence
    print("\n## Step-by-Step Attention Calculation Example")
    print("Let's calculate attention for a simple 3-token sequence:")
    print("Token 1: 'The', Token 2: 'cat', Token 3: 'sits'")

    # Create dummy embeddings for demonstration
    np.random.seed(42)  # For reproducibility
    embeddings = torch.tensor(np.random.randn(seq_len, d_model), dtype=torch.float32)

    # Normally these would be learned projections, but for simplicity:
    W_q = torch.tensor(np.random.randn(d_model, d_model), dtype=torch.float32)
    W_k = torch.tensor(np.random.randn(d_model, d_model), dtype=torch.float32)
    W_v = torch.tensor(np.random.randn(d_model, d_model), dtype=torch.float32)

    # Step 1: Project embeddings to Q, K, V
    Q = embeddings @ W_q
    K = embeddings @ W_k
    V = embeddings @ W_v

    print("\nStep 1: Project token embeddings to Query, Key, Value matrices")
    print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")

    # Format for display with token labels
    token_labels = ["Token 1 ('The')", "Token 2 ('cat')", "Token 3 ('sits')"]
    Q_df = pd.DataFrame(Q.numpy(), index=token_labels)
    K_df = pd.DataFrame(K.numpy(), index=token_labels)
    V_df = pd.DataFrame(V.numpy(), index=token_labels)

    print("\nQuery (Q) values:")
    display(Q_df)
    print("\nKey (K) values:")
    display(K_df)
    print("\nValue (V) values:")
    display(V_df)

    # Step 2: Compute attention scores
    attention_scores = Q @ K.T
    print("\nStep 2: Compute attention scores (Q × K^T)")
    scores_df = pd.DataFrame(
        attention_scores.numpy(), index=token_labels, columns=token_labels
    )
    display(scores_df)

    # Step 3: Scale the scores
    d_k = K.shape[1]
    attention_scores_scaled = attention_scores / math.sqrt(d_k)
    print(f"\nStep 3: Scale scores by √d_k = {math.sqrt(d_k):.2f}")
    scaled_df = pd.DataFrame(
        attention_scores_scaled.numpy(), index=token_labels, columns=token_labels
    )
    display(scaled_df)

    # Step 4: Apply softmax to get attention weights
    attention_weights = F.softmax(attention_scores_scaled, dim=-1)
    print("\nStep 4: Apply softmax to get probability distribution")
    weights_df = pd.DataFrame(
        attention_weights.numpy(), index=token_labels, columns=token_labels
    )
    display(weights_df.style.background_gradient(cmap="Blues").format("{:.4f}"))

    # Step 5: Multiply by values
    attention_output = attention_weights @ V
    print("\nStep 5: Get weighted value vectors (attention_weights × V)")
    output_df = pd.DataFrame(attention_output.numpy(), index=token_labels)
    display(output_df)

    # Visualize the attention weights
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights.numpy(),
        annot=True,
        xticklabels=token_labels,
        yticklabels=token_labels,
        cmap="viridis",
        fmt=".4f",
    )
    plt.title("Attention Weights")
    plt.xlabel("Keys")
    plt.ylabel("Queries")
    plt.tight_layout()
    plt.show()

    print(
        """
    Key Insights:
    1. Each token attends to all tokens (including itself)
    2. The attention weights form a probability distribution (sum to 1) for each query
    3. The final output for each token is a weighted sum of all token values
    4. This creates a context-aware representation for each token
    """
    )


explain_attention_math()


# %%
# 12. Multi-Head Attention Mechanics
def explain_multi_head_attention():
    """Explain how multi-head attention works mathematically"""
    print(
        """
    ## Multi-Head Attention
    
    Instead of performing a single attention function, transformers use multiple attention heads:
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    Benefits:
    1. Each head can focus on different aspects of the input
    2. Some heads may focus on syntax, others on semantics
    3. This creates a more robust representation
    """
    )

    # Visualize concept of multiple attention heads
    plt.figure(figsize=(12, 8))

    # Simple 4-token example
    tokens = ["The", "cat", "sits", "down"]
    n_tokens = len(tokens)

    # Create sample attention patterns for 4 different heads
    np.random.seed(42)

    # Head 1: Local attention (diagonal-focused)
    head1 = np.eye(n_tokens) + 0.1 * np.random.rand(n_tokens, n_tokens)
    head1 = head1 / head1.sum(axis=1, keepdims=True)

    # Head 2: Next token attention
    head2 = np.zeros((n_tokens, n_tokens))
    for i in range(n_tokens):
        if i < n_tokens - 1:
            head2[i, i + 1] = 0.7
        head2[i, i] = 0.3

    # Head 3: Global attention to first token
    head3 = np.zeros((n_tokens, n_tokens))
    head3[:, 0] = 0.7
    for i in range(n_tokens):
        head3[i, i] = 0.3

    # Head 4: Random pattern
    head4 = np.random.rand(n_tokens, n_tokens)
    head4 = head4 / head4.sum(axis=1, keepdims=True)

    heads = [head1, head2, head3, head4]
    titles = [
        "Head 1: Local Focus",
        "Head 2: Next Token Focus",
        "Head 3: First Token Focus",
        "Head 4: Mixed Pattern",
    ]

    for i, (head, title) in enumerate(zip(heads, titles)):
        plt.subplot(2, 2, i + 1)
        sns.heatmap(
            head,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            xticklabels=tokens,
            yticklabels=tokens,
        )
        plt.title(title)

    plt.tight_layout()
    plt.show()

    print(
        """
    In practice, each head has its own set of learnable W_Q, W_K, and W_V projection matrices.
    After each head produces its output, these are concatenated and projected once more 
    through W_O to produce the final output.
    """
    )


print("\n### 12. Multi-Head Attention Mechanics ###")
explain_multi_head_attention()


# %%
# 13. Attention Scaling Effects
def demonstrate_temperature_effects():
    """Show how scaling affects attention weights"""
    print(
        """
    ## Effect of Temperature/Scaling on Attention
    
    The scaling factor (√d_k) in attention is crucial. Let's see what happens 
    when we vary this parameter (sometimes called "temperature"):
    """
    )

    # Create a sample attention score matrix (before softmax)
    raw_scores = torch.tensor(
        [
            [1.0, 0.2, 0.1, 0.05],  # First token attends mostly to itself
            [0.3, 1.0, 0.4, 0.2],  # Second token attends to itself and third
            [0.1, 0.5, 1.0, 0.3],  # Third token attends to itself and second
            [0.2, 0.1, 0.4, 1.0],  # Fourth token attends to itself and third
        ]
    )

    tokens = ["The", "cat", "sits", "down"]

    # Try different scaling factors
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    plt.figure(figsize=(15, 10))

    for i, temp in enumerate(temperatures):
        # Apply temperature scaling
        scaled_scores = raw_scores / temp

        # Apply softmax
        attention_weights = F.softmax(scaled_scores, dim=-1)

        # Plot
        plt.subplot(len(temperatures), 1, i + 1)
        sns.heatmap(
            attention_weights,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            xticklabels=tokens,
            yticklabels=tokens,
        )
        plt.title(f"Temperature = {temp}")

    plt.tight_layout()
    plt.show()

    print(
        """
    Key observations:
    1. Lower temperature (like 0.1) → Sharper focus, more concentrated attention
    2. Higher temperature (like 5.0) → More uniform attention distribution
    3. √d_k in standard attention provides a good balance
    
    The scaling factor prevents extremely small gradients during training when 
    input dimensionality is large.
    """
    )


print("\n### 13. Temperature Effects in Attention ###")
demonstrate_temperature_effects()


# %%
# 14. Positional Encoding
def explain_positional_encoding():
    """Explain and visualize positional encodings"""
    print(
        """
    ## Positional Encodings
    
    Since transformers process all tokens in parallel, they need position information:
    
    PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    
    Where:
    - pos is the position of the token in the sequence
    - i is the dimension index
    - d_model is the embedding dimension
    """
    )

    # Generate positional encodings for visualization
    def get_positional_encoding(seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    # Generate positional encodings for 30 positions and 64 dimensions
    pos_enc = get_positional_encoding(30, 64)

    # Visualize the encodings
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(pos_enc, cmap="viridis")
    plt.title("Positional Encodings")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position in Sequence")

    # Plot a few dimensions to show the wavelength pattern
    plt.subplot(1, 2, 2)
    dimensions = [0, 1, 2, 3, 15, 31, 47, 63]  # Sample dimensions
    for dim in dimensions:
        plt.plot(pos_enc[:, dim], label=f"Dim {dim}")
    plt.legend()
    plt.title("Encoding Values by Position")
    plt.xlabel("Position")
    plt.ylabel("Encoding Value")

    plt.tight_layout()
    plt.show()

    print(
        """
    Key properties of positional encodings:
    1. Each position has a unique encoding pattern
    2. Similar positions have similar encodings
    3. The encoding pattern repeats at different frequencies
    4. This creates a unique "address" for each position without requiring recurrence
    5. The model can attend to relative positions by learning patterns in these encodings
    """
    )


print("\n### 14. Positional Encoding Visualized ###")
explain_positional_encoding()


# %%
# 15. Manual Implementation
def implement_self_attention():
    """Implement self-attention from scratch"""
    print(
        """
    ## Implementing Self-Attention Manually
    
    Let's implement the self-attention mechanism from scratch to understand it fully:
    """
    )

    # Function to implement self-attention
    def self_attention(x, mask=None):
        """Basic self-attention implementation"""
        # x: input tensor of shape (batch_size, seq_len, d_model)
        # For simplicity, we'll use the same projection for Q, K, V
        batch_size, seq_len, d_model = x.shape

        # 1. Linear projections
        # In a real transformer, these would be separate learned projections
        q = x  # (batch_size, seq_len, d_model)
        k = x  # (batch_size, seq_len, d_model)
        v = x  # (batch_size, seq_len, d_model)

        # 2. Calculate attention scores
        scores = torch.bmm(q, k.transpose(1, 2))  # (batch_size, seq_len, seq_len)

        # 3. Scale scores
        scaled_scores = scores / math.sqrt(d_model)

        # 4. Apply mask (optional)
        if mask is not None:
            scaled_scores = scaled_scores.masked_fill(mask == 0, -1e9)

        # 5. Apply softmax
        attention_weights = F.softmax(scaled_scores, dim=-1)

        # 6. Apply attention weights
        output = torch.bmm(attention_weights, v)  # (batch_size, seq_len, d_model)

        return output, attention_weights

    # Create a simple example
    batch_size = 1
    seq_len = 4
    d_model = 8

    # Random input embeddings
    x = torch.randn(batch_size, seq_len, d_model)

    # Apply self-attention
    output, weights = self_attention(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")

    # Visualize attention weights
    plt.figure(figsize=(8, 6))
    sns.heatmap(weights[0].detach().numpy(), annot=True, fmt=".2f", cmap="viridis")
    plt.title("Self-Attention Weights")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.show()

    # Now show the code for reference
    print(
        """
    ```python
    def self_attention(x, mask=None):
        # x: input tensor of shape (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        # 1. Linear projections
        q = x  # In real implementation: x @ W_q
        k = x  # In real implementation: x @ W_k
        v = x  # In real implementation: x @ W_v
        
        # 2. Calculate attention scores
        scores = torch.bmm(q, k.transpose(1, 2))
        
        # 3. Scale scores
        scaled_scores = scores / math.sqrt(d_model)
        
        # 4. Apply mask (optional)
        if mask is not None:
            scaled_scores = scaled_scores.masked_fill(mask == 0, -1e9)
        
        # 5. Apply softmax
        attention_weights = F.softmax(scaled_scores, dim=-1)
        
        # 6. Apply attention weights
        output = torch.bmm(attention_weights, v)
        
        return output, attention_weights
    ```
    """
    )


print("\n### 15. Implementing Self-Attention Manually ###")
implement_self_attention()


# %%
# 17. Attention in Different Domains
def attention_comparison():
    """Compare attention patterns across different types of inputs"""
    print(
        """
    ## Attention Patterns Across Input Types
    
    Different types of text elicit different attention patterns:
    """
    )

    examples = {
        "Question": "What is the capital of France?",
        "Statement": "Paris is the capital of France.",
        "List": "Apples, bananas, oranges, and pears.",
        "Code": "def hello():\n    print('Hello, world!')",
        "Math": "The formula is E = mc².",
    }

    # Analyze each example
    for label, text in examples.items():
        print(f"\nAnalyzing: {label}")
        print(f"Text: '{text}'")
        visualize_attention(text)


print("\n### 17. Attention Across Different Text Types ###")
attention_comparison()

# %%
# Set up model and tokenizer
model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(
    model_id, output_attentions=True, output_hidden_states=True
)


# Helper function for attention visualization
def visualize_attention(text, attention_type="model"):
    """
    Visualize attention patterns for a given text

    Parameters:
    - text: input text to analyze
    - attention_type: "model" for overall model view, "head" for individual attention heads
    """
    inputs = tokenizer.encode(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])

    # Get model outputs with attention
    outputs = model(inputs)
    attention = outputs.attentions

    print(f"Input text: '{text}'")
    print(f"Tokenized into: {tokens}")

    if attention_type == "model":
        return model_view(attention, tokens)
    elif attention_type == "head":
        return head_view(attention, tokens)
    else:
        raise ValueError("attention_type must be 'model' or 'head'")


# %%
# 1. Basic Attention Visualization
print("### 1. Basic Attention Visualization ###")
example_text = "The cat sat on the mat."
visualize_attention(example_text)
visualize_attention(example_text, attention_type="head")
# %%
# 2. Visualizing Attention for Different Linguistic Phenomena
# %%
# 2.1 Coreference resolution
print("\n### 2.1 Coreference Resolution ###")
coref_example = "John went to the store because he needed groceries."
visualize_attention(coref_example)
visualize_attention(coref_example, attention_type="head")


# %%
# 3. Analyzing Attention Across Layers
def analyze_attention_across_layers(text):
    """Analyze how attention changes across model layers"""
    inputs = tokenizer.encode(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])

    outputs = model(inputs)
    attention = outputs.attentions

    # Number of layers
    num_layers = len(attention)

    print(f"Analyzing attention across {num_layers} layers for: '{text}'")

    # For each layer, compute average attention per token
    plt.figure(figsize=(12, num_layers * 2))

    for layer_idx in range(num_layers):
        # Average across attention heads
        layer_attention = attention[layer_idx].mean(dim=1).squeeze().detach().numpy()

        plt.subplot(num_layers, 1, layer_idx + 1)
        sns.heatmap(
            layer_attention, cmap="viridis", xticklabels=tokens, yticklabels=tokens
        )
        plt.title(f"Layer {layer_idx+1} Average Attention")
        plt.tight_layout()

    plt.show()


print("\n### 3. Attention Patterns Across Layers ###")
analyze_attention_across_layers("The transformer model processes text efficiently.")


# %%
# 4. Multi-Head Attention Analysis
def analyze_attention_heads(text, layer_idx=0):
    """Analyze different attention heads in a specific layer"""
    inputs = tokenizer.encode(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])

    outputs = model(inputs)
    attention = outputs.attentions

    # Get the specified layer
    layer_attention = attention[layer_idx].squeeze().detach().numpy()
    num_heads = layer_attention.shape[0]

    print(f"Analyzing {num_heads} attention heads in layer {layer_idx+1} for: '{text}'")

    # Plot each attention head
    plt.figure(figsize=(20, 20))
    rows = int(np.ceil(num_heads / 3))

    for head_idx in range(num_heads):
        plt.subplot(rows, 3, head_idx + 1)
        sns.heatmap(
            layer_attention[head_idx],
            cmap="viridis",
            xticklabels=tokens,
            yticklabels=tokens,
        )
        plt.title(f"Head {head_idx+1}")

    plt.tight_layout()
    plt.show()


print("\n### 4. Multi-Head Attention Analysis ###")
analyze_attention_heads("Different heads capture different linguistic patterns.")


# %%
# 6. Comparing Attention Across Different Inputs
def compare_attention(text1, text2, layer_idx=0, head_idx=0):
    """Compare attention patterns between two different inputs"""
    inputs1 = tokenizer.encode(text1, return_tensors="pt")
    inputs2 = tokenizer.encode(text2, return_tensors="pt")

    tokens1 = tokenizer.convert_ids_to_tokens(inputs1[0])
    tokens2 = tokenizer.convert_ids_to_tokens(inputs2[0])

    outputs1 = model(inputs1)
    outputs2 = model(inputs2)

    attention1 = outputs1.attentions[layer_idx][0, head_idx].detach().numpy()
    attention2 = outputs2.attentions[layer_idx][0, head_idx].detach().numpy()

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    sns.heatmap(attention1, cmap="viridis", xticklabels=tokens1, yticklabels=tokens1)
    plt.title(f"Attention: '{text1}'")

    plt.subplot(1, 2, 2)
    sns.heatmap(attention2, cmap="viridis", xticklabels=tokens2, yticklabels=tokens2)
    plt.title(f"Attention: '{text2}'")

    plt.tight_layout()
    plt.show()


print("\n### 6. Comparing Attention Across Different Inputs ###")
compare_attention(
    "The cat chased the mouse across the room.",
    "The dog barked at the mailman on Tuesday.",
)


# %%
# 7. Attention on Linguistic Features
def analyze_linguistic_features():
    """Analyze attention patterns on various linguistic features"""
    features = {
        "Verb-Subject Agreement": "The boy runs to the store. The boys run to the store.",
        "Prepositional Phrases": "The book on the table is red. I walked to the store on Main Street.",
        "Conjunction": "I like apples and oranges, but I don't like bananas.",
        "Passive vs Active Voice": "The ball was thrown by the boy. The boy threw the ball.",
    }

    for feature, text in features.items():
        print(f"\n7. Attention on {feature}:")
        visualize_attention(text)


analyze_linguistic_features()


# %%
# 8. Interactive Attention Analysis
def token_attention_analysis(text):
    """Analyze how each token attends to other tokens"""
    inputs = tokenizer.encode(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])

    outputs = model(inputs)
    attention = outputs.attentions

    # Average across all layers and heads
    avg_attention = (
        torch.mean(torch.cat([layer.squeeze() for layer in attention]), dim=0)
        .detach()
        .numpy()
    )

    # Create a DataFrame for better readability
    attention_df = pd.DataFrame(avg_attention, index=tokens, columns=tokens)

    print(f"Token attention analysis for: '{text}'")
    print("\nHow each token attends to others (averaged across all layers and heads):")

    # Format as percentage with color gradient (higher percentages are darker)
    styled_df = attention_df.style.background_gradient(cmap="Blues").format("{:.2%}")
    display(styled_df)


print("\n### 8. Token-by-Token Attention Analysis ###")
token_attention_analysis(
    "The transformer architecture revolutionized natural language processing."
)

# %%
# 9. Attention Pattern Evolution During Training
print("\n### 9. How Attention Patterns Evolve ###")
print(
    """
In an untrained model, attention patterns are often uniform or random.
As training progresses:

1. Early layers learn to focus on local patterns and syntax
2. Middle layers develop attention to relevant context
3. Later layers specialize in task-specific patterns
4. Some heads become specialized for linguistic phenomena:
   - Syntactic dependencies
   - Entity tracking
   - Coreference resolution
   - Semantic relationships

This notebook uses a pre-trained model where these patterns have already developed.
"""
)

# %%
# 10. Practical Applications of Attention Analysis
print("\n### 10. Practical Applications of Attention Analysis ###")
print(
    """
Understanding attention patterns helps in:

1. Model interpretability - explaining why a model made a particular prediction
2. Debugging model behavior - identifying when attention focuses on irrelevant tokens
3. Model distillation - determining which heads are most important
4. Linguistic analysis - discovering patterns that models learn about language
5. Model improvement - designing better attention mechanisms based on observed patterns
"""
)
