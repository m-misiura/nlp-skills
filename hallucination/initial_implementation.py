# %% Step 0: Import libraries
import torch
import numpy as np
import networkx as nx
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# %% Step 1: Load GPT-2 model and tokenizer
model_name = "openai-community/gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()

print(f"Model loaded: {model_name}")
print(f"Number of layers: {model.config.n_layer}")
print(f"Number of attention heads: {model.config.n_head}")

# %% Step 2: extract attention maps from text
def extract_attention_maps(text):
    """Extract attention maps from GPT-2"""
    # Tokenize text
    # do we need BOS and EOS??????
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    tokens = tokenizer.tokenize(text)
    
    # Get model outputs with attention
    with torch.no_grad():
        outputs = model(**inputs)
        # Stack attention maps: [layers, heads, seq_len, seq_len]
        attention_maps = torch.stack(outputs.attentions).squeeze(1)
    
    print(f"Input text: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Attention shape: {attention_maps.shape}")
    print(f"  - {attention_maps.shape[0]} layers")
    print(f"  - {attention_maps.shape[1]} heads") 
    print(f"  - {attention_maps.shape[2]} sequence length")
    
    return attention_maps, tokens

# Test with example text
sample_text = "The capital of France is Paris."
attention_maps, tokens = extract_attention_maps(sample_text)

# %% Step 3: Visualize attention maps
def visualize_attention(attention_maps, tokens, layer_idx=5, head_idx=0):
    """Visualize attention heatmap for specific layer/head"""
    attn_matrix = attention_maps[layer_idx, head_idx].numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attn_matrix, cmap='Blues')
    plt.title(f'Attention Map - Layer {layer_idx}, Head {head_idx}')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    
    # Add token labels if sequence is short enough
    if len(tokens) <= 15:
        plt.xticks(range(len(tokens)), tokens, rotation=45)
        plt.yticks(range(len(tokens)), tokens)
    
    plt.colorbar(label='Attention Score')
    plt.tight_layout()
    plt.show()
    
    print(f"Showing attention for Layer {layer_idx}, Head {head_idx}")
    print(f"Matrix shape: {attn_matrix.shape}")
    print(f"Each cell shows how much token i attends to token j")

# Visualize attention
visualize_attention(attention_maps, tokens, layer_idx=5, head_idx=0)

# %% Step 4a: Compute Laplacian matrix and eigenvalues
def compute_laplacian_eigenvalues(attention_maps, k=5):
    """
    Compute top-k Laplacian eigenvalues for each layer/head
    
    Formula: L = D - A
    where D is degree matrix, A is attention matrix
    """
    n_layers, n_heads, seq_len, _ = attention_maps.shape
    all_eigenvals = []
    
    print(f"Processing {n_layers} layers × {n_heads} heads...")
    
    for layer in range(n_layers):
        for head in range(n_heads):
            # Get attention matrix A
            A = attention_maps[layer, head].numpy()
            
            # Compute degree matrix D
            D = np.zeros_like(A)
            for i in range(seq_len):
                # Sum of attention TO token i
                attention_to_i = np.sum(A[:, i])
                # Number of tokens attending to i
                num_attending = np.sum(A[:, i] != 0)
                
                if num_attending > 0:
                    D[i, i] = attention_to_i / num_attending
            
            # Compute Laplacian: L = D - A
            L = D - A
            
            # Extract eigenvalues (diagonal entries)
            eigenvals = np.diag(L)
            
            # Sort and take top-k
            eigenvals_sorted = np.sort(eigenvals)[::-1]
            top_k_eigenvals = eigenvals_sorted[:k]
            
            # Pad if necessary
            if len(top_k_eigenvals) < k:
                top_k_eigenvals = np.pad(top_k_eigenvals, (0, k - len(top_k_eigenvals)))
            
            all_eigenvals.extend(top_k_eigenvals)
    
    feature_vector = np.array(all_eigenvals)
    print(f"Extracted {len(feature_vector)} features")
    print(f"Formula: {n_layers} layers × {n_heads} heads × {k} eigenvalues = {len(feature_vector)}")
    
    return feature_vector

# Compute eigenvalues for our sample
k = 5  # Top-5 eigenvalues per head
features = compute_laplacian_eigenvalues(attention_maps, k=k)
print(f"Feature vector shape: {features.shape}")
print(f"First 10 eigenvalues: {features[:10]}")

# %% Step 4b: additional verification of paper properties
def verify_paper_properties(attention_maps, layer_idx=5, head_idx=0):
    """Verify the properties mentioned in the paper"""
    
    A = attention_maps[layer_idx, head_idx].numpy()
    
    print("Verifying paper's stated properties:")
    
    # 1. Row-stochastic (each row sums to 1)
    row_sums = np.sum(A, axis=1)
    is_row_stochastic = np.allclose(row_sums, 1.0)
    print(f"1. Row-stochastic (rows sum to 1): {is_row_stochastic}")
    print(f"   Row sums: {row_sums[:3]}... (showing first 3)")
    
    # 2. Lower triangular 
    upper_triangle = np.triu(A, k=1)  # k=1 excludes diagonal
    is_lower_triangular = np.allclose(upper_triangle, 0)
    print(f"2. Lower triangular (causal mask): {is_lower_triangular}")
    if not is_lower_triangular:
        print(f"   Max upper triangle value: {np.max(upper_triangle):.6f}")
    
    # 3. Non-negative
    is_non_negative = np.all(A >= 0)
    print(f"3. Non-negative values: {is_non_negative}")
    if not is_non_negative:
        print(f"   Min value: {np.min(A):.6f}")
    
    # 4. Verify Laplacian is lower triangular
    D = np.zeros_like(A)
    for i in range(A.shape[0]):
        attention_to_i = np.sum(A[:, i])
        num_attending = np.sum(A[:, i] != 0)
        if num_attending > 0:
            D[i, i] = attention_to_i / num_attending
    
    L = D - A
    L_upper = np.triu(L, k=1)
    L_is_lower_triangular = np.allclose(L_upper, 0)
    print(f"4. Laplacian is lower triangular: {L_is_lower_triangular}")
    
    # 5. Diagonal entries are eigenvalues (since L is lower triangular)
    eigenvals_diag = np.diag(L)
    print(f"5. Eigenvalues (diagonal entries): {eigenvals_diag[:5]}... (first 5)")
    
    return A, D, L

A, D, L = verify_paper_properties(attention_maps)

# %% Step 5a: Visualize the Laplacian computation process
def show_laplacian_computation(attention_maps, layer_idx=5, head_idx=0):
    """Show step-by-step Laplacian computation"""
    A = attention_maps[layer_idx, head_idx].numpy()
    seq_len = A.shape[0]
    
    # Step 1: Attention matrix A
    print(f"Step 1: Attention Matrix A")
    print(f"Shape: {A.shape}")
    print(f"A[0,:5] = {A[0, :5]}")  # First row, first 5 columns
    
    # Step 2: Degree matrix D
    D = np.zeros_like(A)
    print(f"\nStep 2: Degree Matrix D")
    for i in range(min(3, seq_len)):  # Show first 3 tokens
        attention_to_i = np.sum(A[:, i])
        num_attending = np.sum(A[:, i] != 0)
        if num_attending > 0:
            D[i, i] = attention_to_i / num_attending
        print(f"Token {i}: attention_sum={attention_to_i:.3f}, count={num_attending}, d_ii={D[i,i]:.3f}")
    
    # Step 3: Laplacian L = D - A
    L = D - A
    print(f"\nStep 3: Laplacian L = D - A")
    print(f"L[0,:5] = {L[0, :5]}")
    
    # Step 4: Eigenvalues
    eigenvals = np.diag(L)
    eigenvals_sorted = np.sort(eigenvals)[::-1]
    print(f"\nStep 4: Eigenvalues (diagonal of L)")
    print(f"All eigenvalues: {eigenvals[:5]}... (showing first 5)")
    print(f"Top-5 eigenvalues: {eigenvals_sorted[:5]}")
    
    # Visualize matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im1 = axes[0].imshow(A, cmap='Blues')
    axes[0].set_title('Attention Matrix A')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(D, cmap='Reds') 
    axes[1].set_title('Degree Matrix D')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(L, cmap='RdBu_r')
    axes[2].set_title('Laplacian L = D - A')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    return A, D, L, eigenvals

A, D, L, eigenvals = show_laplacian_computation(attention_maps)

# %% Step 5B Step 5b: Visualize Attention as Explicit Graph
def visualize_attention_graph(attention_matrix, tokens, threshold=0.1):
    """Visualize attention as an explicit graph"""
    
    print(f"Converting attention matrix to graph representation...")
    print(f"Threshold: {threshold} (only showing edges with attention > {threshold})")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes (tokens)
    for i, token in enumerate(tokens):
        G.add_node(i, label=token)
    
    # Add edges (attention scores above threshold)
    edge_count = 0
    for i in range(attention_matrix.shape[0]):
        for j in range(attention_matrix.shape[1]):
            weight = attention_matrix[i, j]
            if weight > threshold:  # Only show significant attention
                G.add_edge(i, j, weight=weight)
                edge_count += 1
    
    print(f"✓ Graph created:")
    print(f"  - Nodes (tokens): {G.number_of_nodes()}")
    print(f"  - Edges (attention > {threshold}): {G.number_of_edges()}")
    print(f"  - Total possible edges: {attention_matrix.shape[0] * attention_matrix.shape[1]}")
    
    # Visualize
    plt.figure(figsize=(15, 10))
    
    # Create layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=2000, alpha=0.8)
    
    # Draw edges with thickness proportional to attention
    edges = G.edges()
    if edges:
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        
        # Normalize edge weights for visualization
        normalized_weights = [w/max_weight * 5 for w in weights]
        
        nx.draw_networkx_edges(G, pos, width=normalized_weights, 
                              alpha=0.6, edge_color='gray', 
                              connectionstyle="arc3,rad=0.1")
    
    # Add labels
    labels = {i: f"{i}:\n{token}" for i, token in enumerate(tokens)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    plt.title(f"Attention as Graph (Layer {layer_idx}, Head {head_idx})\n"
              f"Nodes = Tokens, Edges = Attention Scores > {threshold}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Show some graph statistics
    print(f"\n📊 Graph Statistics:")
    if G.number_of_edges() > 0:
        print(f"  - Average node degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
        print(f"  - Graph density: {nx.density(G):.3f}")
        
        # Show strongest connections
        edge_weights = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
        edge_weights.sort(key=lambda x: x[2], reverse=True)
        
        print(f"  - Strongest attention connections:")
        for i, (u, v, weight) in enumerate(edge_weights[:5]):
            print(f"    {i+1}. {tokens[u]} → {tokens[v]}: {weight:.3f}")
    
    return G

# Use the same layer/head from Step 5
layer_idx, head_idx = 5, 0
attn_matrix = attention_maps[layer_idx, head_idx].numpy()

# Visualize as graph
print(f"Visualizing attention from Layer {layer_idx}, Head {head_idx} as a graph:")
graph = visualize_attention_graph(attn_matrix, tokens, threshold=0.1)

# Optional: Try different thresholds to see how the graph changes
print(f"\n" + "="*30)
print("COMPARING DIFFERENT THRESHOLDS")
print("="*30)

thresholds = [0.05, 0.1, 0.2]
plt.figure(figsize=(18, 6))

for i, threshold in enumerate(thresholds):
    plt.subplot(1, 3, i+1)
    
    # Create graph with this threshold
    G = nx.DiGraph()
    for j, token in enumerate(tokens):
        G.add_node(j, label=token)
    
    for row in range(attn_matrix.shape[0]):
        for col in range(attn_matrix.shape[1]):
            if attn_matrix[row, col] > threshold:
                G.add_edge(row, col, weight=attn_matrix[row, col])
    
    # Simple visualization
    pos = nx.spring_layout(G, k=2)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800, alpha=0.7)
    
    if G.edges():
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights], alpha=0.5)
    
    labels = {j: tokens[j][:3] for j in range(len(tokens))}  # Shortened labels
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title(f'Threshold: {threshold}\nEdges: {G.number_of_edges()}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# %% Step 6: Create training data
# Create example sentences
factual_texts = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius.",
    "The Earth orbits around the Sun.", 
    "Shakespeare wrote Hamlet.",
    "The Pacific Ocean is the largest ocean.",
]

hallucinated_texts = [
    "The capital of France is London.",
    "Water boils at 50 degrees Celsius.", 
    "The Sun orbits around the Earth.",
    "Shakespeare wrote Pride and Prejudice.",
    "The Atlantic Ocean is the largest ocean.",
]

# Combine texts and labels
train_texts = factual_texts + hallucinated_texts
train_labels = [0] * len(factual_texts) + [1] * len(hallucinated_texts)  # 0=factual, 1=hallucinated

print(f"Created {len(train_texts)} training examples")
print(f"  - {len(factual_texts)} factual statements (label=0)")
print(f"  - {len(hallucinated_texts)} hallucinated statements (label=1)")
print("\nExamples:")
for i, (text, label) in enumerate(zip(train_texts[:4], train_labels[:4])):
    label_name = "Factual" if label == 0 else "Hallucinated"
    print(f"{i+1}. [{label_name}] {text}")

# %% Step 7: Extract features from all training texts  
def extract_features_batch(texts, k=5):
    """Extract Laplacian eigenvalue features from multiple texts"""
    all_features = []
    
    print(f"Extracting features from {len(texts)} texts...")
    for i, text in enumerate(texts):
        # Get attention maps
        attention_maps, _ = extract_attention_maps(text)
        
        # Compute eigenvalues
        features = compute_laplacian_eigenvalues(attention_maps, k=k)
        all_features.append(features)
        
        if (i + 1) % 2 == 0:
            print(f"  Processed {i+1}/{len(texts)} texts")
    
    feature_matrix = np.array(all_features)
    print(f"✓ Feature matrix shape: {feature_matrix.shape}")
    return feature_matrix

# Extract features for all training texts
X_train = extract_features_batch(train_texts, k=5)
y_train = np.array(train_labels)

print(f"Training data:")
print(f"  X_train shape: {X_train.shape} (samples × features)")
print(f"  y_train shape: {y_train.shape} (samples)")
print(f"  Label distribution: {np.bincount(y_train)} (factual, hallucinated)")

# %% Step 8: Dimensionality reduction with PCA
# Apply PCA to reduce feature dimensions
n_components = min(32, X_train.shape[1], X_train.shape[0])
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)

print(f"  Original features: {X_train.shape[1]}")
print(f"  After PCA: {X_train_pca.shape[1]}")
print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# Visualize PCA results
plt.figure(figsize=(12, 4))

# Plot 1: Explained variance
plt.subplot(1, 3, 1)
plt.plot(pca.explained_variance_ratio_, 'o-')
plt.title('PCA Explained Variance')
plt.xlabel('Component')
plt.ylabel('Variance Ratio')

# Plot 2: Cumulative variance
plt.subplot(1, 3, 2)
plt.plot(np.cumsum(pca.explained_variance_ratio_), 's-')
plt.title('Cumulative Explained Variance')
plt.xlabel('Component')
plt.ylabel('Cumulative Variance')

# Plot 3: Data in first 2 PCs
plt.subplot(1, 3, 3)
colors = ['blue' if label == 0 else 'red' for label in y_train]
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=colors, alpha=0.7)
plt.title('Data in First Two Components')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(['Factual', 'Hallucinated'])

plt.tight_layout()
plt.show()

# %% Step 9: Train classifier
# Train logistic regression
classifier = LogisticRegression(max_iter=2000, class_weight='balanced')
classifier.fit(X_train_pca, y_train)

# Make predictions on training data
y_train_pred = classifier.predict(X_train_pca)
y_train_proba = classifier.predict_proba(X_train_pca)[:, 1]

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_auroc = roc_auc_score(y_train, y_train_proba)

print(f"✓ Classifier trained successfully")
print(f"✓ Training accuracy: {train_accuracy:.3f}")
print(f"✓ Training AUROC: {train_auroc:.3f}")

# Show training predictions
print(f"\nTraining predictions:")
for i, (text, true_label, pred_label, prob) in enumerate(zip(train_texts, y_train, y_train_pred, y_train_proba)):
    true_name = "Factual" if true_label == 0 else "Hallucinated"
    pred_name = "Factual" if pred_label == 0 else "Hallucinated"
    correct = "✓" if true_label == pred_label else "✗"
    print(f"{i+1}. {correct} True: {true_name}, Pred: {pred_name} ({prob:.3f}) - {text[:40]}...")

# %% step 10: Test on new examples
# Create test examples
test_texts = [
    "Mount Everest is the tallest mountain on Earth.",  # Factual
    "Mount Everest is located in Australia.",           # Hallucinated
    "The Moon affects Earth's tides.",                  # Factual  
    "The Moon is made of green cheese.",                # Hallucinated
]
test_labels = [0, 1, 0, 1]  # True labels

print(f"Testing on {len(test_texts)} new examples:")
for i, text in enumerate(test_texts):
    true_name = "Factual" if test_labels[i] == 0 else "Hallucinated"
    print(f"{i+1}. [{true_name}] {text}")

# Extract features for test texts
X_test = extract_features_batch(test_texts, k=5)
X_test_pca = pca.transform(X_test)

# Make predictions
y_test_pred = classifier.predict(X_test_pca)
y_test_proba = classifier.predict_proba(X_test_pca)[:, 1]

# Calculate test metrics
test_accuracy = accuracy_score(test_labels, y_test_pred)
test_auroc = roc_auc_score(test_labels, y_test_proba)

print(f"\n✓ Test Results:")
print(f"  Accuracy: {test_accuracy:.3f}")
print(f"  AUROC: {test_auroc:.3f}")

# Show detailed predictions
print(f"\nDetailed predictions:")
for i, (text, true_label, pred_label, prob) in enumerate(zip(test_texts, test_labels, y_test_pred, y_test_proba)):
    true_name = "Factual" if true_label == 0 else "Hallucinated"
    pred_name = "Factual" if pred_label == 0 else "Hallucinated"
    correct = "✓" if true_label == pred_label else "✗"
    print(f"\n{i+1}. {correct} Text: '{text}'")
    print(f"    True: {true_name}")
    print(f"    Predicted: {pred_name}")
    print(f"    Confidence: {prob:.3f}")

# %% step 11: Visualize results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Probability distributions
factual_probs = y_test_proba[np.array(test_labels) == 0]
hallucinated_probs = y_test_proba[np.array(test_labels) == 1]

axes[0].hist(factual_probs, alpha=0.7, label='Factual (True)', color='blue', bins=10)
axes[0].hist(hallucinated_probs, alpha=0.7, label='Hallucinated (True)', color='red', bins=10)
axes[0].axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
axes[0].set_xlabel('Hallucination Probability')
axes[0].set_ylabel('Count')
axes[0].set_title('Prediction Probabilities by True Class')
axes[0].legend()

# Plot 2: Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, y_test_pred)

im = axes[1].imshow(cm, cmap='Blues')
axes[1].set_title('Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

# Add numbers to confusion matrix
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16)

axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Factual', 'Hallucinated'])
axes[1].set_yticklabels(['Factual', 'Hallucinated'])

# Plot 3: Individual predictions
colors = ['green' if p == t else 'red' for p, t in zip(y_test_pred, test_labels)]
bars = axes[2].bar(range(len(test_texts)), y_test_proba, color=colors, alpha=0.7)
axes[2].axhline(y=0.5, color='black', linestyle='--', label='Decision Threshold')
axes[2].set_xlabel('Test Example')
axes[2].set_ylabel('Hallucination Probability')
axes[2].set_title('Individual Predictions\n(Green=Correct, Red=Incorrect)')
axes[2].set_xticks(range(len(test_texts)))
axes[2].set_xticklabels([f'Text {i+1}' for i in range(len(test_texts))])
axes[2].legend()

plt.tight_layout()
plt.show()
# %%
