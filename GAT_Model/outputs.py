# outputs.py

def get_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        for conv in model.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        # x_dict now contains embeddings for each node type
    return x_dict

def get_attention_weights(model):
    # Assuming we're interested in attention weights from the last layer
    last_conv = model.convs[-1]
    attention_weights = {}
    for edge_type, conv in last_conv.convs.items():
        # Get attention weights
        if hasattr(conv, 'alpha'):
            attention_weights[edge_type] = conv.alpha
    return attention_weights
