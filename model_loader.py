import torch
from transformers import BertConfig, BertTokenizerFast, BertForSequenceClassification

label2id = {'negatif': 0, 'netral': 1, 'positif': 2}
id2label = {v: k for k, v in label2id.items()}
aspects = ["Fasilitas", "Harga", "Pelayanan"]

def load_model(aspect):
    data = torch.load(
        f"./pipeline/pipeline_absa_indobert_{aspect}.pkl",
        map_location=torch.device("cpu"),
        weights_only=False
    )
    
    config = BertConfig.from_pretrained("indobenchmark/indobert-base-p1", num_labels=3)
    model = BertForSequenceClassification(config)
    model.load_state_dict(data['model'])
    model.eval()
    
    tokenizer = data['tokenizer']
    
    return model, tokenizer
