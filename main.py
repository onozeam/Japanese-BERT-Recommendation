""" Use alllenlp 1.0.0 or later """
from allennlp.common import FromParams
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import ArrayField, TextField, LabelField
from allennlp.models import Model
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder, TextFieldEmbedder
from allennlp.data.dataloader import DataLoader as AllennlpDataLoader
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.training import GradientDescentTrainer
import argparse
import numpy as np
from overrides import overrides
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Dict, List, Optional


class SampleDatasetReader(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]], token_indexers: Dict[str, TokenIndexer]):
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers

    @overrides
    def text_to_instance(self, current_text: str, candidate_text: str,
                         categories: List[int], label: Optional[int]=None) -> Instance:
        current_tokens = [x for x in self.tokenizer(current_text)]
        candidate_tokens = [x for x in self.tokenizer(candidate_text)]
        fields = {}
        fields['current_tokens'] = TextField(current_tokens, self.token_indexers)
        fields['candidate_tokens'] = TextField(candidate_tokens, self.token_indexers)
        fields['categories'] = ArrayField(np.asarray(categories))
        if label is not None:
            fields['label'] = LabelField(label, skip_indexing=True)
        return Instance(fields)

    @overrides
    def _read(self, file_path):
        df = pd.read_csv(file_path)
        # df = pd.read_csv(file_path).head(64)
        for _, row in df.iterrows():
            categories = [row.current_main_category_id, row.candidate_main_category_id,
                          row.current_sub_category_id, row.candidate_sub_category_id]
            instance = self.text_to_instance(row.current_text, row.candidate_text, categories, row.label)
            yield instance


class FullyConnected(nn.Module, FromParams):
    def __init__(self, input_dim: int, output_dim: int=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 1
        hidden1_dim, hidden2_dim, hidden3_dim = int(input_dim/4), int(input_dim/8), int(input_dim/16)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden2_dim, hidden3_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden3_dim, output_dim))

    def forward(self, inp: torch.Tensor):
        return self.fc(inp)

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim


class Classifier(Model):
    def __init__(self, word_embeddings: TextFieldEmbedder,
                 encoder: nn.Module,
                 vocab: Vocabulary):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, current_tokens: Dict[str, Dict[str, torch.Tensor]],
                candidate_tokens: Dict[str, Dict[str, torch.Tensor]],
                categories: torch.Tensor,
                label: Optional[torch.Tensor]=None) -> torch.Tensor:
        current_embed = self.word_embeddings(current_tokens)
        candidate_embed = self.word_embeddings(candidate_tokens)
        fc_inp = torch.cat((current_embed[:, 0], candidate_embed[:, 0], categories), 1)
        class_logits = self.sigmoid(self.encoder(fc_inp))
        output = {"class_logits": class_logits}
        if label is not None:
            output["loss"] = self.loss(class_logits, label.type(torch.FloatTensor))
        return output


def train(clargs: argparse.ArgumentParser):
    # todo: use sort by lenght, instead of shuffle.
    token_indexer = PretrainedTransformerIndexer('cl-tohoku/bert-base-japanese')
    tokenizer = token_indexer._allennlp_tokenizer.tokenize
    reader = SampleDatasetReader(tokenizer, {'tokens': token_indexer})
    train_ds = reader.read('sample.csv')
    test_ds = reader.read('sample.csv')
    vocab = Vocabulary()
    train_ds.index_with(vocab)
    test_ds.index_with(vocab)
    data_loader = AllennlpDataLoader(dataset=train_ds, batch_size=clargs.batch_size, shuffle=False)
    validation_data_loader = AllennlpDataLoader(dataset=test_ds, batch_size=clargs.batch_size, shuffle=False)
    bert_embedder = PretrainedTransformerEmbedder(model_name="cl-tohoku/bert-base-japanese")
    word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": bert_embedder})
    bert_embedding_dim = word_embeddings.get_output_dim()  # 768
    fc_inp_dim = bert_embedding_dim * 2 + args.categories_dim
    encoder = FullyConnected(fc_inp_dim)
    model = Classifier(word_embeddings, encoder, vocab)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = GradientDescentTrainer(model=model,
                                     optimizer=optimizer,
                                     data_loader=data_loader,
                                     patience=1,
                                     validation_data_loader=validation_data_loader,
                                     num_epochs=clargs.epochs,
                                     serialization_dir=clargs.save_dir,
                                     cuda_device=0 if clargs.use_gpu else -1)
    trainer.train()


def infer(clargs: argparse.ArgumentParser):
    token_indexer = PretrainedTransformerIndexer('cl-tohoku/bert-base-japanese')
    tokenizer = token_indexer._allennlp_tokenizer.tokenize
    reader = SampleDatasetReader(tokenizer, {'tokens': token_indexer})
    ds = reader.read('sample.csv')
    vocab = Vocabulary()
    ds.index_with(vocab)
    data_loader = AllennlpDataLoader(dataset=ds, batch_size=1, shuffle=False)
    bert_embedder = PretrainedTransformerEmbedder(model_name="cl-tohoku/bert-base-japanese")
    word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": bert_embedder})
    bert_embedding_dim = word_embeddings.get_output_dim()  # 768
    fc_inp_dim = bert_embedding_dim * 2 + args.categories_dim
    encoder = FullyConnected(fc_inp_dim)
    model = Classifier(word_embeddings, encoder, vocab)
    with open(f"saved/{clargs.model_name}", 'rb') as f:
        model.load_state_dict(torch.load(f))
    for sample in data_loader:
        out = model(sample['current_tokens'], sample['candidate_tokens'], sample['categories'])
        logit = out['class_logits'].reshape(-1).item()
        print('logit:', logit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Initialize training parameter.')
    parser.add_argument('-infer', action='store_true')
    parser.add_argument('-use_gpu', action='store_true')
    parser.add_argument('-epochs', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-learning_rate', type=float, default=0.0001)
    parser.add_argument('-save_dir', type=str, default='saved')
    parser.add_argument('-model_name', type=str)
    args = parser.parse_args()
    args.categories_dim = 4
    if args.infer:
        infer(args)
    else:
        train(args)
