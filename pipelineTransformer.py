from newViterbi.py import VITERBI_Lists

import torch
from transformers import AutoModelForCausalLM , AutoTokenizer
import scipy
import numpy as np

class LMHeadModel:
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        # Initialize the model and tokenizer
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Use EOS token as padding
            self.tokenizer.padding_side = "right"

        self.batch_prediction_count = 0


    def batch_encode(self, sentences):
        """
        Encodes a batch of sentences into input tensors.
        Args:
            sentences (list of str): The input sentences to encode.
        Returns:
            inputs (dict): A dictionary of tokenized inputs ready for the model.
        """
        return self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,  # Pad to the longest sequence in the batch
            truncation=True,  # Truncate sequences longer than the model's max length
        ).to(self.device)

    def batch_decode(self, token_ids):
        """
        Decodes a batch of token IDs back to sentences.
        Args:
            token_ids (torch.Tensor): A tensor of token IDs to decode.
        Returns:
            decoded_sentences (list of str): The decoded sentences.
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    def batch_decode_top_k(self, token_ids_batch, tokenizer):
        """
        Decodes token IDs to meaningful text while merging subword tokens.
        Args:
            token_ids_batch (torch.Tensor): A batch of token IDs (e.g., from `topk`).
            tokenizer: The tokenizer used for encoding/decoding.
        Returns:
            list of list of str: Decoded tokens (words/subwords) for each sequence in the batch.
        """
        decoded_tokens = []
        for token_ids in token_ids_batch:
            # Decode each token ID in the batch, joining subwords correctly
            tokens = [tokenizer.decode([token_id]).strip() for token_id in token_ids]
            decoded_tokens.append(tokens)
        return decoded_tokens

    def get_batch_predictions(self, sentences, top_k=100):
        """
        Predicts the next tokens for a batch of input sentences.
        Args:
            sentences (list of str): The input sentences.
            top_k (int): Number of top tokens to return for each sentence.
        Returns:
            predictions (list of list of tuples): Top-k token predictions for each sentence.
        """
        #Increment to see how many times this function is called after a given layer of trellis.
        self.batch_prediction_count += 1


        # Tokenize inputs
        inputs = self.batch_encode(sentences)

        # Pass through the model
        with torch.no_grad():
            outputs = self.model(**inputs,use_cache = False)

        # Get logits for the last token in each sequence
        logits = outputs.logits[:, -1, :]  # Shape: (batch_size, vocab_size)


        # Compute probabilities using softmax
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_probs, top_token_ids = torch.topk(probs, k=top_k, dim=-1)
        top_tokens = self.batch_decode_top_k(top_token_ids, self.tokenizer)


        predictions = [
            [(token, prob.item()) for token, prob in zip(top_tokens[i], top_probs[i]) if token and token != "\n"]
            for i in range(len(sentences))
        ]
        return predictions

    def get_batch_prediction_count(self):
        """
        Returns the number of times batch predictions have been made.
        """
        return self.batch_prediction_count

    def reset_batch_prediction_count(self):
        """ Resets the count
        """

        self.batch_prediction_count = 0




def TransformerPipeline(rootSentence,loop_runner = 3):
  model = LMHeadModel("gpt2")
  finalSentence = rootSentence
  for i in range(loop_runner):
    tokens_50K = model.get_next_word_probabilities(finalSentence)

    context = tokens_50K[0][0]
    if context in ['.',':',',','?','!',';']:
      finalSentence += context
    else:
      finalSentence = finalSentence + ' ' + context
  return finalSentence
