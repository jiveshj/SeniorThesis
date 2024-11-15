from newViterbi.py import VITERBI_Lists

import torch
from transformers import AutoModelForCausalLM , AutoTokenizer
import scipy
import numpy as np
class LMHeadModel:   # To check if this code is still working correctly

    def __init__(self, model_name):
        # Initialize the model and the tokenizer.
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def get_predictions(self, sentence):
        # Encode the sentence using the tokenizer and return the model predictions.
        inputs = self.tokenizer.encode(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(inputs)
            predictions = outputs[0]
        return predictions
    
    def get_next_word_probabilities(self, sentence, top_k=500):

        # Get the model predictions for the sentence.
        predictions = self.get_predictions(sentence)
        #print(predictions)
        
    
        # Get the next token candidates.
        next_token_candidates_tensor = predictions[0, -1, :]

        # Get the top k next token candidates.
        length = len(next_token_candidates_tensor)
        topk_candidates_indexes = torch.topk(
            next_token_candidates_tensor, length).indices.tolist()
        
        
        #printing 1st token tensor
        # print(next_token_candidates_tensor[0])

        
        #printing 1st token tensor in sorted arr
        # next_token_candidates_sort = torch.sort(next_token_candidates_tensor)
        # print(next_token_candidates_sort[1])


        # Get the token probabilities for all candidates.
        all_candidates_probabilities = torch.nn.functional.softmax(
            next_token_candidates_tensor, dim=-1)
        
        # all_candidates_prob_sorted = torch.nn.functional.softmax(next_token_candidates_sort,dim = -1)
        
        # Filter the token probabilities for the top k candidates.
        topk_candidates_probabilities = \
            all_candidates_probabilities[topk_candidates_indexes].tolist()

        # Decode the top k candidates back to words.
        topk_candidates_tokens = \
            [self.tokenizer.decode([idx]).strip() for idx in topk_candidates_indexes]#topk_candidates_indexes]

        #Return the top k candidates and their probabilities.
        return list(zip(topk_candidates_tokens, topk_candidates_probabilities))
      
        # output=list(zip(next_token_candidates_tensor,all_candidates_probabilities))
        # return output



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