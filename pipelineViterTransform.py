from newViterbi import VITERBI_Lists
from collections import Counter

import torch
from transformers import AutoModelForCausalLM , AutoTokenizer
import scipy
import numpy as np
class LMHeadModel:   # Check if this works 
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



class SearchTree:
    def __init__(self,context,probability,parent = None,child = None):
        self.context = context
        self.probability = probability
        self.parent = parent
        self.child = []
        if child is not None:
           self.child.append(child)
    def build_Context(self):
      
        context_list = []
        node = self
        while node.parent is not None:
           
            context_list.append(node.context)   
            node = node.parent
        context_list.append(node.context)
        context_list.reverse()
        formatted_contextList = []
        for i in range(len(context_list)):
            if context_list[i] in ['.',':',',','?','!',';']:
                if (i-1>= 0):
                    formatted_contextList[i-1] += context_list[i]
            else:
                formatted_contextList.append(context_list[i])
        return ' '.join(formatted_contextList)
    
    def create_child(self):
        if self.parent is not None:
           self.parent.child.append(self)
def decodePath(best_path,unique_tokens_list,root_string):
    resultant_string = ''
    for i in range(len(best_path)):
        resultant_string = resultant_string + ' '+ unique_tokens_list[i][best_path[i]]
    return root_string+resultant_string

# Now, have the probability matrix ready in which one list contains the probability to reach that state from previous list of tokens
#To make the probs ready find the unique tokens and then number them/store in a list and then find top 3 tokens given those tokens, find unique
#tokens and then extract probs of getting those from the previous list. [state transition probmat] and then run viterbi!!

def findProbability(InitialToken,FinalToken,model):
    context = InitialToken.build_Context()
    tokens_50K = model.get_next_word_probabilities(context)
    for token,prob in tokens_50K:
        if token == FinalToken.context:
            return prob
def most_frequent(List):
    return max(set(List), key=List.count)

def sort_with_indices(arr):
  """Sorts a list and returns the original indices of the sorted elements."""
  indices = list(range(len(arr)))
  indices.sort(key=lambda i: arr[i], reverse = True)
  return indices

def find_overlap_children(arr):
  "Find all the children of top two tokens and return their overlap"
  children1 = {child.context for child in arr[0].child}
  children2 = {child.context for child in arr[1].child}
  common = len(children1.intersection(children2))
  return common / len(arr[0].child) if arr[0].child else 0

def generateIntermediates(root,numTokens = 3, loop_runner = 4):
  sentence = SearchTree(root,1)
  context = []
  # root = Node("I enjoy walking in the", prob = 1)\
  prob_list = []
  num_tokens = numTokens
  content = []
  probability = []
  model = LMHeadModel("gpt2")
  tokens_50K = model.get_next_word_probabilities(sentence.context,num_tokens)
  children = []
  overlap = []
  most_common = []
  #unique_elements = []   # to store unique elements at each iteration
  unique_tokens = set()
  probabilityMatrix = []
  uniqueTokensList = []
  new_content = []
  for i in range(num_tokens):
    context = tokens_50K[i][0]
    unique_tokens.add(context)
    new_content.append(context)
    prob = tokens_50K[i][1]
    probability.append(prob)
    context = SearchTree(context,prob,sentence)
    context.create_child()
    uniqueTokensList.append(context)
    children.append(context)

  content.append(new_content)
  previousUniqueLength = num_tokens
  #unique_elements.append(unique_tokens)
  initialStateProbability = probability

  for i in range(2,loop_runner):
    unique_tokens = set()
    probability = []
    new_content = []
    previousSetLength = 0
    for j in range(len(children)):
      token_list = model.get_next_word_probabilities(children[j].build_Context(),num_tokens)
      for s in range(num_tokens):
        context = token_list[s][0]
        prob = token_list[s][1]
        unique_tokens.add(context)
      
        #probability.append(prob)
        context = SearchTree(context,prob,children[j])
        context.create_child()
        if (len(unique_tokens)>previousSetLength):
          previousSetLength = len(unique_tokens)
          uniqueTokensList.append(context)
          new_content.append(context.context)

        children.append(context)    # may be don't need to store everything rather I can just store unique elements in here
    
    #unique_elements.append(unique_tokens) # append the unique tokens list at each iteration to unique_elements list
    content.append(new_content) # for storing tokens which will pass to the decode_path function. 
    for token in uniqueTokensList[previousUniqueLength:]:
      probs = []
      for prevToken in uniqueTokensList[:previousUniqueLength]:
        probabilityCalc = findProbability(prevToken,token,model)
        probs.append(probabilityCalc)
      probability.append(probs)
    probabilityMatrix.append(probability)
    
    previousUniqueLength = len(uniqueTokensList[previousUniqueLength:])
    uniqueTokensList = uniqueTokensList[len(uniqueTokensList)-previousUniqueLength:]

    #    for parents in children[num_tokens**(i-1)]
    #    content = content[num_tokens**(i-1):] 

    #    for probs in range(len(unique_elements)):
    #      for prev_token in children[prev_length:len(unique_elements)]:
    #         probability.append(findProbability(prev_token,content,model))
    
      

    children = children[num_tokens**(i-1):]
          
    #content = content[num_tokens**(i-1):]
    #probability = probability[num_tokens**(i-1):]
    # count = Counter(content)
    # most_common.append(count.most_common(1)[0][1])
  return probabilityMatrix, initialStateProbability, content

def ViterbiTransformerPipeline(rootSentence, numTokens = 3, loop_runner=3):
    probabilityMatrix,initialStateProbability,content = generateIntermediates(rootSentence,numTokens,loop_runner+1)
    best_path,viterbi_mat = VITERBI_Lists(probabilityMatrix, initialStateProbability)
    print('content: ',content)
    print('best_path: ',best_path)
    decodedString = decodePath(best_path,content,rootSentence)
    return decodedString