from newViterbi import VITERBI_Lists
from collections import Counter
from collections import defaultdict

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
    def __init__(self,context,probability,parent = None,child = None,parent_index = None):
        self.context = context
        self.probability = probability
        self.parent = parent
        self.child = []
        self.parent_index = parent_index  # newly created.
        if child is not None:
           self.child.append(child)
        
        # Cache cumulative probability at node creation
        if parent:
            self.cached_prob = parent.calcProbTillNow() * probability
        else:
            self.cached_prob = probability

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
            if context_list[i] in ['.',':',',','?','!',';'] or ("'" in context_list[i]):
                if (i-1>= 0):
                    if context_list[i-1] not in  ['.',':',',','?','!',';'] and ("'" not in context_list[i-1]):#if two consecutive contexts are , ' etc.
                        word = context_list[i-1]+context_list[i]

                        formatted_contextList.remove(context_list[i-1])
                        formatted_contextList.append(word)
                    else:
                        formatted_contextList.append(context_list[i])
            else:

                 formatted_contextList.append(context_list[i])
        return ' '.join(formatted_contextList)

    def create_child(self):
        if self.parent is not None:
           self.parent.child.append(self)

    def replace_parent(self, new_parent):
        """Assign a new parent and update cached probability."""
        self.parent = new_parent
        self.cached_prob = new_parent.calcProbTillNow() * self.probability
    

    def calcProbTillNow(self):
        """Return cached cumulative probability to avoid redundant calculations."""
        return self.cached_prob

    # def calcProbTillNow(self):
    #   prob = self.probability
    #   node = self
    #   while node.parent is not None:
    #     prob = prob*node.parent.probability
    #     node = node.parent
    #   return prob    #can make this negative log probability.

    def assign_parent_index(self,parent_index):
      self.parent_index = parent_index





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
  prob_list = []
  num_tokens = numTokens
  content = []
  probability = []
  model = LMHeadModel("gpt2")
  tokens_50K = model.get_batch_predictions([sentence.context],numTokens+3)
  children = []
  overlap = []
  most_common = []
  #unique_elements = []   # to store unique elements at each iteration
  unique_tokens = set()
  probabilityMatrix = []
  uniqueTokensList = []
  new_content = []
  uniqueTokenLength = []

  flops_counter = {}
  cached_probs = {}
  batch_size = 75
  holdout_number = 5
  for i in range(num_tokens):
    context = tokens_50K[0][i][0]
    unique_tokens.add(context)
    new_content.append(context)
    prob = tokens_50K[0][i][1]
    probability.append(prob)
    context = SearchTree(context,prob,sentence,parent_index = 0)
    context.create_child()
    uniqueTokensList.append(context)
    children.append(context)

  content.append(new_content)
  previousUniqueLength = num_tokens
  #unique_elements.append(unique_tokens)
  initialStateProbability = probability
  uniqueTokenLength.append(num_tokens)
  for i in range(2,loop_runner):
    unique_tokens = set()
    probability = []
    new_content = []
    total_predictions = []
    previousSetLength = 0
    batch_sentences = [child.build_Context() for child in uniqueTokensList]
    if len(batch_sentences)>holdout_number:
        batch_sentences2 = batch_sentences[0:-holdout_number]
        batch_predictions = model.get_batch_predictions(batch_sentences2,numTokens+2)
        total_predictions = []
        total_predictions.extend(batch_predictions)
        batch_predictions1 = model.get_batch_predictions(batch_sentences[-holdout_number:],numTokens+2)
        total_predictions.extend(batch_predictions1)
    else:
        total_predictions = model.get_batch_predictions(batch_sentences,numTokens+2)
    # batch_sentences =  ["I enjoy walking in the park, but I'm not sure", "I enjoy walking in the park, but I'm not a", "I enjoy walking in the park, but I'm not going", "I enjoy walking in the park, but I'm also very", "I enjoy walking in the park, but I'm also not", "I enjoy walking in the park, but I'm afraid I", "I enjoy walking in the park, but I'm afraid that", "I enjoy walking in the park, but I'm afraid to", "I enjoy walking in the park, but I don't like", "I enjoy walking in the park, but I don't want", "I enjoy walking in the park, but I don't think", "I enjoy walking in the park, but I don' t", 'I enjoy walking in the park, but I don � c', 'I enjoy walking in the park, but I don � ve', 'I enjoy walking in the park, but I also enjoy the', 'I enjoy walking in the park, but I also enjoy being', 'I enjoy walking in the park, but I also enjoy walking', "I enjoy walking in the park, but it's a little", "I enjoy walking in the park, but it's a bit", "I enjoy walking in the park, but it's a lot", "I enjoy walking in the park, but it's hard for", 'I enjoy walking in the park, but it is very difficult', 'I enjoy walking in the park, but it is very hard', 'I enjoy walking in the park, but it is very quiet', 'I enjoy walking in the park, but it can get really', 'I enjoy walking in the park, but it can get pretty', "I enjoy walking in the park, but when I'm in", "I enjoy walking in the park, but when I'm out", 'I enjoy walking in the park, but when I walk down', 'I enjoy walking in the park, but when I walk into', 'I enjoy walking in the park, but when I go back', "I enjoy walking in the park, but when you're on", "I enjoy walking in the park, but when it's raining", "I enjoy walking in the park, but when it's time", "I enjoy walking in the park, but when it's dark", 'I enjoy walking in the park, but when it rains,', 'I enjoy walking in the park, but when it rains it', 'I enjoy walking in the park, and the people there are', 'I enjoy walking in the park, and the people there have', 'I enjoy walking in the park, and the people there seem', 'I enjoy walking in the park, and the people who live', 'I enjoy walking in the park, and the people who come', 'I enjoy walking in the park, and the people are always', 'I enjoy walking in the park, and the people are so', 'I enjoy walking in the park, and the people are nice', 'I enjoy walking in the park, and the view is amazing', 'I enjoy walking in the park, and the view is beautiful', 'I enjoy walking in the park, and the view is great', 'I enjoy walking in the park, and the view from my', 'I enjoy walking in the park, and the view from here', 'I enjoy walking in the park, and the view of this', 'I enjoy walking in the park, and the view of Lake', 'I enjoy walking in the park, and the smell and smell', 'I enjoy walking in the park, and the smell and taste', 'I enjoy walking in the park, I like to play with', 'I enjoy walking in the park, I like to watch movies', 'I enjoy walking in the park, I like the view.', 'I enjoy walking in the park, I like the view of', 'I enjoy walking in the park, I like the smell and', 'I enjoy walking in the park, I like the way they', 'I enjoy walking in the park, I like being in front', 'I enjoy walking in the park, I like being around people', 'I enjoy walking in the park, I like being around other', 'I enjoy walking in the park, I enjoy playing with friends', 'I enjoy walking in the park, I enjoy playing the game', 'I enjoy walking in the park, I enjoy playing the guitar', 'I enjoy walking in the park, I enjoy playing the piano', 'I enjoy walking in the park and seeing people. I love', "I enjoy walking in the park and seeing people. I'm", "I enjoy walking in the park and seeing people. It's", 'I enjoy walking in the park and seeing people. It is', 'I enjoy walking in the park and seeing people. It makes', "I enjoy walking in the park and seeing people. We're", 'I enjoy walking in the park and seeing people, but when', 'I enjoy walking in the park and seeing people and seeing what', 'I enjoy walking in the park and seeing people and animals ,"', 'I enjoy walking in the park and seeing all of these different', 'I enjoy walking in the park and seeing all of our neighbors', 'I enjoy walking in the park and seeing all of our favorite', 'I enjoy walking in the park and seeing all these beautiful trees', 'I enjoy walking in the park and seeing all these beautiful birds', 'I enjoy walking in the park and seeing all these different things', 'I enjoy walking in the park and seeing all these different species', 'I enjoy walking in the park and seeing all these different kinds', "I enjoy walking in the park and it's a great place", "I enjoy walking in the park and it's a great way", "I enjoy walking in the park and it's a great experience", "I enjoy walking in the park and it's a nice feeling", "I enjoy walking in the park and it's nice to see", "I enjoy walking in the park and it's nice to be", "I enjoy walking in the park and it's nice. It", "I enjoy walking in the park and it's nice. The", "I enjoy walking in the park and it's nice that we", "I enjoy walking in the park and it's nice that you", "I enjoy walking in the park and it's great for me", 'I enjoy walking in the park and it makes me want more', 'I enjoy walking in the park and it makes my day ."', 'I enjoy walking in the park and it makes my life easier', 'I enjoy walking in the park and it makes my body feel', 'I enjoy walking in the park and it makes my body look', 'I enjoy walking in the park and it makes my body stronger', 'I enjoy walking in the park. The park itself has been', 'I enjoy walking in the park. The park itself has some', 'I enjoy walking in the park. The park itself, though', 'I enjoy walking in the park. The trees are so tall', 'I enjoy walking in the streets of the United Arab Emir ate', 'I enjoy walking in the streets of the United Arab Emir ates', 'I enjoy walking in the streets of the United Arab Emir ati', 'I enjoy walking in the streets of London, but it was', 'I enjoy walking in the streets of London and seeing how many']

    # batch_predictions = model.get_batch_predictions(batch_sentences,numTokens+2)
    for j in range(len(uniqueTokensList)):


      for s in range(num_tokens):
        context = total_predictions[j][s][0]
        prob = total_predictions[j][s][1]

        if (i == loop_runner-1):
           print(context, end = " ")
           print(prob)


        unique_tokens.add(context)

        context = SearchTree(context,prob,uniqueTokensList[j])   #probably redundant: Because I should only create SearchTree of unique tokens
        # context.create_child() Removed this 2/19/2025
        if (len(unique_tokens)>previousSetLength):
          previousSetLength = len(unique_tokens)
          uniqueTokensList.append(context)
          new_content.append(context.context)


    #unique_elements.append(unique_tokens) # append the unique tokens list at each iteration to unique_elements list
    content.append(new_content) # for storing tokens which will pass to the decode_path function.

    for token in uniqueTokensList[previousUniqueLength:]:
      probs = []  #for making probability matrix and applying viterbi.
      probs2 = [] #for finding the new parent.
      for prevToken in uniqueTokensList[:previousUniqueLength]:
        probabilityCalc = findProbability(prevToken,token,model)


        probs.append(probabilityCalc)
        probs2.append(probabilityCalc*prevToken.calcProbTillNow())
        #new code below inserted:
      if not probs2:
        continue
      else:
        max_value = max(probs2)
        max_index = probs2.index(max_value)
        token.replace_parent(uniqueTokensList[:previousUniqueLength][max_index])
        token.assign_parent_index(max_index)

      probability.append(probs)
    probabilityMatrix.append(probability)
    flops_counter[i-1] = model.get_batch_prediction_count()
    model.reset_batch_prediction_count()


    uniqueTokenLength.append(len(uniqueTokensList[previousUniqueLength:]))

    previousUniqueLength = len(uniqueTokensList[previousUniqueLength:])
    uniqueTokensList = uniqueTokensList[len(uniqueTokensList)-previousUniqueLength:]


  return probabilityMatrix, initialStateProbability, content,uniqueTokenLength, flops_counter

def ViterbiTransformerPipeline(rootSentence, numTokens = 3, loop_runner=3):
    probabilityMatrix,initialStateProbability,content, uniqueTokenLength, flops_counter = generateIntermediates(rootSentence,numTokens,loop_runner+1)
    best_path,viterbi_mat = VITERBI_Lists(probabilityMatrix, initialStateProbability)
    print('content: ',content)
    print('best_path: ',best_path)
    decodedString = decodePath(best_path,content,rootSentence)
    return decodedString
