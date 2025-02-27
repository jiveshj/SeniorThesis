from newViterbi import VITERBI_Lists
from collections import Counter
from collections import defaultdict
import itertools

import torch
from transformers import AutoModelForCausalLM , AutoTokenizer
import scipy
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer




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
      if unique_tokens_list[i][best_path[i]] in ['.',':',',','?','!',';']:
            if (i-1>= 0):
                resultant_string = resultant_string+unique_tokens_list[i][best_path[i]]
      elif "'" in unique_tokens_list[i][best_path[i]]:
              resultant_string = resultant_string + unique_tokens_list[i][best_path[i]]
      else:
            resultant_string = resultant_string + ' '+ unique_tokens_list[i][best_path[i]]
    return root_string+resultant_string

# Now, have the probability matrix ready in which one list contains the probability to reach that state from previous list of tokens
#To make the probs ready find the unique tokens and then number them/store in a list and then find top 3 tokens given those tokens, find unique
#tokens and then extract probs of getting those from the previous list. [state transition probmat] and then run viterbi!!

def findProbability(InitialToken, FinalTokens, model):
    context = InitialToken.build_Context()
    tokens_50K = model.get_batch_predictions([context], 500)

    token_dict = {}  # Dictionary to store only the first occurrence of each token

    for token, prob in tokens_50K[0]:
        if token not in token_dict or prob>token_dict[token]:  # Store only the first occurrence
            token_dict[token] = prob

    return [token_dict.get(FinalToken.context, 0) for FinalToken in FinalTokens]  # Return probability if found, else 0

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
  holdout_number = 15
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
        unique_tokens.add(context)

        context = SearchTree(context,prob,uniqueTokensList[j])   #probably redundant: Because I should only create SearchTree of unique tokens
        # context.create_child() Removed this 2/19/2025
        if (len(unique_tokens)>previousSetLength):
          previousSetLength = len(unique_tokens)
          uniqueTokensList.append(context)
          new_content.append(context.context)


    #unique_elements.append(unique_tokens) # append the unique tokens list at each iteration to unique_elements list
    content.append(new_content) # for storing tokens which will pass to the decode_path function.

    # for token in uniqueTokensList[previousUniqueLength:]:
    #   probs = []  #for making probability matrix and applying viterbi.
    #   probs2 = [] #for finding the new parent.
    #   for prevToken in uniqueTokensList[:previousUniqueLength]:
      
    #     probabilityCalc = findProbability(prevToken,token,model)

      
    #     probs.append(probabilityCalc)
    #     probs2.append(probabilityCalc*prevToken.calcProbTillNow())
    #     #new code below inserted:
    #   if not probs2:
    #     continue
    #   else:
    #     max_value = max(probs2)
    #     max_index = probs2.index(max_value)
    #     token.replace_parent(uniqueTokensList[:previousUniqueLength][max_index])
    #     # token.create_child() # not sure about this think about it more. 
    #     token.assign_parent_index(max_index)

    #   probability.append(probs)
    # probabilityMatrix.append(probability)
    comb_prob = []
    for prevToken in uniqueTokensList[:previousUniqueLength]:
      comb_prob.append(findProbability(prevToken,uniqueTokensList[previousUniqueLength:], model))
    comb_prob = list(itertools.chain(*comb_prob)) # flattening the list


    for tokenumber,newToken in enumerate(uniqueTokensList[previousUniqueLength:]):
      probs = [comb_prob[a*len(uniqueTokensList[previousUniqueLength:]) + tokenumber] for a in range(len(uniqueTokensList[:previousUniqueLength]))]
      probs2 = [probs[i]*uniqueTokensList[:previousUniqueLength][i].calcProbTillNow() for i in range(len(probs))]
      if not probs2:
        continue
      else:
        max_value = max(probs2)
        max_index = probs2.index(max_value)
        newToken.replace_parent(uniqueTokensList[:previousUniqueLength][max_index])
        newToken.assign_parent_index(max_index)
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
    best_path,viterbi_mat,best_path_prob = VITERBI_Lists(probabilityMatrix, initialStateProbability)

    decodedString = decodePath(best_path,content,rootSentence)
    return decodedString,best_path_prob
