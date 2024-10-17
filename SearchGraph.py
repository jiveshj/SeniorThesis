class SearchGraph:
    def __init__(self,context,probability,parent = None,child = None):
        self.context = context
        self.probability = probability
        self.parent = parent
        self.child = []
        if child is not None:
           self.child.append(child)
    def build_Context(self):
        if self.parent is not None:
            parent_context = self.parent.context
            return self.parent.build_Context() + ' ' + self.context
        else:
            return self.context
    def create_child(self):
        if self.parent is not None:
           self.parent.child.append(self)


# basic implementation of the above class
sentence = SearchGraph('I enjoy walking in the',1)
context = []
# root = Node("I enjoy walking in the", prob = 1)\
token_list = []
prob_list = []
model = LMHeadModel("gpt2")
tokens_50K = model.get_next_word_probabilities(sentence.context,3)
parents = []
for i in range(3):
     context = tokens_50K[i][0]
     prob = tokens_50K[i][1]
     context = SearchGraph(context,prob,sentence)
     context.create_child()
     parents.append(context)
for i in range(2,4):
   k = 0
   count = 0
   for j in range(3**i):
     if (count <4):
          tokens_2 = model.get_next_word_probabilities(parents[k].build_Context(),3)
     else:
          count = 0
          k +=1
          tokens_2 = model.get_next_word_probabilities(parents[k].build_Context(),3)
     context = tokens_2[count][0]
     prob = tokens_2[count][1]
     context = SearchGraph(context,prob,parents[j])
     context.create_child()
     parents.append(context)
     count +=1
   parents = parents[3**(i-1):]