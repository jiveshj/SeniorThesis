class SearchGraph:
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


# basic implementation of the above class (OLD IMPLEMENTATION)
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
