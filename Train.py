import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import DataProcess as D
import Network

model = Network.LSTMChar()
model.cuda()
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


for epoch in range(10) :
    running_loss = 0.0
    print ("epoch : (%d)" % (epoch))
    for i, sentence in enumerate(D.training_data) :
        # clear gradients for each instance
        model.zero_grad()

        # init model's hidden
        model.EMB_hidden = model.init_hidden('emb')
        model.POS_hidden = model.init_hidden('pos')

        # make our target ready
        sentence_in = D.make_index_list(sentence, 'word')
        targets = D.make_index_list(sentence, 'tag')
        char_in = D.make_index_list(sentence, 'char')

        # feed the model
        tag_scores = model(sentence_in, char_in)

        # Compute the Loss, Gradients, and update the parameters
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

        # Print the result
        running_loss += loss.data[0]
        if i % 200 == 0 :
            print('[%d] epoch, [%5d] Steps - loss : %.3f' % (epoch, i, running_loss/200))
            running_loss = 0


print('Done Training !')

# Writing to File
print('Saving the Model...')
torch.save(model, './saveEntireLSTMCHAR')