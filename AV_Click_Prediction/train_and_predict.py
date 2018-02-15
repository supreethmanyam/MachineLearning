from datetime import datetime
from FTRL_proximal import ftrl_proximal
from FTRL_proximal import data
from FTRL_proximal import epoch
from FTRL_proximal import alpha, beta, L1, L2
from FTRL_proximal import alpha_decay, decay_after, decay_rate
from FTRL_proximal import D, interaction
from FTRL_proximal import train, test, test_status


##############################################################################
# start training #############################################################
##############################################################################
print('############################### Training #############################\n')
start = datetime.now()
learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)
for e in range(epoch):
    if alpha_decay:
        if decay_after == 1:
            learner.alpha = alpha/decay_rate
        else:
            if (e+1 > decay_after) and ((e+1) % decay_after == 1):
                learner.alpha = alpha/decay_rate
    for t, date, ID, x, y, string_to_attach in data(train, D):  # data is a generator
        if x:
            p = learner.predict(x)
            learner.update(x, p, y)
        if (t % 1000000 == 0):
            print('Epoch %d finished, index %d, elapsed time: %s' % (
                e+1, t, str(datetime.now() - start)))


##############################################################################
# start testing #############################################################
##############################################################################
print('############################### Predicting #############################\n')
header_string = ',UserIp,publisherId,advertiserCampaignId,ConversionPayOut'
start = datetime.now()
with open(test_status, 'w') as outfile:
    outfile.write('ID,probability,ConversionStatus' + header_string + '\n')
    for t, date, ID, x, y, string_to_attach in data(test, D):
        if x:
            p = learner.predict(x)
            outfile.write('%s,%s,%s,%s\n' % (ID, str(p), y, string_to_attach))
        if (t % 1000000 == 0):
            print('Index %d elapsed time: %s' % (t, str(datetime.now() - start)))
