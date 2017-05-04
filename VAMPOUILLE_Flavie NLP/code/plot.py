# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 12:03:37 2017

@author: flavie vampouille
"""

import matplotlib.pyplot as plt

"""  ----------------  Part 4  ----------------  """

if False:
    # without dropout
    history = {'acc': [0.80321428571428577, 0.8898571428571429, 0.92478571428571432, 0.95378571428571424, 0.97232142857142856, 0.9816785714285714], 'loss': [0.42502404628481183, 0.27017690519775661, 0.19572004153898784, 0.12841196973302535, 0.081737400571948712, 0.054236977713049521], 'val_acc': [0.85328571428571431, 0.84271428578240526, 0.83557142850330901, 0.81500000006811957, 0.82728571428571429, 0.81771428564616611], 'val_loss': [0.34327532379967823, 0.35362174943515234, 0.39962052835736955, 0.55567008846146715, 0.51984316597666058, 0.6052579183919089]}
    # plot accuracy on train et val
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy without dropout')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
    # plot loss on train et val
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss without dropout')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

elif False: 
    # with drop out
    history = {'acc': [0.73457142857142854, 0.82978571428571424, 0.85689285714285712, 0.875, 0.88649999999999995, 0.89803571428571427], 'loss': [0.52761130758694241, 0.39453638630253929, 0.33534351376124788, 0.29855673459597998, 0.27513613145266258, 0.2490978940810476], 'val_acc': [0.83699999999999997, 0.83585714278902323, 0.83828571421759468, 0.83885714292526248, 0.84128571421759468, 0.83442857149669103], 'val_loss': [0.38137738330023629, 0.37548018860816956, 0.37203658066477097, 0.36317587733268736, 0.36665348812511989, 0.37392496211188181]}
    # plot accuracy on train et val
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy with dropout')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
    # plot loss on train et val
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss with dropout')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

"""  ----------------  Part 5  ----------------  """

if False: 
    # without lstm
    history = {'acc': [0.7455357142857143, 0.82928571428571429, 0.85317857142857145, 0.87071428571428566, 0.88092857142857139, 0.89596428571428577], 'loss': [0.4979014296872275, 0.3794727668932506, 0.33635984439509253, 0.30163884626116072, 0.28143946559940064, 0.25052732389313837], 'val_acc': [0.82842857149669102, 0.84128571435383392, 0.84142857136045179, 0.85299999999999998, 0.84528571421759469, 0.84414285707473757], 'val_loss': [0.39047389090061185, 0.3583864084822791, 0.36085803590502058, 0.34087671848705836, 0.35256222869668691, 0.35472284222500666]}
    # plot accuracy on train et val
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy without lstm')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
    # plot loss on train et val
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss without lstm')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

elif False:
    # with lstm, without dropout
    history = {'acc': [0.73289285714285712, 0.83525000000000005, 0.84699999999999998, 0.86132142857142857, 0.86975000000000002, 0.8773928571428572], 'loss': [0.50499082715170729, 0.37483343012843812, 0.34611563021796088, 0.32173307805401941, 0.30352757121835439, 0.28904914345060079], 'val_acc': [0.83085714278902323, 0.83828571421759468, 0.84814285721097671, 0.84828571421759469, 0.84957142850330902, 0.85071428564616614], 'val_loss': [0.36843021271909987, 0.35325492075511389, 0.34442612876210893, 0.33815159872600009, 0.34046861137662615, 0.33876776816163745]}
    # plot accuracy on train et val
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy with lstm, without dropout')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
    # plot loss on train et val
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss with lstm, witout dropout')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

elif False:
    # with lstm and dropout
    history = {'acc': [0.71357142857142852, 0.82878571428571424, 0.84535714285714281, 0.85646428571428568, 0.86517857142857146, 0.87278571428571428], 'loss': [0.52144347243649614, 0.38488497913735253, 0.35582582802431922, 0.33053374990395135, 0.31122146436146325, 0.29900694379636222], 'val_acc': [0.82785714278902323, 0.84399999993188035, 0.84885714292526249, 0.83585714292526248, 0.84728571421759469, 0.83542857149669103], 'val_loss': [0.37433849339825764, 0.34908302470615932, 0.34295265422548565, 0.37446922200066701, 0.35027265371595112, 0.35801163743223463]}
    # plot accuracy on train et val
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy with lstm and dropout')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
    # plot loss on train et val
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss with lstm and dropout')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

elif False:
    # with lstm, no dropout, activation='sigmoid'    
    history = {'acc': [0.70128571428571429, 0.81792857142857145, 0.84289285714285711, 0.86046428571428568, 0.86975000000000002, 0.87778571428571428], 'loss': [0.55292565120969495, 0.403642332639013, 0.35552308266503468, 0.3268731087275914, 0.30469801010404313, 0.28735244914463587], 'val_acc': [0.82300000006811957, 0.83914285707473757, 0.8530000000681196, 0.85428571421759469, 0.85428571421759469, 0.85257142850330903], 'val_loss': [0.40049462158339366, 0.35742453658580781, 0.33499855073860713, 0.33400129686083113, 0.33241537860461645, 0.33443332305976325]}
    # plot accuracy on train et val
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy with lstm, no dropout, activation = sigmoid')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
    # plot loss on train et val
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss with lstm, no dropout, activation = sigmoid')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

elif True:
    # with lstm, no dropout, activation='hard_sigmoid'
    history = {'acc': [0.70846428571428577, 0.8194285714285714, 0.84528571428571431, 0.85978571428571426, 0.87392857142857139, 0.87892857142857139], 'loss': [0.54310493934154513, 0.39781917037282671, 0.35145169925689695, 0.32553201766524997, 0.30412731600659232, 0.28766215349095209], 'val_acc': [0.82571428571428573, 0.8510000000681196, 0.85228571435383393, 0.8504285713604518, 0.8484285713604518, 0.85385714278902325], 'val_loss': [0.39551871732303073, 0.34549552822113039, 0.33860820330892288, 0.33986698593412129, 0.33942103333132606, 0.33564561886446814]}
    # plot accuracy on train et val
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy with lstm, no dropout, activation = hard_sigmoid')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
    # plot loss on train et val
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss with lstm, no dropout, activation = hard_sigmoid')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()