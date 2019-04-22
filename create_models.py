from RNNSteam import make_rnn

batmanarkhamknight_accuracy = make_rnn('batmanarkhamknightreviews.txt', 'batmanarkhamknightscores.txt', .8, 'batmanarkhamknight')
batmanarkhamknight_split_accuracy = make_rnn('batmanarkhamnightsplitreviews.txt', 'batmanarkhamnightsplitscores.txt', .8, 'batmanarkhamknightsplit')

dota2_accuracy = make_rnn('dota2reviews.txt', 'dota2scores.txt', .8, 'dota2')
dota2_split_accuracy = make_rnn('dota2splitreviews.txt', 'dota2splitscores.txt', .8, 'dota2split')

gtaV_accuracy = make_rnn('gtaVreviews.txt', 'gtaVscores.txt', .8, 'gtaV')
gtaV_split_accuracy = make_rnn('gtaVsplitreviews.txt', 'gtaVsplitscores.txt', .8, 'gtaVsplit')

nomanssky_accuracy = make_rnn('nomansskyreviews.txt', 'nomansskyscores.txt', .8, 'nomanssky')
nomanssky_split_accuracy = make_rnn('nomansskysplitreviews.txt', 'nomansskysplitscores.txt', .8, 'nomansskysplit')

payday2_accuracy = make_rnn('payday2reviews.txt', 'payday2scores.txt', .8, 'payday2')
payday2_split_accuracy = make_rnn('payday2splitreviews.txt', 'payday2splitscores.txt', .8, 'payday2split')

total_accuracy = make_rnn('totalreviews.txt', 'totalscores.txt', .8, 'totalfull')
total_split_accuracy = make_rnn('totalsplitreviews.txt', 'totalsplitscores.txt', .8, 'totalsplit')

print("Batman accuracy:")
print(batmanarkhamknight_accuracy)
print(batmanarkhamknight_split_accuracy)
print("Dota 2 accuracy:")
print(dota2_accuracy)
print(dota2_split_accuracy)
print("GTA V accuracy:")
print(gtaV_accuracy)
print(gtaV_split_accuracy)
print("No Man's Sky accuracy:")
print(nomanssky_accuracy)
print(nomanssky_split_accuracy)
print("Payday 2 accuracy:")
print(payday2_accuracy)
print(payday2_split_accuracy)
print("Total accuracy:")
print(total_accuracy)
print(total_split_accuracy)