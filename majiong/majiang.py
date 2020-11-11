# coding=utf-8
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
import random
from collections import Counter
from os.path import isfile, join, getsize
'''this function is created to achieve a majiong game'''
def card_store_create():
    card_store = []
    '''card announce'''
    number_card = ['p', 's', 'w']  # 饼、条、万
    word_card = ['d', 'n', 'x', 'b', 'z', 'c', 'f']  # 东南西北中百发
    r = random.random
    # random.seed(7914)
    print(r())
    for i in range(3):
        for j in range(9):
            for k in range(4):
                card_store.append(number_card[i] + str(j + 1) + str(k + 1))  #create 3 classes number_cards
    for i in range(7):
        for j in range(4):
            card_store.append(word_card[i] + str(j + 1))      #create 7 classed word_cards
    print(card_store)
    random.shuffle(card_store, random=r)        #random shuffle card_store
    print(card_store)
    return card_store, number_card, word_card
#0.9415091555868781

def card_coding(card_list):#a function to transform str card into int card,use single digits to distinguish number_card and word_card
    coded_code = 0
    if card_list[0] in number_card:
        coded_code = (number_card.index(card_list[0])+1)*100+int(card_list[1])*10+int(card_list[2])
    elif card_list[0] in word_card:
        coded_code = (word_card.index(card_list[0])+4)*100+int(card_list[1])*10
    return coded_code

def card_decoding(card_code):#a function to transform int card into str card,use single digits to distinguish number_card and word_card
    card_list = []
    if card_code % 10 == 0:
        card_list = word_card[card_code//100-4]+str((card_code//10)%10)
    elif card_code % 10 !=0:
        card_list = number_card[card_code//100-1]+str(card_code%100)
    return card_list

def arrange_card(input_card_list):#cards coding,arrange cards, cards decoding,output
    card_code_list = []
    card_list = []
    print(input_card_list)
    for i in range(len(input_card_list)):
        card_code_list.append(card_coding(input_card_list[i]))
    # print(card_code_list)
    card_code_list.sort()
    # print(card_code_list)
    for i in range(len(input_card_list)):
        card_list.append(card_decoding(card_code_list[i]))
    # print(card_list)
    return card_list

def draw_card(remaining_card_store, i=1):#draw cards from remaining card store
    hand_card = []
    while i >=1:
        single_card = random.choice(remaining_card_store)
        remaining_card_store.remove(single_card)
        i -= 1
        hand_card.append(single_card)
    print(hand_card)
    return hand_card, remaining_card_store

def kezi(hand_card_list):
    # check_list = []
    kezi_list = []
    class_number_list = []
    card_number_list = []
    card_word_list = []
    hand_card_list = arrange_card(hand_card_list)
    for i in range(len(hand_card_list)):
        if card_coding(hand_card_list[i])//100 <4:
            card_number_list.append(card_coding(hand_card_list[i])%10)
            class_number_list.append(card_coding(hand_card_list[i])//10)
        elif card_coding(hand_card_list[i])//100 >3:
            card_word_list.append(card_coding(hand_card_list[i])//100)
    # print('class-list is:', class_number_list, card_word_list)
    card_number_distribution = Counter(class_number_list)
    card_word_distribution = Counter(card_word_list)
    number_of_kezi = 0
    whole_distribution = {}
    whole_distribution.update(card_number_distribution)
    whole_distribution.update(card_word_distribution)
    print(r"card's distribution:", whole_distribution)
    card_number_distribution_list = list(card_number_distribution)
    card_word_distribution_list = list(card_word_distribution)
    for i in card_number_distribution_list:
        # print('this hand-card have ' + str(card_distribution[i]) + ' ' + str(i))
        if card_number_distribution[i] == 3:
            # kezi_list.append(i)
            kezi_list.append(number_card[i//10-1]+str(i%10))
            number_of_kezi +=1
    for i in card_word_distribution_list:
        if card_word_distribution[i] == 3:
            kezi_list.append(word_card[i-4])
            number_of_kezi += 1
    # print(card_distribution_list)
    if number_of_kezi !=0:
        print('our hand-card have '+str(number_of_kezi)+' kezi with ', kezi_list)
    elif number_of_kezi ==0:
        print(r"ooh,we don't have any kezi 0.0")
    return

def duizi(hand_card_list):
    # check_list = []
    duizi_list = []
    class_number_list = []
    card_number_list = []
    card_word_list = []
    hand_card_list = arrange_card(hand_card_list)
    for i in range(len(hand_card_list)):
        if card_coding(hand_card_list[i])//100 <4:
            card_number_list.append(card_coding(hand_card_list[i])%10)
            class_number_list.append(card_coding(hand_card_list[i])//10)
        elif card_coding(hand_card_list[i])//100 >3:
            card_word_list.append(card_coding(hand_card_list[i])//100)
    # print('class-list is:', class_number_list, card_word_list)
    card_number_distribution = Counter(class_number_list)
    card_word_distribution = Counter(card_word_list)
    number_of_duizi = 0
    whole_distribitution = {}
    whole_distribitution.update(card_number_distribution)
    whole_distribitution.update(card_word_distribution)
    print(r"card's distribution:", whole_distribitution)
    card_number_distribution_list = list(card_number_distribution)
    card_word_distribution_list = list(card_word_distribution)
    for i in card_number_distribution_list:
        # print('this hand-card have ' + str(card_distribution[i]) + ' ' + str(i))
        if card_number_distribution[i] == 2:
            # kezi_list.append(i)
            duizi_list.append(number_card[i//10-1]+str(i%10))
            number_of_duizi +=1
    for i in card_word_distribution_list:
        if card_word_distribution[i] == 2:
            duizi_list.append(word_card[i-4])
            number_of_duizi += 1
    # print(card_distribution_list)
    if number_of_duizi !=0:
        print('our hand-card have '+str(number_of_duizi)+' duizi with ', duizi_list)
    elif number_of_duizi ==0:
        print(r"ooh,we don't have any duizi 0.0")
    return

def shunzi(hand_card_list):
    check_list = []
    for i in range(len(hand_card_list)):
        check_list.append(card_coding(hand_card_list[i]) // 10)
    # print(check_list)
    card_distribution = Counter(check_list)
    number_of_shunzi = 0
    # print(r"aba's info:", card_distribution, type(card_distribution))
    card_distribution_list = list(card_distribution)
    # for i in card_distribution_list:

    for i in card_distribution_list:
        # print('this hand-card have ' + str(card_distribution[i]) + ' ' + str(i))
        if card_distribution[i] == 2:
            number_of_shunzi += 1

    # print(card_distribution_list)
    print('our hand-card have ' + str(number_of_shunzi) + ' shunzi')
    return

# arranged_hand_card = arrange_card(card_store[0:13])
# draw_card(card_store)
#['s82', 'w82', 'z1', 's12', 'n2', 'x1', 's42', 's44', 'f2', 'p13', 'b2', 's11', 'p22']
# hand_card = ['s81', 's82', 's83', 's71', 's72', 's73', 'c1', 'c2', 'c3', 'b1', 'b2', 'd1', 'd2']
card_store, number_card, word_card= card_store_create()
print('origin card_store: ', len(card_store))
hand_card, card_store= draw_card(card_store, i=13)
print('hand_card and immediate card_store: ', len(hand_card), len(card_store))
hand_card = arrange_card(hand_card)
kezi(hand_card)
duizi(hand_card)




#['p84', 'w22', 'n2', 'p54', 'p44', 's94', 's32', 'p63', 'n1', 'b2', 's91', 'w51', 'p41']
# card_coding(arranged_hand_card[0])



