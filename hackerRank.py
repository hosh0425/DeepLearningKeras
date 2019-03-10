# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 02:28:11 2019

@author: Hossein
"""

if __name__ == '__main__':

    total_list=list()
    for _ in range(int(input())):
        name = input()
        score = float(input())
        mlist=list()
        mlist.append(name)
        mlist.append(score)
        total_list.append(mlist)
    total_list.sort(key=lambda total_list: total_list[0])
    min_score=total_list[0][1]
    names=list()
    second_low=0
    for item in total_list:
        if min_score != item[1]:
            names.append(item[0])
            second_low=item[1]
        if(second_low!=0):
            if(second_low != item[1]):
                break;
    for n in names:
        print(n)