import thulac

# thu1 = thulac.thulac(seg_only=True)
# # text = thu1.cut(" 我很难过，安慰我~",text=True)
# # print(text)
# import re
# lines = open("xiaohuangji50w_nofenci.conv",encoding = 'utf-8').read().strip().split('\n')
# print(len(lines))
# print(lines[0]=='E')

# a = re.search(r'[^M]+',lines[200]).group()
# text = thu1.cut(a,text=True).split(' ')
# print(len(text))
# print(a.strip())
# for i in range(1000000):
#     if lines[i]=='E' and i%3!=0:
#         print('i=%d'%i)
#         print(False)
#         break


import itertools

a = [[1,2,3],[4,5],[6,7,8,9]]
b = itertools.zip_longest(*a,fillvalue = 0)
print(list(b))