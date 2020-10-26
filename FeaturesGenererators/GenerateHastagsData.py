#Order of execution: #1
import numpy as np
#PATH TO DATA
text_file = open("d:\\dane\\HARRISON\\tag_list.txt", "r")
lines = text_file.readlines()
unique_tags = []
all_tags_list = []
for a in range(len(lines)):
    my_line = lines[a].split(' ')
    for b in range(len(my_line) -1):
        unique_tags.append(my_line[b])
text_file.close()
unique_tags = list(set(unique_tags))
print(unique_tags)
print(len(unique_tags))
res_array = np.zeros([len(lines), len(unique_tags)]).astype(np.int)
for a in  range(len(lines)):
    print(str(a) + " of " + str(len(lines)))
    my_line = lines[a].split(' ')
    for b in range(len(my_line) - 1):
        my_index = unique_tags.index(my_line[b])
        res_array[a, my_index] = 1
print(res_array.shape)
np.savetxt("Features/HashtagsLabels.csv", res_array, delimiter=",",fmt='%d')
'''
file_object = open('UniqeHashtags.txt', 'w')
for a in range(len(unique_tags)):
    if a > 0:
        file_object.write("\n")
    file_object.write(unique_tags[a])
file_object.write("\n")
# Close the file
file_object.close()
'''
Y = np.genfromtxt('Features/HashtagsLabels.csv', delimiter=',', skip_header=0)
np.save('Features/HashtagsLabels.bin', Y)
print("Shape of Y: " + str(Y.shape))
#TEST
dd = np.load('../Features/HashtagsLabels.bin.npy')
print(dd.shape)