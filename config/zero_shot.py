
################### 75:25 Split #######################

split_t1_train = [] ## 75:25 split
split_t1_test = [] ## 75:25 split

with open('/home/phd/Desktop/sauradip_research/TAL/CLIP-TAL/CLIPGSM/GSMv4/splits/train_75_test_25/ActivityNet/train/split_0.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('/home/phd/Desktop/sauradip_research/TAL/CLIP-TAL/CLIPGSM/GSMv4/splits/train_75_test_25/ActivityNet/test/split_0.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()

for files in filecontents:
    split_t1_train.append(files[:-1])

for files1 in filecontents1:
    split_t1_test.append(files1[:-1])

#### addd background class #####

# split_t1_train.append('Neutral')
# split_t1_test.append('Neutral')

################### 50:50 Split #######################

split_t2_train = [] ## 50:50 split
split_t2_test = [] ## 50:50 split

with open('/home/phd/Desktop/sauradip_research/TAL/CLIP-TAL/CLIPGSM/GSMv4/splits/train_50_test_50/ActivityNet/train/split_0.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('/home/phd/Desktop/sauradip_research/TAL/CLIP-TAL/CLIPGSM/GSMv4/splits/train_50_test_50/ActivityNet/test/split_0.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()

for files2 in filecontents2:
    split_t2_train.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test.append(files3[:-1])



# split_t1_dict = {}

# cnt_1 = 0

t1_dict_train = {split_t1_train[i] : i for i in sorted(range(150))}
t1_dict_test = {split_t1_test[i] : i for i in sorted(range(50))}

# {'Baton twirling': 0, 'Beach soccer': 1, 'Blowing leaves': 2, 'Bullfighting': 3, 'Capoeira': 4, 'Croquet': 5, 'Disc dog': 6, 'Drum corps': 7, 'Fixing the roof': 8, 'Having an ice cream': 9, 'Kneeling': 10, 'Making a lemonade': 11, 'Playing beach volleyball': 12, 'Playing blackjack': 13, 'Rafting': 14, 'Removing ice from car': 15, 'Swimming': 16, 'Trimming branches or hedges': 17, 'Tug of war': 18, 'Using the monkey bar': 19, 'Waterskiing': 20, 'Welding': 21, 'Drinking coffee': 22, 'Zumba': 23, 'High jump': 24, 'Wrapping presents': 25, 'Cricket': 26, 'Preparing pasta': 27, 'Grooming horse': 28, 'Preparing salad': 29, 'Playing polo': 30, 'Long jump': 31, 'Tennis serve with ball bouncing': 32, 'Layup drill in basketball': 33, 'Cleaning shoes': 34, 'Shot put': 35, 'Fixing bicycle': 36, 'Using parallel bars': 37, 'Playing lacrosse': 38, 'Cumbia': 39, 'Tai chi': 40, 'Mowing the lawn': 41, 'Walking the dog': 42, 'Playing violin': 43, 'Breakdancing': 44, 'Windsurfing': 45, 'Removing curlers': 46, 'Archery': 47, 'Polishing forniture': 48, 'Playing badminton': 49}

t2_dict_train = {split_t2_train[i] : i for i in sorted(range(100))}
t2_dict_test = {split_t2_test[i] : i for i in sorted(range(100))}

# for i in range(100)
#     split_t1_dict[i] = {

#     }


# split_t2_dict = {}

# print(t1_dict_train)

# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

# import torch
# from denseclip.untils import tokenize
# import numpy as np

# def text_prompt(classes,context):
#     # print(classes)
#     text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
#                 f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
#                 f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
#                 f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
#                 f"The man is {{}}", f"The woman is {{}}"]
#     text_dict = {}
#     num_text_aug = len(text_aug)

#     for ii, txt in enumerate(text_aug):

#         text_dict[ii] = torch.cat([tokenize(txt.format(c),context_length=context) for c in classes])
#         # text_dict.append([tokenize(txt.format(c),context_length=context) for c in classes])
#     text_dict_fin = torch.cat([text_dict[t] for t in text_dict.keys()])
#     # text_dict = np.array(text_dict)
    
#     # text_dict = np.vstack(text_dict).astype(np.float)
#     # text_dict = torch.cat(text_dict)
#     # classes_list = torch.cat([v for k, v in text_dict.items()])

#     return text_dict_fin