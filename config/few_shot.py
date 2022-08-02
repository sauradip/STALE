
################### 75:25 Split #######################

# split_t1_train = [] ## 75:25 split
# split_t1_test = [] ## 75:25 split

# with open('/home/phd/Desktop/sauradip_research/TAL/CLIP-TAL/CLIPGSM/GSMv4/splits/train_75_test_25/ActivityNet/train/split_0.list', 'r') as filehandle:
#     filecontents = filehandle.readlines()
# with open('/home/phd/Desktop/sauradip_research/TAL/CLIP-TAL/CLIPGSM/GSMv4/splits/train_75_test_25/ActivityNet/test/split_0.list', 'r') as filehandle:
#     filecontents1 = filehandle.readlines()

# for files in filecontents:
#     split_t1_train.append(files[:-1])

# for files1 in filecontents1:
#     split_t1_test.append(files1[:-1])

# #### addd background class #####

# # split_t1_train.append('Neutral')
# # split_t1_test.append('Neutral')

# ################### 50:50 Split #######################

# split_t2_train = [] ## 50:50 split
# split_t2_test = [] ## 50:50 split

# with open('/home/phd/Desktop/sauradip_research/TAL/CLIP-TAL/CLIPGSM/GSMv4/splits/train_50_test_50/ActivityNet/train/split_0.list', 'r') as filehandle:
#     filecontents2 = filehandle.readlines()
# with open('/home/phd/Desktop/sauradip_research/TAL/CLIP-TAL/CLIPGSM/GSMv4/splits/train_50_test_50/ActivityNet/test/split_0.list', 'r') as filehandle:
#     filecontents3 = filehandle.readlines()

# for files2 in filecontents2:
#     split_t2_train.append(files2[:-1])

# for files3 in filecontents3:
#     split_t2_test.append(files3[:-1])



# # split_t1_dict = {}

# # cnt_1 = 0

# t1_dict_train = {split_t1_train[i] : i for i in sorted(range(150))}
# t1_dict_test = {split_t1_test[i] : i for i in sorted(range(50))}

# # {'Baton twirling': 0, 'Beach soccer': 1, 'Blowing leaves': 2, 'Bullfighting': 3, 'Capoeira': 4, 'Croquet': 5, 'Disc dog': 6, 'Drum corps': 7, 'Fixing the roof': 8, 'Having an ice cream': 9, 'Kneeling': 10, 'Making a lemonade': 11, 'Playing beach volleyball': 12, 'Playing blackjack': 13, 'Rafting': 14, 'Removing ice from car': 15, 'Swimming': 16, 'Trimming branches or hedges': 17, 'Tug of war': 18, 'Using the monkey bar': 19, 'Waterskiing': 20, 'Welding': 21, 'Drinking coffee': 22, 'Zumba': 23, 'High jump': 24, 'Wrapping presents': 25, 'Cricket': 26, 'Preparing pasta': 27, 'Grooming horse': 28, 'Preparing salad': 29, 'Playing polo': 30, 'Long jump': 31, 'Tennis serve with ball bouncing': 32, 'Layup drill in basketball': 33, 'Cleaning shoes': 34, 'Shot put': 35, 'Fixing bicycle': 36, 'Using parallel bars': 37, 'Playing lacrosse': 38, 'Cumbia': 39, 'Tai chi': 40, 'Mowing the lawn': 41, 'Walking the dog': 42, 'Playing violin': 43, 'Breakdancing': 44, 'Windsurfing': 45, 'Removing curlers': 46, 'Archery': 47, 'Polishing forniture': 48, 'Playing badminton': 49}

# t2_dict_train = {split_t2_train[i] : i for i in sorted(range(100))}
# t2_dict_test = {split_t2_test[i] : i for i in sorted(range(100))}

# for i in range(100)
#     split_t1_dict[i] = {

#     }




base_class = ['Fun sliding down', 'Beer pong', 'Getting a piercing', 'Shoveling snow', 'Kneeling', 'Tumbling', 'Playing water polo', 'Washing dishes', 'Blowing leaves', 'Playing congas', 'Making a lemonade', 'Playing kickball', 'Removing ice from car', 'Playing racquetball', 'Swimming', 'Playing bagpipes', 'Painting', 'Assembling bicycle', 'Playing violin', 'Surfing', 'Making a sandwich', 'Welding', 'Hopscotch', 'Gargling mouthwash', 'Baking cookies', 'Braiding hair', 'Capoeira', 'Slacklining', 'Plastering', 'Changing car wheel', 'Chopping wood', 'Removing curlers', 'Horseback riding', 'Smoking hookah', 'Doing a powerbomb', 'Playing ten pins', 'Getting a haircut', 'Playing beach volleyball', 'Making a cake', 'Clean and jerk', 'Trimming branches or hedges', 'Drum corps', 'Windsurfing', 'Kite flying', 'Using parallel bars', 'Doing kickboxing', 'Cleaning shoes', 'Playing field hockey', 'Playing squash', 'Rollerblading', 'Playing drums', 'Playing rubik cube', 'Sharpening knives', 'Zumba', 'Raking leaves', 'Bathing dog', 'Tug of war', 'Ping-pong', 'Using the balance beam', 'Playing lacrosse', 'Scuba diving', 'Preparing pasta', 'Brushing teeth', 'Playing badminton', 'Mixing drinks', 'Discus throw', 'Playing ice hockey', 'Doing crunches', 'Wrapping presents', 'Hand washing clothes', 'Rock climbing', 'Cutting the grass', 'Wakeboarding', 'Futsal', 'Playing piano', 'Baton twirling', 'Mooping floor', 'Triple jump', 'Longboarding', 'Polishing shoes', 'Doing motocross', 'Arm wrestling', 'Doing fencing', 'Hammer throw', 'Shot put', 'Playing pool', 'Blow-drying hair', 'Cricket', 'Spinning', 'Running a marathon', 'Table soccer', 'Playing flauta', 'Ice fishing', 'Tai chi', 'Archery', 'Shaving', 'Using the monkey bar', 'Layup drill in basketball', 'Spread mulch', 'Skateboarding', 'Canoeing', 'Mowing the lawn', 'Beach soccer', 'Hanging wallpaper', 'Tango', 'Disc dog', 'Powerbocking', 'Getting a tattoo', 'Doing nails', 'Snowboarding', 'Putting on shoes', 'Clipping cat claws', 'Snow tubing', 'River tubing', 'Putting on makeup', 'Decorating the Christmas tree', 'Fixing bicycle', 'Hitting a pinata', 'High jump', 'Doing karate', 'Kayaking', 'Grooming dog', 'Bungee jumping', 'Washing hands', 'Painting fence', 'Doing step aerobics', 'Installing carpet', 'Playing saxophone', 'Long jump', 'Javelin throw', 'Playing accordion', 'Smoking a cigarette', 'Belly dance', 'Playing polo', 'Throwing darts', 'Roof shingle removal', 'Tennis serve with ball bouncing', 'Skiing', 'Peeling potatoes', 'Elliptical trainer', 'Building sandcastles', 'Drinking beer', 'Rock-paper-scissors', 'Using the pommel horse', 'Croquet', 'Laying tile', 'Cleaning windows', 'Fixing the roof', 'Springboard diving', 'Waterskiing', 'Using uneven bars', 'Having an ice cream', 'Sailing', 'Washing face', 'Knitting', 'Bullfighting', 'Applying sunscreen', 'Painting furniture', 'Grooming horse', 'Carving jack-o-lanterns']
val_class = ['Swinging at the playground', 'Dodgeball', 'Ballet', 'Playing harmonica', 'Paintball', 'Cumbia', 'Rafting', 'Hula hoop', 'Cheerleading', 'Vacuuming floor', 'Playing blackjack', 'Waxing skis', 'Curling', 'Using the rowing machine', 'Ironing clothes', 'Playing guitarra', 'Sumo', 'Putting in contact lenses', 'Brushing hair', 'Volleyball']
test_class = ['Hurling', 'Polishing forniture', 'BMX', 'Riding bumper cars', 'Starting a campfire', 'Walking the dog', 'Preparing salad', 'Plataform diving', 'Breakdancing', 'Camel ride', 'Hand car wash', 'Making an omelette', 'Shuffleboard', 'Calf roping', 'Shaving legs', 'Snatch', 'Cleaning sink', 'Rope skipping', 'Drinking coffee', 'Pole vault']

base_dict = { base_class[i] : i for i in sorted(range(len(base_class)))}
val_dict = { val_class[i] : i for i in sorted(range(len(val_class)))}
test_dict = { test_class[i] : i for i in sorted(range(len(test_class)))}


base_train = base_class + val_class
base_train_dict = { base_train[i] : i for i in sorted(range(len(base_train)))}



