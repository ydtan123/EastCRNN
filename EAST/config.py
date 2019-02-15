#data
dataroot='./dataset_digit_75k'  #./data/train/img      ./data/train/gt
test_img_path='./dataset_digit_75k/test'
result = './result_digit_75k'
number_of_test = 50

lr = 0.0001
gpu_ids = [0]
gpu = 1
init_type = 'xavier'

resume = False
checkpoint = 'checkpoint'# should be file
train_batch_size_per_gpu  = 10 #14
num_workers = 1

print_freq = 1
eval_iteration = 20
save_iteration = 20
max_epochs = 1000000







