#data
dataroot='./dataset_7w'  #./data/train/img      ./data/train/gt
test_img_path='./dataset_7w/test'
result = './result_7w'
number_of_test = 50

lr = 0.0001
gpu_ids = [0] #[0,1]
gpu = 1 #2
init_type = 'xavier'

resume = False
checkpoint = 'checkpoint'# should be file
train_batch_size_per_gpu  = 10 #14
num_workers = 1

print_freq = 1
eval_iteration = 50
save_iteration = 20
max_epochs = 1000000







