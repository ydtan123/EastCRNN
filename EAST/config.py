#data
dataroot='./dataset_resnet152'  #./data/train/img      ./data/train/gt
test_img_path='./dataset_resnet152/test'
result = './result_resnet152'
number_of_test = 50

lr = 0.0001
gpu_ids = [0]
gpu = 1
init_type = 'xavier'

resume = False
checkpoint = 'checkpoint'# should be file
train_batch_size_per_gpu  = 14
num_workers = 1

print_freq = 1
eval_iteration = 50
save_iteration = 50
max_epochs = 1000000







