###############
# 2022年1月4日： Test1
# Update Details: 
# 1. 暂时使用 fastai 的框架，根据 kaggle 中讨论区的方法，进行初步常识性测试
# 2. 原数据集中存在重复图片 且重复图片标签积分不一致， 换用  kaglle 讨论区中提供的 clean 版本数据集
# 3. 模型就使用最原始的 swin_tr 不加入任何修改
#
# 2022年1月5日： Test3
# 1. 删除 earlystop 跑满全部10个 epoch
# 2. 学习率 暂确定仍然为 2e-5
# 3. 加入 csv 的 callback
# 4. 数据集替换为 清理过后的数据集  从 9912 -> 98xx 张
# 5. fold的 保存 pth 从 0 开始计数
# 6. 取消 Mixup  固定 seed为 365
#
# 2022年1月5日： Test4
# 1. 删除 earlystop 跑满全部10个 epoch
# 2. 学习率 暂确定仍然为 2e-5
# 3. 加入 csv 的 callback
# 4. 数据集替换为 清理过后的数据集  从 9912 -> 98xx 张
# 5. fold的 保存 pth 从 0 开始计数
# 6. 使用 384 学习率 定为2e-5 进行测试

# 2022年1月6日： Test5
# 1. 为了节约时间 重新加入 EarlyStop  Patiance 设置为3
# 2. 这次的图片为使用YOLOV5预处理过后，只提取出图片中的宠物部分
# 3. 使用 精度更高的 fp_32运行

# 2022年1月6日： Test6
# 1. 为了节约时间 重新加入 EarlyStop Patiance 设置为3
# 2. 这次的图片为使用YOLOV5预处理过后，只提取图片中的宠物部分
# 3. 维持 fp_16 因为模型更加复杂的原因

# 2022年1月6日: Test7
# 1. 将原 test9的模型结构引入  通过 hack 两个关键函数 得以实现
# 2. 修改原 loss 的计算方式，为后续使用 catboost 做准备

# 2022年1月7日： Test9
# 1. 加入 dropout
# 2. 暂时不再加入 yolov5 预处理的图片
# 3. 保持 embed方法  以便于catboost的使用

# 2022年1月7日： Test10
# 1. 加入 dropout
# 2. 暂时不再加入 yolov5 预处理的图片
# 3. 保持 embed方法  以便于catboost的使用
# 4. bin划分的数量予以修复

# 2022年1月7日： Test11
# 1. 相比 test9 对numbin的数量予以修正

# 2022年1月9日： Test12
# 1. 基于 Test4 进行 保留 Test4的一切设定  同时Mixup 下调为 0.2  
# 2. 使用newbin
###############

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
import sys
from timm import create_model
# from timm.data.mixup import Mixup
from fastai.vision.all import *
import random
from sklearn.model_selection import StratifiedKFold
from toolbox.log_writers.log import get_logger
import gc
from sklearn.metrics import mean_squared_error
from shutil import copyfile  # 用于复制文件

RANDOMSEED = random.randint(2000,9000)

hyper_parameter_group ={
    # 主要环境参数设定
    # test1 到  test7 都是使用这个 随机种子
    "seed": RANDOMSEED,
    "out_dir": r"/storage/Kaggle_Pet_Finder/model/Swin_Tr_224/fastai_test12",
    "log_file_name": "Fastai_SwinTr_test12",
    # 数据集设定
    "image_resize_resolution": 384,
    # skf n-fold参数设定
    "skf_fold": 10,
    # 模型设定
     #  test1 2 : swin_base_patch4_window7_224_in22k
    # test 3 5 6 7 swin_large_patch4_window12_384_in22k
    # test 4 9  13 swin_large_patch4_window7_224   完成测试
    # test 8 swin_base_patch4_window12_384_in22k
    # test 10 swin_large_patch4_window12_384   完成测试
    # test 12  swin_large_patch4_window7_224_in22k
    "encoder_name" : "swin_large_patch4_window12_384",
    "output_dims": 1,
    # 学习率计划 优化器 相关
    "total_epoch" : 10,
    "batch_size" : 20,
    "start_lr" : 2e-5,
    # excel 中的meta data 信息
    "dense_features": [
        'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
        'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
    ],
}

# 设定随机种子，方便复现代码
def set_seeds(seed= hyper_parameter_group["seed"] ):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 定义评估函数
def petfinder_rmse(input,target):
    # 避免出现之前的问题 这里判断一下 他们的范围
    # print(input.shape)
    # print(target.shape)
    # 2022年1月4日 更新： 这里有潜在的问题  需要做出修复
    # 已使用 AccumMetric 方法做出修复 此方法类似于一个装饰器，会自动存储所有的 eval 结果 然后喂给 评估函数方法
    metric = np.sqrt(mean_squared_error( target * 100., torch.sigmoid(input.flatten()) * 100.))
    return metric



# 定义数据获取函数
def get_data(fold):
    train_df_f = train_df.copy()
    train_df_f['is_valid'] = (train_df_f['fold'] == fold)

    dls = ImageDataLoaders.from_df(train_df_f, #pass in train DataFrame
                               valid_col='is_valid', #
                               seed=hyper_parameter_group["seed"], #seed
                               fn_col='path', #filename/path is in the second column of the DataFrame
                               label_col='norm_score', #label is in the first column of the DataFrame
                               y_block=RegressionBlock, #The type of target
                               bs=hyper_parameter_group["batch_size"], #pass in batch size
                               num_workers=8,
                               item_tfms=Resize(hyper_parameter_group["image_resize_resolution"]), #pass in item_tfms
                               batch_tfms=setup_aug_tfms([Brightness(), Contrast(), Hue(), Saturation()])) #pass in batch_tfms
    
    return dls

# 定义模型结构
def get_learner(fold_num):
    data = get_data(fold_num)
    
    model = create_model(hyper_parameter_group["encoder_name"], pretrained=True, num_classes=data.c)

    learn = Learner(data, model, 
    path = hyper_parameter_group["out_dir"],
    loss_func=BCEWithLogitsLossFlat(), 
    metrics= [AccumMetric(func=petfinder_rmse)], 
    cbs=[MixUp(0.2)]).to_fp16()
    
    return learn

# Step0. outdir准备， 日志记录准备
os.makedirs(hyper_parameter_group["out_dir"],exist_ok= True)
# 2021年7月14日更新： 在输出目录下 放入 source_code目录 存储每一次运行时候的 源代码赋值一份加入
for f in ['source_code']:
    os.makedirs(hyper_parameter_group["out_dir"] + '/' + f, exist_ok=True)
# 2022年1月4日更新： 将 每次的 split csv 文件进行存储
for f in ['csv_split_save']:
    os.makedirs(hyper_parameter_group["out_dir"] + '/' + f, exist_ok=True)

logdir = hyper_parameter_group["out_dir"] + '/log'
logger = get_logger(logdir,OutputOnConsole = True,log_initial= hyper_parameter_group["log_file_name"],logfilename=hyper_parameter_group["log_file_name"])
stdout = logger.info

# 重定向 print 
# sys.stdout = logger.info

stdout('** hyper parameter setting **\n')
stdout('hyper parameter group : \n%s\n' % (hyper_parameter_group))
stdout('\n')

device = 'cuda'

name = os.path.basename(__file__)
stdout('Store Source_Code :')
stdout('\n')
copyfile(str(name), hyper_parameter_group["out_dir"] + "/source_code/train_source_code.py")

# Step1. 设定随机数种子  方便复现代码
set_seeds(hyper_parameter_group["seed"])

# Step2. 读入csv 文件 并用bin划分
train_df = pd.read_csv('/storage/Kaggle_Pet_Finder/clean_data_version1/train.csv')
train_df.head()
dataset_path = Path("/storage/Kaggle_Pet_Finder/clean_data_version1/")


train_df['path'] = train_df['Id'].map(lambda x:str(dataset_path/'train'/x)+'.jpg')
train_df = train_df.drop(columns=['Id'])
train_df = train_df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
train_df.head()
len_df = len(train_df)
stdout(f"There are {len_df} images")
train_df['norm_score'] = train_df['Pawpularity']/100
train_df['norm_score']

# num_bins = int(np.floor(1+np.log2(len(train_df))))
num_bins = int(np.ceil(2*(len(train_df)**(1/3)) ))
stdout(num_bins)
train_df['bins'] = pd.cut(train_df['norm_score'], bins=num_bins, labels=False)
train_df['fold'] = -1
strat_kfold = StratifiedKFold(n_splits=hyper_parameter_group["skf_fold"], random_state=hyper_parameter_group["seed"], shuffle=True)
for i, (_, train_index) in enumerate(strat_kfold.split(train_df.index, train_df['bins'])):
    train_df.iloc[train_index, -1] = i
train_df['fold'] = train_df['fold'].astype('int')

train_df.to_csv(os.path.join(hyper_parameter_group["out_dir"],"csv_split_save","split_save_seed{}.csv".format(hyper_parameter_group["seed"])))

# Step3. 开始训练
for i in range(hyper_parameter_group["skf_fold"]):
    stdout(f'Fold {i} results')
    learn = get_learner(fold_num=i)
    learn.model = torch.nn.DataParallel(learn.model)
    learn.fit_one_cycle(hyper_parameter_group["total_epoch"], 
    hyper_parameter_group["start_lr"], 
    cbs=[SaveModelCallback('petfinder_rmse',comp=np.less,fname = "best_model_fold{}".format(i)),
         CSVLogger(fname = "fold{}_history.csv".format(i)),
        # EarlyStoppingCallback(monitor='petfinder_rmse', comp=np.less, patience=3)
        ]) 
    learn.recorder.plot_loss()
    plt.savefig(os.path.join(logdir,'loss_plot_fold{}.png'.format(i+1))) # 保存训练过程中的 plot图片
    # learn.export(os.path.join(hyper_parameter_group["out_dir"],"checkpoint",'model_fold_{}.pkl'.format(i)))
    del learn
    torch.cuda.empty_cache()
    gc.collect()



