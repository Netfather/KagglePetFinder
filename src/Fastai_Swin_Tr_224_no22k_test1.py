###############
# 2022年1月4日： Test1
# Update Details: 
# 1. 暂时使用 fastai 的框架，根据 kaggle 中讨论区的方法，进行初步常识性测试
# 2. 原数据集中存在重复图片 且重复图片标签积分不一致， 换用  kaglle 讨论区中提供的 clean 版本数据集
# 3. 模型就使用最原始的 swin_tr 不加入任何修改
#
#
# 待定修改： 删除 earlystop  重新确认学习率  加入 csv的log方便记录本地cv  data换成 clean过的 dataset   保存格式改为 pickle 存储   fold 从 0开始计数
###############
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
from timm import create_model
# from timm.data.mixup import Mixup
from fastai.vision.all import *
import random
from sklearn.model_selection import StratifiedKFold
from toolbox.log_writers.log import get_logger
import gc
from sklearn.metrics import mean_squared_error

RANDOMSEED = random.randint(2000,9000)

hyper_parameter_group ={
    # 主要环境参数设定
    # test1 到  test7 都是使用这个 随机种子
    "seed": RANDOMSEED,
    "out_dir": r"/storage/Kaggle_Pet_Finder/model/Swin_Tr_224/fastai_test1",
    "log_file_name": "Fastai_SwinTr_test1",
    # 数据集设定
    "image_resize_resolution": 224,
    # skf n-fold参数设定
    "skf_fold": 10,
    # 模型设定
     #  test1 2 : swin_base_patch4_window7_224_in22k
    # test 3 5 6 7 swin_large_patch4_window12_384_in22k
    # test 4 9  13 swin_large_patch4_window7_224   完成测试
    # test 8 swin_base_patch4_window12_384_in22k
    # test 10 swin_large_patch4_window12_384   完成测试
    # test 12  swin_large_patch4_window7_224_in22k
    "encoder_name" : "swin_large_patch4_window7_224",
    "output_dims": 1,
    # 学习率计划 优化器 相关
    "total_epoch" : 10,
    "batch_size" : 32,
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
    metric = np.sqrt(mean_squared_error( target * 100., F.sigmoid(input.flatten()) * 100.))
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
    
    model = create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=data.c)

    learn = Learner(data, model, path = os.path.join(hyper_parameter_group["out_dir"],"checkpoint"),
    loss_func=BCEWithLogitsLossFlat(), metrics= [AccumMetric(func=petfinder_rmse)], cbs=[MixUp(0.4)]).to_fp16()
    
    return learn

# 定义 callback 每32次之后 进行一次 评估  暂时没有修改完成 待定
# class ValidateBatch(Callback):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.total_bach_iteration = 0
    
#     def before_batch(self):
#         self.total_bach_iteration = self.total_bach_iteration + 1
#         if (self.total_bach_iteration % 32 == 0 and self.total_bach_iteration != 0):
#             print( " Start Custom Validate !!!")


# Step0. outdir准备， 日志记录准备
os.makedirs(hyper_parameter_group["out_dir"],exist_ok= True)
# 在output目标路径下 新建一个文件夹名字为 checkpoint
for f in ['checkpoint']:
    os.makedirs(hyper_parameter_group["out_dir"] + '/' + f, exist_ok=True)
# 在output目标路径下 新建一个文件夹名字为 checkpoint_best 用于存储 最优的验证结果
for f in ['checkpoint_best']:
    os.makedirs(hyper_parameter_group["out_dir"] + '/' + f, exist_ok=True)
# 2021年7月14日更新： 在输出目录下 放入 source_code目录 存储每一次运行时候的 源代码赋值一份加入
# 2022年1月4日更新： 将 每次的 split csv 文件进行存储
for f in ['csv_split_save']:
    os.makedirs(hyper_parameter_group["out_dir"] + '/' + f, exist_ok=True)

logdir = hyper_parameter_group["out_dir"] + '/log'
logger = get_logger(logdir,OutputOnConsole = True,log_initial= hyper_parameter_group["log_file_name"],logfilename=hyper_parameter_group["log_file_name"])
stdout = logger.info

stdout('** hyper parameter setting **\n')
stdout('hyper parameter group : \n%s\n' % (hyper_parameter_group))
stdout('\n')

device = 'cuda'

# Step1. 设定随机数种子  方便复现代码
set_seeds(hyper_parameter_group["seed"])

# Step2. 读入csv 文件 并用bin划分
train_df = pd.read_csv('/storage/Kaggle_Pet_Finder/data/train.csv')
train_df.head()
dataset_path = Path("/storage/Kaggle_Pet_Finder/data/")


train_df['path'] = train_df['Id'].map(lambda x:str(dataset_path/'train'/x)+'.jpg')
train_df = train_df.drop(columns=['Id'])
train_df = train_df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
train_df.head()
len_df = len(train_df)
stdout(f"There are {len_df} images")
train_df['norm_score'] = train_df['Pawpularity']/100
train_df['norm_score']

num_bins = int(np.floor(1+np.log2(len(train_df))))
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
    learn.fit_one_cycle(hyper_parameter_group["total_epoch"], 
    hyper_parameter_group["start_lr"], 
    cbs=[SaveModelCallback('petfinder_rmse',comp=np.less,fname = "best_model_fold{}".format(i+1))
        # EarlyStoppingCallback(monitor='petfinder_rmse', comp=np.less, patience=3)
        ]) 
    learn.recorder.plot_loss()
    plt.savefig(os.path.join(logdir,'loss_plot_fold{}.png'.format(i+1))) # 保存训练过程中的 plot图片
    # learn.export(os.path.join(hyper_parameter_group["out_dir"],"checkpoint",'model_fold_{}.pkl'.format(i)))
    del learn
    torch.cuda.empty_cache()
    gc.collect()



