## data part
对于这个比赛，我一共尝试了三种不同的数据集变体。
1. 1.不经过任何变化的原始数据集。
2. 2.经过YOLOV5预处理，因为图片的分辨率有些过于大，而swin系列最高支持分辨率为384。这个分辨率对于2000的像素图片过于巨大，因此考虑使用YOLOV5将图片中的宠物部分提取出来。
3. 3.抛出train数据集中的完全一样的图片。

对于1. 中的数据集，请直接下载官方的[数据链接地址](https://www.kaggle.com/c/petfinder-pawpularity-score/data)

对于2. 中的数据集，请参考如下[notebook](https://www.kaggle.com/shinewine/yolov5-preprossed)

对于3. 中的数据集，请参考如下[notebook](https://www.kaggle.com/schulta/petfinder-pawpularity-score-clean)

## 相关思考

根据目前打比赛的经验，额外的数据集几乎百分百对最终结果都有所提升。
同时额外数据集的使用方法也是很有技巧性的。需要针对不同的数据使用不同的方法。
例如根据这个比赛的金牌区方法，这个比赛存在一个额外的数据集，将那个数据集中的相似图片的metadata提取出来，可以非常好的提升最终的结果。


