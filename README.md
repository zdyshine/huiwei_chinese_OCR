# huiwei_chinese_OCR
2019年华为汉字书法大赛
# 说明
汉字书法多场景识别，使用了两步策略：先检测在识别。  
## 参考代码
检测部分使用了两种方案：  
1.EAST：[ESAT](https://github.com/argman/EAST)  
  EAST在实际使用中，由于比赛方提供的数据是竖向排版且文字长度，大小不一，并且有136张图片字体倾斜严重。并不能很好的切合本次数据集。效果一般。  
2.Faster-Rcnn:[Faster-Rcnn](https://github.com/roytseng-tw/Detectron.pytorch)  
  Faster-Rcnn在实际使用中，整体的检测效果比EAST更好。具体配置可以查看config文件。  
3.识别，使用CRNN：[CRNN](https://github.com/Sierkinhane/crnn_chinese_characters_rec)  
  由于在比赛方要求不能使用预训练模型，故没有使用densent那个版本的ocr。在改动以后，效果还不错。 
## 改进
### 1.EAST：  
        1.基础网络VGG-->Resnet_v1_101<br>
        2.通过数据尺度统计，加入多尺度训练<br>
        3.对网络输出部分，引入残差信息<br>
        4.输出部分，借鉴FPN思想，但是需要在本地生成对应尺度的图片及label。对硬件要求较高。<br>
### 2.Faster-Rcnn:  
       1.通过数据分析，重新设计anchor:[0.5,1,2] 改为[0.1,0.5,1]  
       2.Roi polling改为Roi Align   
       3.随机多尺度训练  
       4.FPN算法  
       5.多尺度测试  
### 3.CRNN:  
       1.基础网络：VGG-->VGG19+BN  
       2.修改基础网络的maxpooling，是网络的输出更长：512*1*5-->512*1*41,以适应数据集横向较长，特征较多的情况  
       3.RNN部分的LSTM未改动。
       4.数据分析后，数据分布(14,1320),故设置了多尺度训练的策略。
## 数据增强
       数据增强很重要！！！  
       本次使用了：GaussianBlur，sharpen，Affine，GaussianNoise，Add，Multiply，contrastNorm-alization，PiecwiseAffine  
