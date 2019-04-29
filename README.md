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
1.EAST：  
       基础网络VGG-->Resnet_v1_101
       通过数据尺度统计，加入多尺度训练。  
       对网络输出部分，引入残差信息。  
       输出部分，借鉴FPN思想，但是需要在本地生成对应尺度的图片及label。对硬件要求较高。  
2.Faster-Rcnn:  
       通过数据分析，重新设计anchor:[0.5,1,2] 改为[0.1,0.5,1]  
       Roi polling改为Roi Align   
       随机多尺度训练  
       FPN算法  
       多尺度测试  
       
