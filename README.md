# huiwei_chinese_OCR
2019年华为汉字书法大赛
# 说明
汉字书法多场景识别，使用了两步策略：先检测在识别。  
检测部分使用了两种方案：  
1.EAST：[ESAT](https://github.com/argman/EAST)  
  EAST在实际使用中，由于比赛方提供的数据是竖向排版且文字长度，大小不一，并且有136张图片字体倾斜严重。并不能很好的切合本次数据集。效果一般。  
2.Faster-Rcnn:[Faster-Rcnn](https://github.com/roytseng-tw/Detectron.pytorch)  
  Faster-Rcnn在实际使用中，整体的检测效果比EAST更好。具体配置可以查看config文件。  
