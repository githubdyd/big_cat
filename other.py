import jieba
import re
import settings
#line = "送货挺快，价格不贵，要是有赠品就好了！"
line = "我之前的都是大瓶150ml的，结果坐火车旅游过不了安检被收了，知道还有小瓶装的才来买的，是有二维码，但是我没有扫，也是放在阳光下会变紫色。但是刚刚打开喷时酒精味很重，而且与我之前买的大瓶装的香味不同，也不知道是不是我的错觉。"
line =  line.replace("，", "").replace("《", "").replace("》", "").replace("。", "").replace("/", "").replace("？","").replace(
            "；", "").replace("：", "").replace("‘", "").replace("’", "").replace("“", "").replace("”", "").replace("【", "").replace(
            "】", "").replace("{", "").replace("}", "").replace("、", "").replace("|", "").replace("（", "").replace("）","").replace(
            "*", "").replace("&", "").replace("……", "").replace("%", "").replace("￥", "").replace("#", "").replace("@","").replace(
            "！", "").replace("~", "").replace("`", "").replace("_", "").replace("-", "").replace("+", "").replace("=", "")
line = line.replace(",", "").replace(".", "").replace("/", "").replace(";", "").replace("'", "").replace("<","").replace(
    ">", "").replace("?", "").replace(":", "").replace("\"", "").replace("[", "").replace("]", "").replace("{", "").replace(
    "}", "").replace("\\", "").replace("|", "").replace("·", "").replace("~", "").replace("！", "").replace("～", "")
line = line.replace("ಡ", "")
seg_list = jieba.cut(line, cut_all=False)  # 分词 精确模式
line = " ".join(seg_list)
line = line.split(" ")
print(line)
# 导入停用词表
sf = open(settings.stop_words, encoding="utf-8-sig")
stopword = []
for m in sf:
    m = m.rstrip("\n")
    stopword.append(m)
print(stopword)

after_line = ""
for i in line:
    if i not in stopword:
        after_line += i
print(after_line)

seg_list = jieba.cut(after_line, cut_all=False)  # 再分词
line = " ".join(seg_list)
line = line.replace("不 ", "不")
r1 = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
line = re.sub(r1, '', line)
line = line.replace("  ", "")
line = line.lstrip(" ")
print("分词结果:",line)