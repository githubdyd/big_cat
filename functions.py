import sqlite3
import jieba
import re
import numpy
from gensim.models import word2vec
from gensim import corpora,models,similarities
import gensim
import torch
import torch.nn.functional as F
import settings


# 将初始文本读入数据库
def txt_to_database(origin_file,db_name):   # .txt;.db
    cd = sqlite3.connect(db_name)  # 打开
    print("open database success")
    c = cd.cursor()
   # c.execute("drop table comment")  # 删旧表

    # 建表
    c.execute('''create table comment
                (ID INT PRIMARY KEY NOT NULL,
                content char);''')
    print("table create successfully")

    # 插入数据
    line_count = 1
    for line in open(origin_file, "r", encoding="utf-8-sig"):
        c.execute(" insert into comment(id,content)\
                values (?,?)", (line_count, line))
        line_count += 1

    cursor = c.execute("SELECT ID,content from comment")  # 读
    for row in cursor:
        print("id = ", row[0])
        print("content = " + row[1])

    cd.commit()
    cd.close()


# 针对已标注数据，选取其中有效数据（指口红）
def select_valid():
    f = open(settings.origin_file, encoding="utf-8")
    w = open(settings.valid_origin, "w+", encoding="utf-8")
    for line in f:
        print(line)
        if ("有效评论" in line):
            w.write(line)
    f.close()
    w.close()


# 处理口红评论
def treat_test_comment():
    f = open(settings.valid_origin,encoding="utf-8-sig")
    frev = open(settings.useful_comment, "w+", encoding="utf-8")
    ftag = open(settings.useful_targets, "w+", encoding="utf-8")
    fatt = open(settings.train_attitude, "w+", encoding="utf-8")
    for line in f:
        if("positive" in line):
            attitude = 1
        else:
            attitude = 0
        comment = line.split("：,")[0]
        tags = line.split("：,")[1]
        comment = comment.split(":",1)[1]
        comment = comment.split("\n")
        # 此时comment为一个line的评论内容string
        sign = r"-（.+?\）"   # 正则
        rule = re.compile(sign)
        selected_tags = rule.findall(tags)
        # selected_tags为 ['-（产品（笼统）', '-（颜色深浅）', '-（促销赠品）']这种

        # 检测是否为笼统评论：所有标签均为笼统标签的评论
        for i in selected_tags:
            if("笼统" in i):
                longtong_count += 1
        longtong_count = 0
        longtong_tag = 0
        if(longtong_count == len(selected_tags)):
            longtong_tag = 1

        # 去除非笼统评论中的笼统标签，输出为final_tags
        final_tags = []
        att_list = []
        if(longtong_tag == 0):  # 非笼统评论
            for i in selected_tags:
                if("笼统" not in i):
                    t = i.rstrip("）'").lstrip("'-（")
                    final_tags.append(t)
            att_list.append(attitude)

        for i,j,k in zip(comment,final_tags,att_list):
            if(len(j)!= 0):
                frev.writelines(i + "\n")
             #   print("j:",j)
                for p in final_tags:
                    p = p.replace("/ ","/")
                    ftag.write(p + " ")
            ftag.writelines("\n")
            fatt.write(str(k))
            fatt.write("\n")
    f.close()
    frev.close()
    ftag.close()
    fatt.close()

# 制作口红标签集
def make_target_list():
    f = open(settings.useful_targets, encoding= "utf-8-sig")
    number = 0
    target_list = {}

    # 返回字典格式
    for i in f:
        one_line = i.rstrip(" \n")
        one_line = one_line.split(" ")
        for one_target in one_line:
            if(one_target not in target_list.keys()):
                target_list[one_target] = number
                number += 1
    return target_list
    '''
        # 返回列表格式
        target_list = []
        for i in f:
            one_line = i.rstrip(" \n")
            one_line = one_line.split(" ")
            for one_target in one_line:
                if(one_target not in target_list):
                    target_list.append(one_target)
        return target_list
    '''


# 生成loss函数需要的标签独热编码
def target_tensor(useful_targets,save_file):    # 输入为存储标签文件名;存储的.npy名
    target_dict = make_target_list()
    f = open(useful_targets, encoding="utf-8")
    full_list = []
    for line in f:
        line_list = []
        line = line.rstrip(" \n").split(" ")
        '''
        for word in line:
            #line_list.append(target_dict[word])
            line_list.append(1)

        while(len(line_list )< 52):
            line_list.append(0)     # bce的target为[1,0,1...]这样的
        '''
        while(len(line_list) < 52):
            line_list.append(0)
        for word in line:
            line_list[target_dict[word]] = 1

        full_list.append(line_list)
    full_list = numpy.array(full_list)
    numpy.save(save_file, full_list)


# 分词，去停用词等，从数据库读入，写至txt
def select_sentences(need_fenci_comments,result_name):
    w = open(result_name, "w", encoding="utf-8")    # 待写入
    # 读
    #cursorc = c.execute("SELECT ID,content from comment")
    # 导入停用词表
    sf = open(settings.stop_words, encoding="utf-8-sig")
    stopword = []
    for line in sf:
        line = line.rstrip("\n")
        stopword.append(line)
    print(stopword)
    sf.close()
    f = open(need_fenci_comments,encoding="utf-8-sig")
    count = 1
    for data in f:
        count += 1
        # 去中文符号

        line = data.replace("，", "").replace("《", "").replace("》", "").replace("。", "").replace("/", "").replace("？","").replace(
            "；", "").replace("：", "").replace("‘", "").replace("’", "").replace("“", "").replace("”", "").replace("【", "").replace(
            "】", "").replace("{", "").replace("}", "").replace("、", "").replace("|", "").replace("（", "").replace("）","").replace(
            "*", "").replace("&", "").replace("……", "").replace("%", "").replace("￥", "").replace("#", "").replace("@","").replace(
            "！", "").replace("~", "").replace("`", "").replace("_", "").replace("-", "").replace("+", "").replace("=", "")

        # 去英文符号
        line = line.replace(",", "").replace(".", "").replace("/", "").replace(";", "").replace("'", "").replace("<","").replace(
            ">", "").replace("?", "").replace(":", "").replace("\"", "").replace("[", "").replace("]", "").replace("{", "").replace(
            "}", "").replace("\\", "").replace("|", "").replace("·", "").replace("~", "").replace("！", "").replace("～", "")
        line = line.replace("ಡ", "")
        seg_list = jieba.cut(line, cut_all=False)  # 分词 精确模式
        # 插入数据

        line = " ".join(seg_list)  # 去除停用词

        x = line.split(" ")
        line = ""
        for i in x:
            if i not in stopword:
                line += i

        seg_list = jieba.cut(line, cut_all=False)  # 再分词
        line = " ".join(seg_list)
        line = line.replace("不 ", "不")
        r1 = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        line = re.sub(r1, '', line)
        line = line.replace("  ", "")
        line = line.lstrip(" ")
        if (line.isspace()):  # 清洗后评论可能没有有效信息，则放弃该评论
            continue
        w.writelines(line)
        #n.execute(" insert into comment(id,content)\
         #           values (?,?)", (data[0], line))
    w.close()



# 计算词向量，从分词结果txt读入，模型保存至model
def cal_vec(fenci_result,save_model):
    sentences = word2vec.Text8Corpus(fenci_result)
    model=word2vec.Word2Vec(sentences,sg = 1,min_count=1,size=100,window=5)        # 训练模型
    # 保存模型
    model.save(save_model)
    # 读取模型
    #model = word2vec.Word2Vec.load('new_model.model')
    # 两词之间的相关性
    # y1=model.similarity(u"快递", u"物流")
    # y2=model.similarity(u"快递", u"美白")
    # print(y1)
    # print(y2)
    # 一个词的全部向量
    print(model['滋润'])
    # 一个词相关度最高的一些向量
    #for i in model.most_similar(u"快递"):
        # print (i[0],i[1])


class get_numpy():
    def __init__(self,vec_model,result_name):
        vec_model = word2vec.Word2Vec.load(vec_model)  # 读取词向量模型
        # 读取分词结果至texts
        f = open(result_name, encoding="utf-8")
        texts = []
        for i in f:
            i = i.rstrip("\n ")
            i = i.split(" ")
            texts.append(i)
        f.close()  # 文件的光标已经被放到最后了，所以需要重新打开一次
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]   # 词袋模型
        self.dictionary = dictionary
        self.corpus = corpus
        tfidf = models.TfidfModel.load(settings.tfidf_model)  # 读取tf-idf模型
        corpus_tfidf = tfidf[corpus]  # 计算评论的tf -idf 值
        similarity = similarities.Similarity('Similarity-tfidf-index', corpus_tfidf, num_features=600)  # 相似性检索
        self.tfidf_model = tfidf
        self.vec_model = vec_model
        self.result_name = result_name

    def ret_dictionary(self):
        return self.dictionary

    def ret_corpus(self):
        return self.dictionary

    def ret_tfidf_model(self):
        return self.tfidf_model

    def ret_vec_model(self):
        return self.vec_model
    '''         # 去除词袋中的低频词 ， 未实现
    loca = 0
    locb = 0
    for a in corpus:
        locb = 0
        for b in a:
            if b[1] < 5:
                #print( corpus[loca][locb])
                del corpus[loca][locb]
                print("loca = ",loca,"locb = ",locb,)
            locb += 1
        loca += 1
        '''

    def number_to_token(self,number):
        for i in self.ret_dictionary().token2id.items():
            if(i[1]==number):
                return i[0]
    #print(ret_dictionary().token2id)

    def select_by_tfidf(self,str0):
        checked_list = []
        str1 = str0.split(" ")
        str1 = self.ret_dictionary().doc2bow(str1)
        str_vec_tfidf = self.ret_tfidf_model()[str1]
        word_expect_length = 12 # 经计算平均评论为10.77词，此处取整为12词
        if (len(str_vec_tfidf) > word_expect_length):
            max_tf = []
            countvec = 0
            min_tf_loc = 0  # 前word_expect_length中最小权的位置
            min_tf = 0  # 前word_expect_length中最小权的权值
            for pp in str_vec_tfidf:
                if (len(max_tf) < word_expect_length):          # 新的列表max_tf未满
                    max_tf.append({str_vec_tfidf[countvec][0]: str_vec_tfidf[countvec][1]})
                else:  # 此时max_tf中已有word_expect_length个词的权重
                    min_tf_dict = max_tf[0]
                    for h in min_tf_dict.items():  # 取第一个词的权值
                        min_tf = h[1]
                        min_tf_loc = 0
                    for r in range(1, word_expect_length):  # 求前word_expecte_length个中的权值
                        min_tf_dict = max_tf[r]
                        for p in min_tf_dict.items():
                            if (p[1] < min_tf):
                                min_tf = p[1]
                                min_tf_loc = r
                    if ( pp[1] > min_tf ):
                        max_tf[min_tf_loc] = {str_vec_tfidf[countvec][0]: pp[1]}
                countvec += 1
            checked_vec = max_tf        # 此处checked_vec是被裁剪为12个权值最高的字典组成的列表
            for checked_counts in checked_vec:
                for vary in checked_counts.items():  # 此处checked_counts是筛选出的权值最大的字典键值对
                    checked_list.append(vary[0])
                '''
            if(len(checked_list)>word_expect_length):
                print("error0")
                print(checked_list)
                exit(0)
                '''
        else :                   # 长度小于12，不需要使用tf-idf计算权值
            for cal in str_vec_tfidf:
                checked_list.append(cal[0])
            while ( len(checked_list)< word_expect_length):
                checked_list.append(-1)
                '''
            if (len(checked_list) > word_expect_length):
                print("error1")
                print(checked_list)
                exit(0)
                '''
        return checked_list


    def search_vec(self,i):
        word = self.number_to_token(i)
        the_vec = self.vec_model[word]
        the_vec = list(the_vec)
        return the_vec

    def output(self,numpy_name):
        full_zero = []  # 100维填充矩阵
        for i in range(100):
            full_zero.append(0)
        f = open(self.result_name, encoding="utf-8")
        lstm_in = []          # 暂作为lstm的输入，评论*词*词向量
        lstm_in_in = []
        for line in f:
            line = line.rstrip("\n").replace("  ", " ")
            # 此处line形如'确实 不错 赞赞赞 '
            line = self.select_by_tfidf(line)
            # 此处line形如[65, 97, 242, 271, 588, 593, 1405, 11443, 12763, -1, -1, -1]，-1表示评论长度不足 word_expect_length
            for one_word_number in line:        # 每条评论是一个字典
                if(one_word_number >= 0):
                    lstm_in_in.append(self.search_vec(one_word_number))
                else:
                    lstm_in_in.append(full_zero)
            #print(lstm_in_in)
            lstm_in.append(lstm_in_in)
            lstm_in_in = []
        lstm_numpy = numpy.array(lstm_in)
   #     print(lstm_numpy.shape)
        numpy.save(numpy_name,lstm_numpy)
        f.close()


# 从分词结果txt读入，写入矩阵numpy.npy
def save_numpy(vec_model,result_name):
    example = get_numpy(vec_model,result_name)
    example.output()


def cal_vocab_size(vec_model,result_name):
    example = get_numpy(vec_model,result_name)
    return example.ret_corpus()


class LSTMTagger(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim ,hidden_dim, dropout= 0.3, bidirectional= True)
        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)    # 线性变换
                                    # 输入样本大小，输出样本大小
  #      self.testlstm = torch.nn.Linear(400,52)
        self.test_layer = torch.nn.Linear(200,52)
    def forward(self, sentence):
        sentence = sentence.float()
        # 求有效词数量，解决变长输入问题
        k = 0
        seq_lengths = []
        for one_word_vec in sentence:
            if(one_word_vec[0] == 0):
                seq_lengths.append(k)
                break
            k += 1
        if(len(seq_lengths) == 0):
            seq_lengths.append(12)
        '''
        sentence = sentence.unsqueeze(1)
      #  print(sentence.shape)
        output, (hn, cn) = self.testlstm(sentence)
        hn = hn.squeeze(0)
      #  hn = self.hidden2tag(hn)
        hn = self.test_layer(hn)
        return hn
    '''
        # 压缩
        seq_lengths = torch.from_numpy(numpy.array(seq_lengths))
        sentence = sentence.unsqueeze(1)

        pack = torch.nn.utils.rnn.pack_padded_sequence(sentence, seq_lengths)
        lstm_out , _ = self.lstm(pack)
        # 前者是最后一层 lstm 的每个词向量对应隐藏层的输出，pack_sequence格式
        # 后者 hn , cn是所有层最后一个隐藏元和记忆元的输出



        # 展开
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
        tag_scores = unpacked.data  # lstm每个词向量对隐藏层的输出·tensor格式
        lenth = unpacked_len.item() # 求长度


        output = _[0].squeeze(0)    # 取hn为句向量
     #   output = output.squeeze(1)

        output = torch.cat([output[0], output[-1]], dim = 1)
        output = output.squeeze(0)

        output = self.test_layer(output) # 全连接层，将向量调整为52维概率
     #   output = F.sigmoid(output)

        return output


def tfidf(fenci_result):
    f = open(fenci_result,encoding="utf-8")
    texts = []
    for i in f:
        i = i.rstrip("\n ")
        i = i.split(" ")
        texts.append(i )
    f.close()                                   # 文件的光标已经被放到最后了，所以需要重新打开一次
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]  # 词袋模型
    print(dictionary.token2id)
    #tfidf = models.TfidfModel(corpus)   # 词袋模型→tf-idf模型
    #tfidf.save(settings.tfidf_model)                    # 保存模型
   # tfidf = models.TfidfModel.load(settings.tfidf_model)  # 读取模型


def feature_extract():
    f = open(settings.fenci_result, encoding="utf-8")
    texts = []
    for i in f:
        i = i.rstrip("\n ")
        i = i.split(" ")
        texts.append(i)
    f.close()  # 文件的光标已经被放到最后了，所以需要重新打开一次

    dictionary = gensim.corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]  # 词袋模型
    tfidf = gensim.models.TfidfModel.load(settings.tfidf_model)  # 读取tf-idf模型
    corpus_tfidf = tfidf[corpus]  # 计算评论的tf -idf 值
    print(corpus_tfidf[2])
    output = []
    for i in corpus_tfidf:
        for p in i:
            if ((p[1] > 0.55) & (p[1] < 0.65)):
                if (dictionary[p[0]] not in output):
                    output.append(dictionary[p[0]])
                    print(dictionary[p[0]], p[1])
    '''
    import collections
    from sklearn.cluster import KMeans
    import pandas
    import numpy
    weight = []
    one_weight = []
    half_weight = []
    for i in corpus_tfidf:
        for k in i:
            half_weight.append(k[0])
            half_weight.append(k[1])
            one_weight.append(half_weight)
            half_weight = []
        weight.append(one_weight)
        one_weight = []
    weight = numpy.array(weight)
    weight = numpy.concatenate(weight, axis=0)
    # print(weight.shape)

    kmean = KMeans(n_clusters=10)   #
    kmean.fit(weight)
    label = kmean.labels_
    index1=list(range(len(weight)))
    vc = pandas.Series(label,index=index1)
    aa = collections.Counter(label)
    v = pandas.Series(aa)
    v1=v.sort_values(ascending=False)


    vc1 = vc[vc == v1.index[0]]
    vindex = list(vc1.index)

    kkp = pandas.Series(dictionary)

    print('前10个类')
    one_class = []
    count = 0
    for i in kkp:
        if(count < 10):
            one_class.append(dictionary[count])
            count += 1
        else:
            break
    print(one_class)
    '''


# 统计各标签的出现次数
# 无法区分各标签的输入，尝试减少标签数量，故此处改为选择最频繁出现的标签
def target_count():
    f = open(settings.useful_targets, encoding="utf-8")
    list = []
    new_list = []
    for i in f:
        m = i.rstrip(" \n").split(" ")
        for p in m:
            tag = 0
            for q in list:
                if (q[0] == p):
                    tag = 1     # 有

            if(tag == 1):
                for c in list:
                    if(c[0] == p):
                        c[1] += 1
            else:
                list.append([p, 1])

    while( len(list) != 0):
        max = list[0]
        for q in list:
            if(max[1] < q[1]):
                max = q
        list.remove(max)
        new_list.append(max)
    print(new_list)


# 结果评价
def accuracy():
    answer = open(settings.shengcheng_result, encoding= "utf-8")    # 生成的标签
    standard = open(settings.test_targets, encoding= "utf-8") # 标准标签

    accuracy_sum = 0    # 准确率/杰卡德
    precision_sum = 0   # 精准率
    recall_sum = 0      # 召回率
    total_amount = 0
    # Subset Accuracy（衡量正确率，预测的样本集和真实的样本集完全一样才算正确。）
    subset_accuracy = 0
    # Hamming Loss（衡量的是错分的标签比例，正确标签没有被预测以及错误标签被预测的标签占比）
    hamming_loss = 0
    for answer_line, standard_line in zip(answer, standard):
        total_amount += 1
        # 杰卡德系数
        ajb = []  # A∩B ,
        abb = []  # A∪B

        standard_list = standard_line.split()
        answer_list = answer_line.split()

        # 完全正确数计算
        subset_accuracy_flag = 1
        if(len(answer_list) != len(standard_list)):
            subset_accuracy_flag = 0
        for i in answer_list:
            if(i not in standard_list):
                subset_accuracy_flag = 0
        subset_accuracy += subset_accuracy_flag

        # 汉明损失计算
        hanming_count = 0
        for i in answer_list:
            if(i not in standard_list):
                hanming_count += 1
        for i in standard_list:
            if(i not in answer_list):
                hanming_count += 1
        hanming_count = hanming_count / 52 # 共52 个标签
        hamming_loss += hanming_count

        # 杰卡德系数(即准确率)计算
        if(len(answer_list) == 0):  # 没识别出来, 该条评论相似度为0
            continue
        for j in answer_list:
            if(j in standard_list):
                ajb.append(j)
        for b in answer_list:
            if(b not in abb):
                abb.append(b)
        for b in standard_list:
            if(b not in abb):
                abb.append(b)
        accuracy = len(ajb) / len(abb)
        accuracy_sum += accuracy

        # 精准率计算
        precision = len(ajb) / len(answer_list)
        precision_sum += precision

        # 召回率
        recall = len(ajb) / len(standard_list)
        recall_sum += recall


    if(total_amount == 0):
        accuracy = 0
        print("error")
        exit()
    else:
        subset_accuracy = subset_accuracy / total_amount    # 正确率
        hamming_loss = hamming_loss / total_amount          # 汉明损失
        accuracy = accuracy_sum / total_amount              # 准确率
        precision = precision_sum / total_amount            # 精确率
        recall = recall_sum / total_amount                  # 召回率
        F1 = 2 * (precision * recall ) / (precision + recall)   # F1值
   # print("subset_accuracy:", subset_accuracy)
   # print("汉明损失：", hamming_loss)
   # print("准确率:", accuracy)
    result = []
    result.append(F1)
    result.append(subset_accuracy)
    result.append(hamming_loss)
    result.append(precision)
    result.append(recall)
    answer.close()
    standard.close()

    return result

