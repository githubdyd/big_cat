import functions
import torch
import numpy
import settings

import torch.utils.data as Data
import torch.autograd.variable as V
from goto import *

EMBEDDING_DIM = 100
HIDDEN_DIM = 200
word_expect_length = 12

learning_rate = 0.005
poch_number = 400
vocab_size = len(functions.cal_vocab_size(settings.vec_model,settings.fenci_result))  # 词表大小


target_dict = functions.make_target_list()
print(target_dict)


if __name__ == "__main__":
    # 预处理
    '''
    functions.select_valid()        # 选取有效数据
    functions.treat_test_comment()  # 去除上述数据中笼统评论，并分为评论和标签保存
    functions.select_sentences(settings.useful_comment,settings.fenci_result)    # 分词
    functions.cal_vec(settings.fenci_result,settings.vec_model)                  # 计算词向量(根据分词结果)
    # tfidf(settings.fenci_result)    # 生成tf-idf模型
    functions.select_sentences(settings.test_comments,settings.test_result)    # 分词

    train = functions.get_numpy(settings.vec_model,settings.fenci_result)       # 训练用词向量
    train.output(settings.train_comment_numpy)      
    test = functions.get_numpy(settings.vec_model,settings.test_result)         # 测试用词向量
    test.output(settings.test_comment_numpy)
    functions.target_tensor(settings.useful_targets, settings.train_targets_numpy)  # 
    functions.target_tensor(settings.test_targets, settings.test_targets_numpy)
    '''

    test = functions.get_numpy(settings.vec_model, settings.test_result)
    test.output(settings.test_comment_numpy)
    test_comment = torch.from_numpy(numpy.load(settings.test_comment_numpy))
    # numpy数据读入
    test_comment = torch.from_numpy(numpy.load(settings.test_comment_numpy))
    train_comment = torch.from_numpy(numpy.load(settings.train_comment_numpy)).float()
    train_targets = torch.from_numpy(numpy.load(settings.train_targets_numpy)).float()
    functions.target_tensor(settings.test_targets, settings.test_comment_numpy)
    test_targets = torch.from_numpy(numpy.load(settings.test_targets_numpy))
  #  print("train_comment:",train_comment.shape)
   # print("train_targets:",train_targets.shape)
    print("test_comment:",test_comment.shape)
  #  print("test_targets:",test_targets.shape)
    train_comment.requires_grad = True
    train_targets.requires_grad = True

    # torch_dataset 和 torch_tagset 分别为长度相等的评论集和标签集

  #  model = torch.load(settings.trained_model)
   # print("模型已保存，学习率:",learning_rate,"epoch:",poch_number)
  #  model.eval()


    model = functions.LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, vocab_size,len(target_dict))
    loss_function = torch.nn.BCELoss()
    #loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

    '''
    for i in range (10):
        # 训练前
        with torch.no_grad():
            inputs = test_comment[i]

            #print(inputs)
            # tag_scores为训练前每个词对应的向量
            tag_scores = model(inputs)
            tag_scores = torch.sigmoid(tag_scores)
            print("before1:", tag_scores)
    model.zero_grad()


    lstm = torch.nn.LSTM(100,52)
    for i in range(10):
        inputs = test_comment[i]
        inputs = inputs.unsqueeze(1).float()
        output , _ = lstm(inputs)
        tag_scores = _[0]
    #    tag_scores =torch.sigmoid(tag_scores)
        print(tag_scores)
        '''

    flag = 0
    if(flag == 1):
        # 训练
        for epoch in range(poch_number):
            print(epoch)
            step = 0
            for tag_scores, packet_targets in zip(train_comment, train_targets):

                model.zero_grad()
                tag_scores = torch.autograd.Variable(tag_scores)
                tag_scores = model(tag_scores)
                tag_scores = torch.sigmoid(tag_scores)
                tag_scores = tag_scores.double()
                tag_scores = tag_scores.unsqueeze(0)
                packet_targets = torch.autograd.Variable(packet_targets.unsqueeze(0).double())

                tag_scores = tag_scores.squeeze(0)   # bceloss
                packet_targets = packet_targets
              #  print("tag_scores:",tag_scores)
              # print("packet_targets:",packet_targets)
              #  print(tag_scores)
                loss = loss_function(tag_scores, packet_targets)

                loss.backward()

               # optimizer.zero_grad()
                optimizer.step()
                step += 1

                if(step == 16000):   # 训练后
                    break
            print("loss:", loss)
        torch.save(model, settings.trained_model)  # 保存训练参数


    model = torch.load(settings.trained_model)
  #  print("模型已保存，学习率:",learning_rate,"epoch:",poch_number)
    model.eval()

    with torch.no_grad():
        n = 1325         # 测试评论数量
        save_loc = open(settings.shengcheng_result,"w+",encoding="utf-8")
        t = 0
        max_accuracy = 0
        max_accuracy_index = 0
        for dtyz in range(1,100): # 动态阈值
           # save_loc.truncate(0) # 清空输出结果
            x = 0.26
            for amount in range(n):
                inputs = test_comment[amount]
                # tag_scores为训练前每个词对应的向量
             #   print("inputs:",inputs)
                tag_scores = model(inputs)
                tag_scores = torch.sigmoid(tag_scores)
            #    print("after:",tag_scores)
                k = 0
                for i in tag_scores.squeeze(0):
                    if(i > x):
                        for m in target_dict:
                            if(target_dict[m] == k):
                          #      print(m + " ",end="")
                                save_loc.write(m + " ")
                    k += 1
                save_loc.write("\n")
            t += 1

            save_loc.close()
         #   print(x,functions.accuracy()[0])
            if(functions.accuracy()[0] > max_accuracy):
                max_accuracy = functions.accuracy()
                max_accuracy_index = x
           # exit()
            save_loc = open(settings.shengcheng_result, "w+", encoding="utf-8")
            break   # 阈值已确定时用break跳出

        print("最大准确率:", max_accuracy[0])
        print("此时阈值:", max_accuracy_index)

                          # 根据阈值进行标签分类
        save_loc.close()

        save_loc = open(settings.shengcheng_result,"w+",encoding="utf-8")
        for amount in range(n):
            inputs = test_comment[amount]
            # tag_scores为训练前每个词对应的向量
            tag_scores = model(inputs)
            tag_scores = torch.sigmoid(tag_scores)
            k = 0
            for i in tag_scores.squeeze(0):
                if(i > max_accuracy_index):
                    for m in target_dict:
                        if(target_dict[m] == k):
        #                    print(m + " ",end="")
                            save_loc.write(m + " ")
                k += 1
            save_loc.write("\n")
        save_loc.close()

    # 计算其它指标
    # One-error（度量的是：“预测到的最相关的标签” 不在 “真实标签”中的样本占比。值越小，表现越好）
    one_error_amount = 0
    file = open(settings.test_targets, "r", encoding="utf-8")

    for amount, one_standard_line in zip(range(n), file):
        inputs = test_comment[amount]
        tag_scores = model(inputs)
        tag_scores = torch.sigmoid(tag_scores)
        k = 0
        max_chance = 0
        max_chance_index = 0
        for i in tag_scores:
            if (i > max_chance):
                max_chance = i
                max_chance_index = k
            k += 1
        most_likely = ""  # 可能性最大的词
        # 遍历形如{'显色度': 22, '光泽效果': 18, '色号丰富': 39}的字典时，i为前面的字符串
        # 调用后面的索引号为target_dict[i]
        for i in target_dict:
            if (max_chance_index == target_dict[i]):
                most_likely = i
        if (most_likely not in one_standard_line):
            one_error_amount += 1
    one_error_amount = one_error_amount / n
    print("one-error值", one_error_amount)
    file.close()

    # 基于标签的评价指标
    # 宏平均
    macro_p = 0
    macro_r = 0
    # 微平均
    tp_sum = 0  # 微平均精确率和召回率都要用
    tp_fp = 0  # 计算微平均精确率
    tp_fn = 0  # 计算微平均召回率
    target_list = []
    for i in target_dict:
        t = i.rstrip("'").lstrip("'")
        target_list.append([t])  # 前者为精确率，后者为召回率

    for i in target_list:  # 对每一个标签进行计算    ，i为['包装外观新奇', 0, 0]...
        answer = open(settings.shengcheng_result, "r", encoding="utf-8")  # 生成的标签
        standard = open(settings.test_targets, "r", encoding="utf-8")  # 标准标签
        tp = 0  # 查到且正确
        fp = 0  # 查到但不正确
        tn = 0  # 没查到且没有
        fn = 0  # 没查到但有
        target = i[0]
        # print(target)

        for each1, each2 in zip(answer, standard):
            if ((target in each1) and (target in each2)):
                tp += 1
                continue
            if ((target in each1) and (target not in each2)):
                fp += 1
                continue
            if ((target not in each1) and (target not in each2)):
                tn += 1
                continue
            if ((target not in each1) and (target in each2)):
                fn += 1
        #   print(tp,fp,tn,fn)
        # 计算宏平均
        if (tp + fp == 0):  # 样本过小，没有出现该标签，暂时用0代替， 扩大测试集后不会出现这种情况
            p = 0
        else:
            p = tp / (tp + fp)  # 该标签i的精确率
        if (tp + fn == 0):
            r = 0  # 同上
        else:
            r = tp / (tp + fn)  # 该标签i的召回率
        macro_p += p
        macro_r += r
        # 计算微平均
        tp_sum += tp
        tp_fn += tp
        tp_fn += fn
        tp_fp += tp
        tp_fp += fp
        answer.close()
        standard.close()
    macro_p = macro_p / 52
    macro_r = macro_r / 52
    micro_p = tp_sum / tp_fp
    micro_r = tp_sum / tp_fn
    print("宏平均", " 精确率:", macro_p, " 召回率:", macro_r)
    print("微平均", " 精确率:", micro_p, " 召回率:", micro_r)
    print("正确率：", functions.accuracy()[1])
    print("汉明损失：", functions.accuracy()[2])
    print("精确率：", functions.accuracy()[3])
    print("召回率：", functions.accuracy()[4])
