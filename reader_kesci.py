# coding:utf-8
import pandas as pd
import random
import codecs
import re
import jieba
from utils import UNK, ModelType, TaskType, load_dic, sent2ids, logger, ModelType


series_dict = {}
with codecs.open('/home/kesci/work/series.dict', "r", "utf-8") as fr:
    for line in fr.readlines():
        line = line.strip()
        series_dict[line] = True
        if re.findall(r'^[a-z0-9]+$', line):
            continue
        jieba.add_word(line, 1000, 'series')

class Dataset(object):
    def __init__(self, train_path, test_path, source_dic_path, target_dic_path,
                 model_type):
        self.train_path = train_path
        self.test_path = test_path
        self.source_dic_path = source_dic_path
        self.target_dic_path = target_dic_path
        self.model_type = ModelType(model_type)

        self.source_dic = load_dic(self.source_dic_path)
        self.target_dic = load_dic(self.target_dic_path)

        _record_reader = {
            ModelType.CLASSIFICATION_MODE: self._read_classification_record,
            ModelType.REGRESSION_MODE: self._read_regression_record,
            ModelType.RANK_MODE: self._read_rank_record,
        }

        assert isinstance(model_type, ModelType)
        self.record_reader = _record_reader[model_type.mode]
        self.is_infer = False
        
        self.train_data_csv = "/home/kesci/input/qichedashi/train_set.csv"
        self.dev_data_csv = "/home/kesci/input/qichedashi/final_round_dev_set.csv"
        self.test_data_csv = "/home/kesci/input/qichedashi/final_round_test_set.csv"
        self.NEG = 3
        self.train_samples = 200000

    def train(self):
        '''
        Load trainset.
        '''
        logger.info("[reader] load trainset from %s" % self.train_data_csv)
        train_data = pd.read_csv(self.train_data_csv)
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        qa_pair = []
        icount = 0
        for problem, report in zip(train_data["Problem"], train_data["Report"]):
            qa_pair.append((problem, report, 1))
            random_data = train_data.sample(frac=0.1).reset_index(drop=True)
            for i in random_data[:self.NEG]["Report"]:
                qa_pair.append((problem, i, 0))
            icount += 1
            if icount > self.train_samples:
                break
        random.shuffle(qa_pair)
        
        for line_id, line in enumerate(qa_pair):
            q = " ".join([w for w in jieba.cut(line[0]) if w not in series_dict])
            a = " ".join([w for w in jieba.cut(line[1]) if w not in series_dict])
            line = q + "\t" + a + "\t" + str(line[2])
            yield self.record_reader(line)
        
        #with open(self.train_path) as f:
            # TODO:转换数据格式
        #    for line_id, line in enumerate(f):
        #        yield self.record_reader(line)

    def test(self):
        '''
        Load testset.
        '''
        test_data = pd.read_csv(self.dev_data_csv)
        qa_pair = []
        for problem, report in zip(test_data["Problem"], test_data["Report"]):
            qa_pair.append((problem, report, 1))
            random_data = test_data.sample(frac=0.1).reset_index(drop=True)
            for i in random_data[:self.NEG]["Report"]:
                qa_pair.append((problem, i, 0))
        random.shuffle(qa_pair)
        
        for line_id, line in enumerate(qa_pair):
            q = " ".join([w for w in jieba.cut(line[0]) if w not in series_dict])
            a = " ".join([w for w in jieba.cut(line[1]) if w not in series_dict])
            line = q + "\t" + a + "\t" + str(line[2])
            if len(line.split("\t")) != 3:
                continue
            yield self.record_reader(line)
        #with open(self.test_path) as f:
        #    for line_id, line in enumerate(f):
        #        yield self.record_reader(line)

    def infer(self):
        self.is_infer = True
        test_data = pd.read_csv(self.test_data_csv)
        qa_pair = []
        for problem, report in zip(test_data["Problem"], test_data["Report"]):
            qa_pair.append((problem, report, 1))
            random_data = test_data.sample(frac=0.1).reset_index(drop=True)
            for i in random_data[:self.NEG]["Report"]:
                qa_pair.append((problem, i, 0))
        random.shuffle(qa_pair)
        
        for line_id, line in enumerate(qa_pair):
            q = " ".join([w for w in jieba.cut(line[0]) if w not in series_dict])
            a = " ".join([w for w in jieba.cut(line[1]) if w not in series_dict])
            line = q + "\t" + a + "\t" + str(line[2])
            yield self.record_reader(line)
        #with open(self.train_path) as f:
        #    for line in f:
        #        yield self.record_reader(line)

    def _read_classification_record(self, line):
        '''
        data format:
            <source words> [TAB] <target words> [TAB] <label>
        @line: str
            a string line which represent a record.
        '''
        fs = line.strip().split('\t')
        assert len(fs) == 3, "wrong format for classification\n" + \
            "the format shoud be " +\
            "<source words> [TAB] <target words> [TAB] <label>'"
        source = sent2ids(fs[0], self.source_dic)
        target = sent2ids(fs[1], self.target_dic)
        if not self.is_infer:
            label = int(fs[2])
            return (
                source,
                target,
                label, )
        return source, target

    def _read_regression_record(self, line):
        '''
        data format:
            <source words> [TAB] <target words> [TAB] <label>
        @line: str
            a string line which represent a record.
        '''
        fs = line.strip().split('\t')
        assert len(fs) == 3, "wrong format for regression\n" + \
            "the format shoud be " +\
            "<source words> [TAB] <target words> [TAB] <label>'"
        source = sent2ids(fs[0], self.source_dic)
        target = sent2ids(fs[1], self.target_dic)
        if not self.is_infer:
            label = float(fs[2])
            return (
                source,
                target,
                [label], )
        return source, target

    def _read_rank_record(self, line):
        '''
        data format:
            <source words> [TAB] <left_target words> [TAB] <right_target words> [TAB] <label>
        '''
        fs = line.strip().split('\t')
        assert len(fs) == 4, "wrong format for rank\n" + \
            "the format should be " +\
            "<source words> [TAB] <left_target words> [TAB] <right_target words> [TAB] <label>"

        source = sent2ids(fs[0], self.source_dic)
        left_target = sent2ids(fs[1], self.target_dic)
        right_target = sent2ids(fs[2], self.target_dic)
        if not self.is_infer:
            label = int(fs[3])
            return (source, left_target, right_target, label)
        return source, left_target, right_target


if __name__ == '__main__':
    path = './data/classification/train.txt'
    test_path = './data/classification/test.txt'
    source_dic = './data/vocab.txt'
    dataset = Dataset(path, test_path, source_dic, source_dic,
                      ModelType.CLASSIFICATION)

    for rcd in dataset.train():
        print rcd