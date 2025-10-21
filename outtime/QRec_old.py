from util.config import OptionConf
from util.dataSplit import *
from multiprocessing import Process, Manager
from util.io import FileIO
from time import strftime, localtime, time
import mkl


class QRec(object):
    def __init__(self, config):
        self.trainingData = []  # training data
        self.testData = []  # testData
        self.relation = []
        self.measure = []
        self.config = config
        self.ratingConfig = OptionConf(config['ratings.setup'])

        # 数据加载逻辑（保持原有不变）
        if self.config.contains('evaluation.setup'):
            self.evaluation = OptionConf(config['evaluation.setup'])
            binarized = False
            bottom = 0
            if self.evaluation.contains('-b'):
                binarized = True
                bottom = float(self.evaluation['-b'])
            if self.evaluation.contains('-testSet'):
                self.trainingData = FileIO.loadDataSet(config, config['ratings'], binarized=binarized, threshold=bottom)
                self.testData = FileIO.loadDataSet(config, self.evaluation['-testSet'], bTest=True, binarized=binarized,
                                                   threshold=bottom)
            elif self.evaluation.contains('-ap'):
                self.trainingData = FileIO.loadDataSet(config, config['ratings'], binarized=binarized, threshold=bottom)
                self.trainingData, self.testData = DataSplit. \
                    dataSplit(self.trainingData, test_ratio=float(self.evaluation['-ap']), binarized=binarized)
            elif self.evaluation.contains('-cv'):
                self.trainingData = FileIO.loadDataSet(config, config['ratings'], binarized=binarized, threshold=bottom)
            elif self.evaluation.contains('-predict'):
                self.trainingData = FileIO.loadDataSet(config, config['ratings'], binarized=binarized, threshold=bottom)
                self.testData = FileIO.loadUserList(self.evaluation['-predict'])
        else:
            print('Wrong configuration of evaluation!')
            exit(-1)

        # 模体加载（MCHCN需标签数据，此处保留社交关系以兼容父类，无影响）
        if config.contains('social'):
            self.socialConfig = OptionConf(self.config['social.setup'])
            self.relation = FileIO.loadRelationship(config, self.config['social'])
        print('Reading data and preprocessing...')

    def execute(self):
        # 适配MCHCN的模型导入路径
        model_name = self.config['model.name']
        try:
            # 1. 优先尝试排序模型路径（MCHCN为API推荐模型，属于排序任务）
            importStr = f'from model.ranking.{model_name} import {model_name}'
            exec(importStr)
        except ImportError:
            # 2. 若排序路径不存在
            importStr = f'from model.rating.{model_name} import {model_name}'
            exec(importStr)
        except Exception as e:
            # 报错提示：MCHCN需放在指定路径
            print(f"Import {model_name} failed! Please put {model_name}.py in 'model/ranking/' directory.")
            print(f"Error details: {str(e)}")
            exit(-1)

        # 关键修改2：MCHCN构造函数需4个参数（config, train, test, relation, fold），补充fold默认值
        if self.evaluation.contains('-cv'):
            k = int(self.evaluation['-cv'])
            if k < 2 or k > 10:
                print("k for cross-validation should not be greater than 10 or less than 2")
                exit(-1)
            mkl.set_num_threads(max(1, mkl.get_max_threads() // k))
            manager = Manager()
            mDict = manager.dict()
            i = 1
            tasks = []
            binarized = False
            if self.evaluation.contains('-b'):
                binarized = True
            for train, test in DataSplit.crossValidation(self.trainingData, k, binarized=binarized):
                fold = '[' + str(i) + ']'
                # MCHCN继承SocialRecommender，需传入relation；即使无社交数据，relation为空列表不影响
                recommender = f"{model_name}(self.config, train, test, self.relation, fold)"
                p = Process(target=run, args=(mDict, eval(recommender), i))
                tasks.append(p)
                i += 1
            for p in tasks:
                p.start()
                if not self.evaluation.contains('-p'):
                    p.join()
            if self.evaluation.contains('-p'):
                for p in tasks:
                    p.join()
            # 交叉验证结果处理（保持原有不变）
            self.measure = [dict(mDict)[i] for i in range(1, k + 1)]
            res = []
            for i in range(len(self.measure[0])):
                if self.measure[0][i][:3] == 'Top':
                    res.append(self.measure[0][i])
                    continue
                measure = self.measure[0][i].split(':')[0]
                total = 0
                for j in range(k):
                    total += float(self.measure[j][i].split(':')[1])
                res.append(measure + ':' + str(total / k) + '\n')
            currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
            outDir = OptionConf(self.config['output.setup'])['-dir']
            fileName = f"{model_name}@${currentTime}-${str(k)}-fold-cv.txt"
            FileIO.writeFile(outDir, fileName, res)
            print(f'The result of {k}-fold cross validation:\n{"".join(res)}')

        else:
            # 非交叉验证模式：MCHCN需传入relation（空列表不影响）
            recommender = f"{model_name}(self.config, self.trainingData, self.testData, self.relation)"
            eval(recommender).execute()


def run(measure, algor, order):
    # 保持原有逻辑：执行模型并返回评估结果
    measure[order] = algor.execute()