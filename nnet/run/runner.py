import argparse
import sys
from nnet.util import *
from nnet.training import *
from nnet.testing import *
from nnet.corpus import *


class Runner(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description="neural networks trainer")

        parser.add_argument("--debug", help="print debug info",
                            action="store_true", default=False)

        parser.add_argument("--test", help="validation set")
        parser.add_argument("--dev", help="vadation set2")
        parser.add_argument("--train", help="training set", required=False)
        parser.add_argument("--batch", help="batch size", default=128, type=int)
        parser.add_argument("--epochs", help="n of epochs",
                            default=sys.maxsize, type=int)
        #parser.add_argument("--seed", help="manual set seed", default=1, type=int)
        #parser.add_argument("-optimizer", default="adadelta")
        parser.add_argument("--out", help="output dir", default="out")
        #parser.add_argument("--dump-ratio", default=200000, type=int)
        #parser.add_argument("log-level", default="ERROR")
        parser.add_argument("--finetune", help="pretrained model path")
        #parser.add_argument("--early-stop", action="store_true")
        parser.add_argument("--dbg-print-rate", help="in BATCHES", type=int,
                            default=5000)
        parser.add_argument("--test-only", action="store_true", required=False)
        parser.add_argument("--test-model", required=False)
        parser.add_argument("--params-path", help="pretrained model path", required=False)
        self.add_special_args(parser)
        self.a = parser.parse_args()

        # you just choose training or test
        if not (self.a.train or self.a.test_only):
            parser.error("either specify --train or --test-only")

        if self.a.test_only and not self.a.test:
            parser.error('specify --test')

    def get_reader(self):
        raise NotImplemented()

    def get_tester(self):
        raise NotImplemented()

    def get_parser(self):
        raise NotImplemented()

    def get_converter(self):
        raise NotImplemented()

    def load_model(self):
        raise NotImplemented()


    def add_special_args(self, parser):
        raise NotImplemented()

    def run(self):
        a = self.a

        if a.finetune:
            log('init model from' + a.finetune)
            model = torch.load(a.finetune)
        else:
            #model = self.load_model()
            model = self.load_model()
            log("a new model has been built")
            if a.test_model:
                params_path = a.params_path
                model.load_state_dict(torch.load(params_path))
                log('parameters loaded from ' + params_path)

        log('loading corpus from %s' % a.train)

        train_set = Corpus(
            parser=self.get_parser(),
            batch_size=a.batch,
            path=a.train,
            #return in run.py, just the simple_reader() in corpus.py
            reader=self.get_reader()
        )

        dev_set = Corpus(
            parser=self.get_parser(),
            batch_size=a.batch,
            path=a.dev,
            # return in run.py, just the simple_reader() in corpus.py
            reader=self.get_reader()
        )

        test_set = Corpus(
            parser=self.get_parser(),
            batch_size=a.batch,
            path=a.test,
            reader=self.get_reader()
        )

        log('dataset loaded!')

        if not a.test_model:
            train(
                model=model,
                train_set=train_set,
                dev_set=test_set,
                test_set=test_set,
                epochs=a.epochs,
                converter=self.get_converter(),
                dbg_print_rate=a.dbg_print_rate,
                params_path=a.params_path
            )
        else:
            test(
                model=model,
                train_set=train_set,
                test_set=test_set,
                converter=self.get_converter(),
                params_path=a.params_path
            )
