#!/usr/bin/env bash
cd ..
python -mnnet.run.srl.conll --data data/conll2009/CoNLL2009-ST-English-development.txt > data/conll2009/dev_conll2009.json.dep
#python -mnnet.run.srl.conll --data data/conll2009/CoNLL2009-ST-evaluation-English.txt > data/conll2009/test_conll2009.json.dep
python -mnnet.run.srl.conll --data data/conll2009/CoNLL2009-ST-English-train.txt  > data/conll2009/train_conll2009.json.dep
#python -mnnet.run.srl.conll --data data/conll2009/CoNLL2009-ST-evaluation-English-ood.txt > data/conll2009/ood_conll2009.json.dep

python -mnnet.run.srl.training_sample_prune --data data/conll2009/train_conll2009.json.dep --frames data/nombank_descriptions-1.0+prop3.1.json > conll2009.train.dep_prune
#python -mnnet.run.srl.training_sample_2 --data data/conll2009/ood_conll2009.json.dep --frames data/nombank_descriptions-1.0+prop3.1.json > conll2009.ood.dep_l
#python -mnnet.run.srl.training_sample_2 --data data/conll2009/test_conll2009.json.dep --frames data/nombank_descriptions-1.0+prop3.1.json > conll2009.test.dep_l
python -mnnet.run.srl.training_sample_prune --data data/conll2009/dev_conll2009.json.dep --frames data/nombank_descriptions-1.0+prop3.1.json > conll2009.dev.dep_prune

#cat conll2009.ood conll200c9.test conll2009.train conll2009.dev conll2009.test > conll2009.combined
##cut -f6 conll2009.combiccned | python -mnnet.ml.voc --tokenizer space >frames.voc.conll2009
##cut -f2 conll2009.train | python -mnnet.ml.voc --tokenizer space >words.voc_unk.conll2009
##cut -f2 conll2009.combined | python -mnnet.ml.voc -f --tokenizer space >freq.voc_unk.conll2009
##cut -f10 conll2009.combined | python -mnnet.ml.voc --tokenizer space >labels.voc.conll2009
##cut -f2 conll2009.combined | python -mnnet.ml.voc --tokenizer space >words.voc.conll2009

#python -mnnet.run.srl.glove_select_srl words.voc.conll2009 data/sskip.100.vectors > word_embeddings_proper.sskip.conll2009.txt
#cut -f1 word_embeddings_proper.sskip.conll2009.txt > p.words.voc_sskip.conll2009


