# import the required library
import pandas as pd
import numpy as np
import argparse

# import the training script for two models
# from lte250_train import run as ltrain
from gt250_train import run as gtrain

def loadData(path):
    data = pd.read_csv(path)
    data = data.drop_duplicates("review")
    data["len"] = data.apply(lambda x : len(str(x["review"]).split()) , axis = 1)
    return data

def train(
    train_file_path,
    test_file_path,
    gt_use_path,
    gt_save_model_path,
    lte_save_model_path,
    lte_save_model_file,
    gt_use_dim = 512,
    gt_pad = 50,
    gt_test_split = .2, 
    gt_lstm_depth = 500, 
    gt_lstm_dropout = .25, 
    gt_recurrent_dropout = .25, 
    gt_dropout = .3,
    gt_input1_dense_layers = (256,128,64),
    gt_input2_dense_layers = (256,128,64),
    gt_dense_activation = "relu",
    gt_output_activation = "softmax",
    gt_learning_rate = 2e-5,
    gt_loss='categorical_crossentropy',
    
    
    gt_metrics=['accuracy'],
    gt_save_monitor='val_accuracy',
    gt_save_mode='max',
    gt_earlystopping_monitor='val_accuracy',
    gt_earlystopping_patience=5,
    
    gt_epochs = 25,
    gt_batch_size=128,
    lte_num_epochs = 1,
    lte_batch_size = 8
):
    
    train = loadData(train_file_path)
    test = loadData(test_file_path)
    
    train = train.dropna()
    test = test.dropna()
    
    trainlte = train[train.len <= 250].sample(1000)
    testlte = test[test.len <= 250].sample(1000)
    
    traingt = train[train.len > 250].sample(1000)
    testgt = test[test.len > 250].sample(1000)
    
    gtrain(list(traingt.review),
       list(traingt.summary),
       list(testgt.review),
       list(testgt.summary),
       traingt.sentiment.values, 
       testgt.sentiment.values,
       3 , gt_use_path,
       gt_save_model_path,
       gt_use_dim ,
        gt_pad ,
        gt_test_split , 
        gt_lstm_depth , 
        gt_lstm_dropout , 
        gt_recurrent_dropout , 
        gt_dropout ,
        gt_input1_dense_layers,
        gt_input2_dense_layers,
        gt_dense_activation,
        gt_output_activation,
        gt_learning_rate,
        gt_loss,
        gt_metrics,
        gt_save_monitor,
        gt_save_mode,
        gt_earlystopping_monitor,
        gt_earlystopping_patience,
        gt_epochs ,
        gt_batch_size,
      
      )
    
#     lr(
#        trainlte.review.values,
#        testlte.review.values,
#        trainlte.sentiment.values,
#        testlte.sentiment.values, 
#         3,lte_save_model_path,
#         lte_save_model_file,
#         lte_num_epochs,
#         lte_batch_size
#      )




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Sentiment Train Model')
    req_grp = parser.add_argument_group(title='Required Optional')
    req_grp.add_argument("--trainfile",  required=True, help="Enter the training data path")
    req_grp.add_argument("--testfile",  required=True, help="Enter the testing data path")
    req_grp.add_argument("--usefile",  required=True, help="Enter the Univeral Sentence Encoder Path")
    req_grp.add_argument("--gtsave",  required=True, help="Enter the path for saving LSTM model")
    req_grp.add_argument("--ltesave", required=True, help="Enter the path for saving XLNET model")
    req_grp.add_argument("--ltefile", required=True, help="Enter the file name for XLNET model")
    req_grp.add_argument("--gtusedim",  help="Enter the dimention of USE",type = int, default = 512)
    req_grp.add_argument("--gtpad", help="Enter the max padding for LSTM",type = int, default = 50)
    req_grp.add_argument("--validratio", help="Enter the validation ratio in data split",type = float, default = .2)
    req_grp.add_argument("--gtlstmdepth", help="Enter the number of Layers in LSTM", type = int, default = 500)
    req_grp.add_argument("--lstmdrop", help="Enter the lstm dropout rate", type = float, default = .25)
    req_grp.add_argument("--reccdrop", help="Enter the recurrent dropout rate", type = float, default = .25)
    req_grp.add_argument("--drop", help="Enter the dropout rate", type = float, default = .3)
    req_grp.add_argument("--gt1layers", nargs='+', type=int , help="Enter the depth of hidden layers1", default = (256,128,64) )
    req_grp.add_argument("--gt2layers", nargs='+', type=int , help="Enter the depth of hidden layers2 ", default = (256,128,64))
    req_grp.add_argument("--gtdenceact", help="Enter the activtion for dence layers", default = "relu")
    req_grp.add_argument("--gtoutact", help="Enter the activation for output layer", default = "softmax")
    req_grp.add_argument("--gtlr", help="Enter learning rate", default = 2e-5)
    req_grp.add_argument("--gtloss", help="Enter the loss function", default = "categorical_crossentropy")
    req_grp.add_argument("--metrics", nargs='+', help="Enter the metrics to monitor", default = ["accuracy"])
    req_grp.add_argument("--savemonitor", help="metric to monitor for model saving", default = "val_accuracy")
    req_grp.add_argument("--savemode", help="define the save mode", default = "max")
    req_grp.add_argument("--earlystopmonitor", help="metric to monitor for model saving", default = "val_accuracy")
    req_grp.add_argument("--earlystoppatience", help="patience to check for early stopping", type = int, default = 5)
    req_grp.add_argument("--gtepoch", help="Epochs for lstm model", type = int, default = 25)
    req_grp.add_argument("--gtbatch", help="batch size for lstm", type = int, default = 128)
    req_grp.add_argument("--lteepoch", help="Epochs for xlnet", type = float, default = 1)
    req_grp.add_argument("--ltebatch", help="Batch size of xlnet", type = float, default = 8)
    args = parser.parse_args()
    
    train(
        args.trainfile,
        args.testfile,
        args.usefile,
        args.gtsave,
        args.ltesave,
        args.ltefile,
        args.gtusedim,
        args.gtpad,
        args.validratio,
        args.gtlstmdepth,
        args.lstmdrop,
        args.reccdrop,
        args.drop,
        args.gt1layers,
        args.gt2layers,
        args.gtdenceact,
        args.gtoutact,
        args.gtlr,
        args.gtloss,
        args.metrics,
        args.savemonitor,
        args.savemode,
        args.earlystopmonitor,
        args.earlystoppatience,
        args.gtepoch,
        args.gtbatch,
        args.lteepoch,
        args.ltebatch,
        
    )
    