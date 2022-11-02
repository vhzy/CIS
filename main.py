#!/usr/bin/env python
import argparse
import sys
# import pdb

# torchlight
import torchlight
from torchlight import import_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict()
    # processors['demo-stgcn'] = import_class('processor.demo-stgcn.DemoOffline')

    # processors['show-info'] = import_class('processor.show-info.ShowInfo')
    # processors['show-node'] = import_class('processor.show-node.ShowNode')
    # processors['visualize-auf'] = import_class('processor.visualize-auf.ShowInfo')
    # processors['cal-landmark'] = import_class('processor.cal-landmark.ShowInfo')
    # processors['save-result'] = import_class('processor.save-result.ShowInfo')
    # processors['train-stgcn'] = import_class('processor.train-stgcn.REC_Processor')
    # processors['train-image'] = import_class('processor.train-image.REC_Processor')
    # processors['test-image'] = import_class('processor.test-image.REC_Processor')
    # processors['test-image-contrast'] = import_class('processor.test-image-contrast.REC_Processor')
    # processors['quant-image'] = import_class('processor.quant-image.REC_Processor')
    processors['train-image-causal'] = import_class('processor.train-image-causal.REC_Processor')
    # processors['train-image-causal-woP'] = import_class('processor.train-image-causal-woP.REC_Processor')
    # processors['train-image-contrast'] = import_class('processor.train-image-contrast.REC_Processor')
    # processors['train-image-contrast-proto-sep'] = import_class('processor.train-image-contrast-proto-sep.REC_Processor')
    # processors['train-image-contrast-proto'] = import_class('processor.train-image-contrast-proto.REC_Processor')
    # processors['train-image-causal-detach'] = import_class('processor.train-image-causal-detach.REC_Processor')
    # processors['train-image-causal-fix'] = import_class('processor.train-image-causal-fix.REC_Processor')
    # processors['train-image-intensity'] = import_class('processor.train-image-intensity.REC_Processor')
    # processors['train-fsnet'] = import_class('processor.train-fsnet.REC_Processor')
    # processors['train-fsnet-nofix'] = import_class('processor.train-fsnet-nofix.REC_Processor')
    # processors['train-flowcycle'] = import_class('processor.train-flowcycle.REC_Processor')
    # processors['test-twostream'] = import_class('processor.test-twostream.REC_Processor')
    # processors['quant-twostream'] = import_class('processor.quant-twostream.REC_Processor')
    # processors['train-keypoint'] = import_class('processor.train-keypoint.REC_Processor')
    # processors['train-twostream'] = import_class('processor.train-twostream.REC_Processor')
    # processors['train-twostream-causal'] = import_class('processor.train-twostream-causal.REC_Processor')
    # processors['train-twostream-causal-fix'] = import_class('processor.train-twostream-causal-fix.REC_Processor')
    # processors['train-twostream-intensity'] = import_class('processor.train-twostream-intensity.REC_Processor')
    # processors['train-wrinkle'] = import_class('processor.train-wrinkle.REC_Processor')
    #endregion yapf: enable

    # add sub-parser
    # 在之后向其中添加子parser
    # 这个subparser只是一个句柄，可以向其中添加多个子parser.
    # 其中dest 属性是可以让uer在之后能够获得到subparser的当前子parser的名字，相当于别名。
    #dest指定的值用作key值，从解析后的对象中取出用户输入的第一个参数
    #所以上述的parser拥有一个arg.processor的属性。而这个属性对应了cmd中第一个输入值，也等于sys.argv[1],这里就是train-image-casual
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()
    
    # start
    #实例化REC_Processor类并传入命令行中参数（同时完成类初始化）
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:]) #第2个之后的参数是我们传入的参数，这里是yaml配置文件
    # pdb.set_trace()
    #调用REC_Processor类中开始函数
    p.start()
