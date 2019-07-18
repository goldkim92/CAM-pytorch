import os
import argparse

from solver import CLS_Trainer


# ===========================================================
# settings
# ===========================================================
parser = argparse.ArgumentParser(description='')

parser.add_argument('--gpu_number', type=str, default='0')
parser.add_argument('--epochs',     type=int, default=10)
parser.add_argument('--lr',         type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--phase',       type=str, default='train', help='"train" or "test" or "continue_train"') 

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_number


# ===========================================================
# main
# ===========================================================
if __name__ == "__main__":
    
    model = CLS_Trainer(args)
    
    if args.phase in ['train', 'continue_train']:
        print('\n===== Train VOC 2007 Classification Model =====')
        
        # Validate the initialized model
        model.test(0)
        
        # Train model
        for epoch in range(1, args.epochs+1):
            print(f'\nEPOCH: {epoch}')
            model.train(epoch)    
            model.test(epoch)


#         # Evaluate the final model
#         print('\n\n Evaluate normal {}-MIL'.format(args.method))
#         net.test()
#     
#     elif args.phase == 'test':
#         print('\n\n Evaluate normal {}-MIL'.format(args.method))
#         net.test()
        
    else:
        raise Exception("phase should be in ['train', 'test', 'continue_train']")
