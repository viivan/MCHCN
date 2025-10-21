from QRec import QRec
from util.config import ModelConf
import time

if __name__ == '__main__':
    print('='*80)
    print('   QRec: An effective python-based recommendation model library.   ')
    print('='*80)
    print(f'[Auto-selected Model] MCHCN (Multi-Channel Hypergraph Convolutional Network)')
    print(f'[Corresponding Paper] MCHCN.pdf')
    print('='*80)

    model_name = 'MCHCN'
    conf_path = './config/' + model_name + '.conf'  # ./config/MCHCN.conf

    # Load the configuration file and initialize QRec
    try:
        conf = ModelConf(conf_path)
        print(f'[Config Load Success] Path: {conf_path}')
    except Exception as e:
        print(f'[Config Load Failed] Error: {str(e)}')
        print(f'Please check if {conf_path} exists and format is correct.')
        exit(-1)

    # Execute the MCHCN model (training + evaluation)
    s = time.time()
    try:
        recSys = QRec(conf)
        recSys.execute()
        print(f'[Model Execute Success] MCHCN training and evaluation completed.')
    except Exception as e:
        print(f'[Model Execute Failed] Error: {str(e)}')
        exit(-1)

    # 4. 输出总运行时间
    e = time.time()
    total_time = e - s
    print('='*80)
    print(f'[Run Summary]')
    print(f'Model: {model_name}')
    print(f'Total Running Time: {total_time:.2f} s ({total_time/60:.2f} min)')
    print('='*80)