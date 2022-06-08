import os
from options.test_options import TestOptions
from data.create_data_loader import CreateDataLoader
from model.models import create_model
from util.visualizer import Visualizer
from util import html
from util.metrics import PSNR, SSIM
import lpips
from DISTS_pytorch import DISTS

perceptual_eval = True
if perceptual_eval:
    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
    # loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
    dists = DISTS()

opt = TestOptions().parse()
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(dataset)
print('#test images = %d' % dataset_size)
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

eLPIPS = 0.0
eDISTS = 0.0
ePSNR = 0.0
eSSIM = 0.0

for i, data in enumerate(dataset):
    model.set_input(data)
    model.test()
    eval_data = model.get_eval_data()
    visuals = model.get_current_visuals()

    x = eval_data['x']
    y = eval_data['y']
    output = eval_data['x_hat']

    img_path = model.get_image_paths()
    print('process image... %s' % img_path)

    if perceptual_eval:
        tensors = model.get_tensor_raw_data()
        # for DISTS [0, 1]
        tensor_data = model.get_tensor_raw_data()
        x_tensor = tensor_data['x']
        y_tensor = tensor_data['y']
        output_tensor = tensor_data['x_hat']
        eDISTS += dists(x_tensor, output_tensor)

        # for LIPIPS [-1, 1]
        x_tensor = x_tensor * 2.0 - 1
        y_tensor = y_tensor * 2.0 - 1
        output_tensor = output_tensor * 2.0 - 1
        eLPIPS += loss_fn_alex(x_tensor, output_tensor)

    ePSNR += PSNR(x, output)
    eSSIM += SSIM(x, output)
    # visualizer.display_current_results(visuals, 1)
    visualizer.save_images(webpage, visuals, img_path)

avgPSNR = ePSNR/dataset_size
avgSSIM = eSSIM/dataset_size
print('avgPSNR : %f' % avgPSNR)
print('avgSSIM : %f' % avgSSIM)
if perceptual_eval:
    avgLPIPS = eLPIPS/dataset_size
    avgDISTS = eDISTS/dataset_size
    print('avgLPIPS : %f' % avgLPIPS)
    print('avgDISTS : %f' % avgDISTS)
webpage.save()
