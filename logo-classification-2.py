import mxnet as mx
import cv2
import os
from pathlib import Path
from os.path import basename, dirname
from time import time
import shutil
import matplotlib.pyplot as plt
import random
import configparser
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.data import DataLoader
from mxnet.gluon.utils import split_and_load
from mxnet.gluon.nn import Sequential, Conv2D, Dropout, MaxPool2D, Flatten, Dense


base_logos_dir = "./FlickrLogos_47/"
# train_logos_dir = base_logos_dir + "train"
# test_logos_dir = base_logos_dir + "test"


# number of labeled logo training images = 833
# number of nologo training images = 3000
train_logosonly_filename_list = base_logos_dir+"train/filelist-logosonly.txt"
train_both_filename_list = base_logos_dir+"train/filelist.txt"
test_filename_list = base_logos_dir+"test/filelist.txt"

train_dest_folder_name = 'train_data'
val_dest_folder_name = 'val_data'
test_dest_folder_name = 'test_data'

config_file = "./logodetectionconfig.ini"

className2ClassID_file_path = base_logos_dir + "className2ClassID.txt"

className2ClassID_list = []

image_label_dict={}

batch_size = -1
num_classes = -1
num_epochs = -1
num_cpu = -1
image_width = -1
image_height = -1
horizontal_flip = -1
num_workers = -1
num_fc = -1
dropout =-1
ctx = []

def load_file(file):
    with open(file) as f:
        filenames = f.read().splitlines()
    return filenames


def create_val_from_train_dataset(logosonly_filename, nonlogos_filename, val_nonlogo_total, val_logo_total):
    # get random 50% of the total indexes of val_nonlogos_filename
    index = random.sample(range(len(logosonly_filename)), val_logo_total)

    # get 50% of the val_nonlogos_filename elements chosen randomley above
    val_logos_filename = [logosonly_filename[i] for i in index]

    index = random.sample(range(len(nonlogos_filename)), val_nonlogo_total)

    val_nonlogos_filename = [nonlogos_filename[i] for i in index]

    # merge the above randombly chosen elements with the train_logos_filename
    train_filenames = list(set(logosonly_filename).difference(val_logos_filename)) + list (set(nonlogos_filename).difference(val_nonlogos_filename))

    val_filenames = val_logos_filename+ val_nonlogos_filename

    return (train_filenames, val_filenames)


def prepare_datasets(base_dir,folder, filenames,dest_folder_name):
    for filename in filenames:
        image_src_path = base_dir + folder +filename
        class_names = read_label_from_image_name(image_src_path)
        for name in class_names:
            image_dest_path = base_dir + dest_folder_name + "/" + name + "/"
            dest_dir_path = Path(os.path.dirname(image_dest_path))
            dest_dir_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_src_path, image_dest_path)


def read_config_values():
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_file)

    global batch_size, num_classes, num_epochs, num_cpu, num_workers, image_width, image_height, horizontal_flip, num_fc, ctx, dropout
    batch_size = int(config['MxnetConfigValues']['batch_size'])
    num_classes = int(config['MxnetConfigValues']['num_classes']) 
    num_epochs = int(config['MxnetConfigValues']['num_epochs']) 
    num_cpu = int(config['MxnetConfigValues']['num_cpu'])
    num_workers = int(config['MxnetConfigValues']['num_workers'])
    image_width = int(config['ImageConfig']['image_width'])
    image_height = int(config['ImageConfig']['image_height'])
    horizontal_flip = float(config['ImageConfig']['horizontal_flip'])
    num_fc = int(config['cnn']['num_fc'])
    dropout = float(config['cnn']['dropout'])
    ctx = [mx.cpu(i) for i in range(num_cpu)]
    #ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
    print("{}:{}:{}:{}:{}:{}:{}:{}:{}".format(batch_size, num_classes, num_epochs, num_cpu, image_width, image_height, horizontal_flip, dropout, num_fc))


def read_image_classname_mapping():
    global className2ClassID_list
    with open(className2ClassID_file_path) as file:
        lines=file.read().splitlines()
        className2ClassID_list=[""]*len(lines)
        for line in lines:
            l=line.split()
            className2ClassID_list[int(l[1])] = l[0]


def read_label_from_image_name(file_path):
    class_name = set()

    if basename(dirname(file_path)) == "no-logo":
        return class_name

    image_annotation_path = file_path[0:file_path.find(".png")]+".gt_data.txt"

    try:
        if Path(image_annotation_path).is_file():
            with open(image_annotation_path) as file:
                for line in file:
                    class_name.add(className2ClassID_list[int(line.split()[4])])
    except Exception as e:
        print("file doesn't exist:"+image_annotation_path)
        class_name.clear()

    return class_name


def create_folder_structure_for_training():
    folders=[train_dest_folder_name, test_dest_folder_name, val_dest_folder_name]

    for folder in folders:
        dest_dir_path = base_logos_dir + folder
        Path(dest_dir_path).mkdir(exist_ok=True)


def get_train_augs():
    train_augs = [
        mx.image.HorizontalFlipAug(horizontal_flip),
        # mx.image.CenterCropAug((image_width, image_height)),
        # mx.image.BrightnessJitterAug(.3), # randomly change the brightness
        # mx.image.HueJitterAug(.1)         # randomly change hue
        mx.image.RandomCropAug((image_width, image_height))
    ]

    return train_augs


def get_val_test_augs():
    val_test_augs=[
        mx.image.CenterCropAug((image_width, image_height))
    ]

    return val_test_augs


def transform(data, label, augs ):
    data = data.astype('float32')
    # print("image size:{}".format(data.shape))

    for aug in augs:
        data  = aug(data)

    data = mx.nd.transpose(data, (2,0,1))/255

    return data, mx.nd.array([label]).asscalar().astype('float32')


def perpare_data_loader(imgs, shuffle=False):
    # print(type(batch_size), type(num_workers))
    data = DataLoader(imgs, batch_size= batch_size, num_workers = num_workers, shuffle=shuffle)
    return data


def show_images(imgs, nrows, ncols, figsize=None):
    # print("inside show images:{}:{}".format(nrows, ncols))
    figsize = (ncols, nrows)
    _, figs = plt.subplots(nrows, ncols, figsize=figsize)
    for r in range(nrows):
        for c in range(ncols):
            figs[r][c].imshow(imgs[r*ncols+c].asnumpy())
            figs[r][c].axes.get_xaxis().set_visible(False)
            figs[r][c].axes.get_yaxis().set_visible(False)
    # print("showing images")
    plt.show()


def get_image_folder_dataset(folder_name, augs):

    imgs = ImageFolderDataset(
        folder_name,
        transform = lambda x, y: transform(x, y, augs)
    )

    return imgs

# do we really need this
def _get_batch(batch, ctx):
    data, label = batch
    return (split_and_load(data, ctx, even_split=False), split_and_load(label, ctx, even_split=False), data.shape[0])


def evaluate_accuracy(data_iter, net, ctx):
    acc = mx.nd.array([0])
    n = 0
    for batch in data_iter:
        data, label, batch_size = _get_batch(batch, ctx)
        for X, y in zip(data, label):
            output = net(X)
            predictions = output.argmax(axis=1)
            # print(predictions,y)
            acc += mx.nd.sum(net(X).argmax(axis=1)==y).copyto(mx.cpu())
            n += y.size
        acc.wait_to_read()
    return acc.asscalar()/n


# def evaluate_accuracy(data_iter, net, ctx):
#     acc = mx.metric.Accuracy()
#     for batch in data_iter:
#         data, label, batch_size = _get_batch(batch, ctx)
#         output = net(data)
#         predictions = output.argmax(axis=1)
#         acc.update(preds=predictions, labels=label)
#     return acc.get()[1]


def train(net, ctx, train_data, val_data, test_data, batch_size, num_epochs, model_prefix,
hybridize=False, learning_rate=0.01, wd=0.001):
    net.collect_params().reset_ctx(ctx)
    if hybridize:
        net.hybridize()
    softmax_cross_entropy = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = mx.gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': learning_rate, 'wd': wd
        })
    
    best_epoch = -1
    best_acc = 0.0
    
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    for epoch in range(num_epochs):
        train_loss = train_acc = n = 0.0
        # moving_loss = 0
        start = time()
        for idx, batch in enumerate(train_data):
            print(idx)
            data, label, temp_batch_size = _get_batch(batch, ctx)
            losses = []
            # t_loss = 0
            with mx.autograd.record():
                output = [net(X) for X in data]
                losses = [softmax_cross_entropy(ypred, y) for ypred, y in zip(output, label)]
            for loss in losses:    
                loss.backward()
                # t_loss += loss.sum().asscalar() 
            # print(t_loss)
            # train_loss += sum(t_loss)
            # print(train_loss)
            # print(losses)
            # t_loss = [l.sum().asscalar() for l in losses]
            # # print(t_loss)
            # trainer.step(temp_batch_size)
            # curr_loss = mx.nd.mean(mx.nd.array(t_loss)).asscalar()
            # # print(curr_loss)
            # moving_loss = (curr_loss if (idx==0 and epoch==0)
            #     else (1- learning_rate)*moving_loss + learning_rate*curr_loss)
            train_loss += sum([l.sum().asscalar() for l in losses])
            trainer.step(temp_batch_size)
            n += temp_batch_size

        train_acc = evaluate_accuracy(train_data, net, ctx)
        val_acc = evaluate_accuracy(val_data, net, ctx)
        test_acc = evaluate_accuracy(test_data, net, ctx)
        print("Epoch %s. Loss: %s, Train acc %s, Val acc %s, Test acc %s,\
            Time %s sec" %(epoch, train_loss/n, train_acc, val_acc, test_acc, time()-start))


def get_image(url):
    image_path = mx.test_utils.download(url)
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_width, image_height))
    plt.imshow(img)
    return image_path


def classify_logo(net, url, train_imgs):
    image_path = get_image(url)
    with open(image_path, 'rb') as f:
        img = mx.image.imdecode(f.read())
    data, _ = transform(img, -1, val_test_augs)
    data = data.expand_dims(axis=0)
    output = net(data.as_in_context(ctx[0]))
    output = mx.nd.SoftmaxActivation(output)
    print("predicting label:{}".format(output))
    pred = int(mx.nd.argmax(output, axis=1).asscalar())
    prob = output[0][pred].asscalar()
    label = train_imgs.synsets

    return "prob={} label={}".format(prob, label[pred])



if __name__=="__main__":
    train_logosonly_filename=load_file(train_logosonly_filename_list)
    train_both_filename=load_file(train_both_filename_list)
    # val_logos_filename=load_file(val_logos_filename_list)
    # val_nonlogos_filename=load_file(val_nonlogos_filename_list)
    test_filenames=load_file(test_filename_list)

    read_config_values()

    read_image_classname_mapping()

    create_folder_structure_for_training()

    train_nonlogos_filename = list(set(train_both_filename).difference(train_logosonly_filename))

    # move 1500 nologo and 233 logo images from train to validation 
    train_filenames, val_filenames = create_val_from_train_dataset(logosonly_filename=train_logosonly_filename, nonlogos_filename = train_nonlogos_filename, val_nonlogo_total = 1500, val_logo_total=233)

    # create required folder to keep train/validation/test images
    prepare_datasets(base_dir=base_logos_dir, folder="train/", filenames=train_filenames, dest_folder_name=train_dest_folder_name)
    # print("train_data done")
    
    prepare_datasets(base_dir=base_logos_dir, folder="train/", filenames=val_filenames, dest_folder_name=val_dest_folder_name)
    # print("val_data done")

    prepare_datasets(base_dir=base_logos_dir, folder="test/", filenames=test_filenames, dest_folder_name=test_dest_folder_name)
    # print("test_data done")

    train_augs = get_train_augs()
    # print("train augs done")

    val_test_augs = get_val_test_augs()
    # print("val_test augs done")
    train_imgs = get_image_folder_dataset(base_logos_dir + train_dest_folder_name, train_augs)
    # print("train imgs done")
    val_imgs = get_image_folder_dataset(base_logos_dir + val_dest_folder_name, val_test_augs)
    # print("val imgs done")
    test_imgs = get_image_folder_dataset(base_logos_dir + test_dest_folder_name, val_test_augs)
    # print("test augs done")
    train_data_loader = perpare_data_loader(train_imgs, shuffle=True)
    # print("train_data done")
    val_data_loader = perpare_data_loader(val_imgs)
    # print("val_data done")
    test_data_loader = perpare_data_loader(test_imgs)
    # print("test_data done")
    # print(len(train_data_loader))
    # for data_batch, label_batch in train_data_loader:
    #     # print("looping train_data")
    #     data_batch = data_batch.transpose((0,2,3,1)).clip(0, 255)/255
    #     show_images(data_batch,4,5)
        # break

    cnn_net = Sequential()

    with cnn_net.name_scope():
        cnn_net.add(Conv2D(channels=96, kernel_size=(3,3), strides=(1,1), activation='relu'))
        cnn_net.add(MaxPool2D(pool_size=3, strides=(2,2)))
        # cnn_net.add(Dropout(0.1))
        cnn_net.add(Conv2D(channels=192, kernel_size=(3,3), strides=(1,1), activation='relu'))
        cnn_net.add(MaxPool2D(pool_size=3, strides=(2,2)))
        # cnn_net.add(Dropout(0.1))
        cnn_net.add(Conv2D(channels=384, kernel_size=(3,3),  activation='relu'))
        # cnn_net.add(MaxPool2D(pool_size=3, strides=2))
        cnn_net.add(Conv2D(channels=384, kernel_size=(3,3),  activation='relu'))
        # cnn_net.add(Dropout(0.1))
        # cnn_net.add(MaxPool2D(pool_size=3, strides=2))
        # cnn_net.add(Conv2D(channels=256, kernel_size=3, activation='relu'))
        cnn_net.add(MaxPool2D(pool_size=3, strides=(2,2)))
        # cnn_net.add(Dropout(0.1))
        cnn_net.add(Flatten())
        cnn_net.add(Dense(num_fc, activation='relu'))
        cnn_net.add(Dropout(0.5))
        cnn_net.add(Dense(num_fc, activation='relu'))
        cnn_net.add(Dropout(0.5))
        cnn_net.add(Dense(num_classes))

    cnn_net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

    train(cnn_net, ctx, train_data_loader, val_data_loader, test_data_loader, batch_size, num_epochs, model_prefix='cnn')