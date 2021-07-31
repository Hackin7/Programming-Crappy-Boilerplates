import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.utils.data
import torchvision

from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision import transforms
from torchvision.ops import batched_nms
from torchvision.transforms import functional as F

'''
%%shell

# Download TorchVision repo to use some files from references/detection
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.3.0

cp references/detection/utils.py ../
cp references/detection/coco_eval.py ../
'''

import utils
from coco_eval import CocoEvaluator

# check if cuda GPU is available, make sure you're using GPU runtime on Google Colab
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device) # you should output "cuda"

### Object Detection Dataset ################################
# You should have uploaded the training dataset onto your google drive.

base_folder = '/content/drive/MyDrive/TIL2021' # Change this path to your folder for this competition
training_path = os.path.join(base_folder, "c1_release.zip") # zip file for training dataset
'''
# unzip the training dataset
!unzip $training_path
'''

### Defining dataset ######################################
class TILDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat2name = {cat['id']:cat['name'] for cat in cats} # maps category id to category name (useful for visualization)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index] # Image ID
        ann_ids = coco.getAnnIds(imgIds=img_id) # get annotation id from coco
        coco_annotation = coco.loadAnns(ann_ids) # target coco_annotation file for an image
        path = coco.loadImgs(img_id)[0]['file_name'] # path for input image
        img = Image.open(os.path.join(self.root, 'images', path)).convert('RGB') # open the input image

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Labels
        labels = []
        for i in range(num_objs):
            labels.append(coco_annotation[i]['category_id'])
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Tensorise img_id
        img_id = torch.tensor([img_id])

        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

til_root = "/content/c1_release/" # extracted training dataset path
train_annotation = os.path.join(til_root, "train.json")
val_annotation = os.path.join(til_root, "val.json")

dataset = TILDataset(til_root, train_annotation)
dataset[30]

### Drawing 1 Example
source_img, img_annots = dataset[30]
draw = ImageDraw.Draw(source_img)
for i in range(len(img_annots["boxes"])):
    x1, y1, x2, y2 = img_annots["boxes"][i]
    label = int(img_annots["labels"][i])

    draw.rectangle(((x1, y1), (x2, y2)), outline="red")
    text = f'{dataset.cat2name[label]}'
    draw.text((x1+5, y1+5), text)
display(source_img)

# hyper-parameters
params = {'BATCH_SIZE': 8,
          'LR': 0.005,
          'CLASSES': 6,
          'MAXEPOCHS': 5,
          'BACKBONE': 'resnet50',
          'FPN': True,
          'ANCHOR_SIZE': ((32,), (64,), (128,), (256,), (512,)),
          'ASPECT_RATIOS': ((0.5, 1.0, 2.0),),
          'MIN_SIZE': 512,
          'MAX_SIZE': 512,
          'IMG_MEAN': [0.485, 0.456, 0.406],
          'IMG_STD': [0.229, 0.224, 0.225],
          'IOU_THRESHOLD': 0.5
          }

def get_resnet_backbone(backbone_name: str):
    """
    Returns a resnet backbone pretrained on ImageNet.
    Removes the average-pooling layer and the linear layer at the end.
    """
    if backbone_name == 'resnet18':
        pretrained_model = models.resnet18(pretrained=True, progress=False)
        out_channels = 512
    elif backbone_name == 'resnet34':
        pretrained_model = models.resnet34(pretrained=True, progress=False)
        out_channels = 512
    elif backbone_name == 'resnet50':
        pretrained_model = models.resnet50(pretrained=True, progress=False)
        out_channels = 2048
    elif backbone_name == 'resnet101':
        pretrained_model = models.resnet101(pretrained=True, progress=False)
        out_channels = 2048
    elif backbone_name == 'resnet152':
        pretrained_model = models.resnet152(pretrained=True, progress=False)
        out_channels = 2048

    backbone = torch.nn.Sequential(*list(pretrained_model.children())[:-2])
    backbone.out_channels = out_channels

    return backbone


def get_resnet_fpn_backbone(backbone_name: str):
    """
    Returns a specified ResNet backbone with FPN pretrained on ImageNet.
    """
    return resnet_fpn_backbone(backbone_name, pretrained=True, trainable_layers=3)

def get_anchor_generator(anchor_size: Tuple[tuple] = None, aspect_ratios: Tuple[tuple] = None):
    """Returns the anchor generator."""
    if anchor_size is None:
        anchor_size = ((16,), (32,), (64,), (128,))
    if aspect_ratios is None:
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_size)

    anchor_generator = AnchorGenerator(sizes=anchor_size,
                                       aspect_ratios=aspect_ratios)
    return anchor_generator

def get_roi_pool(featmap_names: List[str] = None, output_size: int = 7, sampling_ratio: int = 2):
    """Returns the ROI Pooling"""
    if featmap_names is None:
        # default for resnet with FPN
        featmap_names = ['0', '1', '2', '3']

    roi_pooler = MultiScaleRoIAlign(featmap_names=featmap_names,
                                    output_size=output_size,
                                    sampling_ratio=sampling_ratio)

    return roi_pooler

def get_fasterRCNN(backbone: torch.nn.Module,
                   anchor_generator: AnchorGenerator,
                   roi_pooler: MultiScaleRoIAlign,
                   num_classes: int,
                   image_mean: List[float] = [0.485, 0.456, 0.406],
                   image_std: List[float] = [0.229, 0.224, 0.225],
                   min_size: int = 512,
                   max_size: int = 1024,
                   **kwargs
                   ):
    """Returns the Faster-RCNN model. Default normalization: ImageNet"""
    model = FasterRCNN(backbone=backbone,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler,
                       num_classes=num_classes,
                       image_mean=image_mean,  # ImageNet
                       image_std=image_std,  # ImageNet
                       min_size=min_size,
                       max_size=max_size,
                       **kwargs
                       )
    model.num_classes = num_classes
    model.image_mean = image_mean
    model.image_std = image_std
    model.min_size = min_size
    model.max_size = max_size

    return model

def get_fasterRCNN_resnet(num_classes: int,
                          backbone_name: str,
                          anchor_size: List[float],
                          aspect_ratios: List[float],
                          fpn: bool = True,
                          min_size: int = 512,
                          max_size: int = 1024,
                          **kwargs
                          ):
    """Returns the Faster-RCNN model with resnet backbone with and without fpn."""

    # Backbone
    if fpn:
        backbone = get_resnet_fpn_backbone(backbone_name=backbone_name)
    else:
        backbone = get_resnet_backbone(backbone_name=backbone_name)

    # Anchors
    anchor_size = anchor_size
    aspect_ratios = aspect_ratios * len(anchor_size)
    anchor_generator = get_anchor_generator(anchor_size=anchor_size, aspect_ratios=aspect_ratios)

    # ROI Pool
    with torch.no_grad():
        backbone.eval()
        random_input = torch.rand(size=(1, 3, 512, 512))
        features = backbone(random_input)

    if isinstance(features, torch.Tensor):
        from collections import OrderedDict

        features = OrderedDict([('0', features)])

    featmap_names = [key for key in features.keys() if key.isnumeric()]

    roi_pool = get_roi_pool(featmap_names=featmap_names)

    # Model
    return get_fasterRCNN(backbone=backbone,
                          anchor_generator=anchor_generator,
                          roi_pooler=roi_pool,
                          num_classes=num_classes,
                          min_size=min_size,
                          max_size=max_size,
                          **kwargs)

model = get_fasterRCNN_resnet(num_classes=params['CLASSES'],
                              backbone_name=params['BACKBONE'],
                              anchor_size=params['ANCHOR_SIZE'],
                              aspect_ratios=params['ASPECT_RATIOS'],
                              fpn=params['FPN'],
                              min_size=params['MIN_SIZE'],
                              max_size=params['MAX_SIZE'],
                              image_mean=params['IMG_MEAN'],
                              image_std=params['IMG_STD'])


# load pretrained weights for FasterRCNN ResNet50 FPN
pretrained_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth', progress=True)
model_dict = model.state_dict()

# filter out roi_heads.box_predictor weights
pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('roi_heads.box_predictor')}
# overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# load the new state dict
model.load_state_dict(model_dict)

# move model to the right device
model.to(device)

# construct an optimizer
model_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(model_params,
                            lr=params['LR'],
                            momentum=0.9,
                            weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

### Data Augmentation #####################################
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

# converts the image, a PIL image, into a PyTorch Tensor
class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

# randomly horizontal flip the images and ground-truth labels
class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class RandomTranslation(object):
    def __init__(self, r):
        self.shift = r

    def __call__(self, image, target):
        if random.random() > 0.5:
            shiftX = random.randint(0, self.shift)
        else:
            shiftX = -random.randint(0, self.shift)

        if random.random() > 0.5:
            shiftY = random.randint(0, self.shift)
        else:
            shiftY = -random.randint(0, self.shift)
        height, width = image.shape[-2:]

        bbox = target["boxes"]
        bbox += torch.tensor([[shiftX, shiftY, shiftX, shiftY]])
        bbox = torch.clamp(bbox, torch.tensor([[0,0,0,0]]), torch.tensor([[width, height, width, height]]))
        bbox = bbox.type(torch.IntTensor)
        if bbox[0][2] -bbox[0][0] <= 0 or bbox[0][1] - bbox[0][3] <= 0:
            return image, target

        image = torch.roll(image, shiftX, 2)
        image = torch.roll(image, shiftY, 1)

        target["boxes"] = bbox
        return image, target

# https://blog.paperspace.com/data-augmentation-for-object-detection-building-input-pipelines/
class Resize(object):
    def letterbox_image(img, inp_dim):
        '''resize image with unchanged aspect ratio using padding'''
        img_w, img_h = img.size[0], img.size[1]
        w, h = inp_dim
        new_w = int(img_w * min(w/img_w, h/img_h))
        new_h = int(img_h * min(w/img_w, h/img_h))

        resized_image = np.array(img.resize((new_w,new_h)))

        #create a black canvas
        canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

        #paste the image on the canvas
        canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image

        return  Image.fromarray(canvas, 'RGB')
    def __call__(self, image, target):
        h, w = image.size[-2:]
        bbox = target["boxes"]
        inp_dim = (random.randint(1,w), random.randint(1,h))

        image = Resize.letterbox_image(image, inp_dim)


        scale = min(inp_dim[1]/h, inp_dim[0]/w)
        bbox[:,:4] *= (scale)

        new_w = scale*w
        new_h = scale*h
        del_h = (inp_dim[1] - new_h)/2
        del_w = (inp_dim[0] - new_w)/2

        add_matrix = torch.tensor([[del_w, del_h, del_w, del_h]])

        bbox[:,:4] += add_matrix

        target["boxes"] = bbox
        return image, target

# Testing Functionality
class ToPIL(object):
    def __call__(self, image, target):
        image = F.to_pil_image(image)
        return image, target

# https://pytorch.org/vision/master/_modules/torchvision/transforms/functional.html
def get_view_transforms():
    transforms = Compose([
            ToTensor(),
            RandomHorizontalFlip(0.5),
            RandomTranslation(200),
            ToPIL(),
            #Resize(),
        ])
    return transforms
view_dataset = TILDataset(til_root, train_annotation, get_view_transforms())

source_img, img_annots = view_dataset[30]
draw = ImageDraw.Draw(source_img)
for i in range(len(img_annots["boxes"])):
    x1, y1, x2, y2 = img_annots["boxes"][i]
    label = int(img_annots["labels"][i])

    draw.rectangle(((x1, y1), (x2, y2)), outline="red")
    text = f'{view_dataset.cat2name[label]}'
    draw.text((x1+5, y1+5), text)
display(source_img)


def get_transform(train):
    if train:
        transforms = Compose([
            ToTensor(),
            RandomHorizontalFlip(0.5),
            RandomTranslation(50),
        ])
    else: # during evaluation, no augmentations will be done
        transforms = Compose([
            ToTensor()
        ])

    return transforms


### Data Loaders ################################################
NUM_WORKERS = 4

# use our dataset and defined transformations
train_dataset = TILDataset(til_root, train_annotation, get_transform(train=True))
val_dataset = TILDataset(til_root, val_annotation, get_transform(train=False))

# define training and validation data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=params['BATCH_SIZE'], shuffle=True, num_workers=NUM_WORKERS,
    collate_fn=utils.collate_fn)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
    collate_fn=utils.collate_fn)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = data_loader.dataset.coco
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

 # load model weights (if not using the current trained model)
save_path = os.path.join(base_folder, "c1_weights.pth")
model.load_state_dict(torch.load(save_path, map_location=device))
model.to(device)
model.eval()


for epoch in range(1):#params['MAXEPOCHS']):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    train_one_epoch(model, optimizer, val_loader, device, epoch, print_freq=10)

    # update the learning rate
    lr_scheduler.step()

    # evaluate on the test dataset
    evaluate(model, val_loader, device=device)
    save_path = os.path.join(base_folder, "c1_weights.pth")
    torch.save(model.state_dict(), save_path)

### Visualization of results ################################
# pick one image from the validation set
img, _ = val_dataset[391]

model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])

# convert the image, which has been rescaled to 0-1 and had the channels flipped
pred_img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
draw = ImageDraw.Draw(pred_img)

img_preds = prediction[0]
for i in range(len(img_preds["boxes"])):
    x1, y1, x2, y2 = img_preds["boxes"][i]
    label = int(img_preds["labels"][i])
    score = float(img_preds["scores"][i])

    draw.rectangle(((x1, y1), (x2, y2)), outline="red")
    text = f'{dataset.cat2name[label]}: {score}'
    draw.text((x1+5, y1+5), text)

display(pred_img)

### Post Processing #################################################
img_preds = prediction[0]
keep_idx = batched_nms(boxes=img_preds["boxes"], scores=img_preds["scores"], idxs=img_preds["labels"], iou_threshold=params['IOU_THRESHOLD'])

# convert the image, which has been rescaled to 0-1 and had the channels flipped
pred_img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
draw = ImageDraw.Draw(pred_img)

for i in range(len(img_preds["boxes"])):
    if i in keep_idx:
        x1, y1, x2, y2 = img_preds["boxes"][i]
        label = int(img_preds["labels"][i])
        score = float(img_preds["scores"][i])

        draw.rectangle(((x1, y1), (x2, y2)), outline="red")
        text = f'{dataset.cat2name[label]}: {score}'
        draw.text((x1+5, y1+5), text)

display(pred_img)

det_threshold = 0.7

# convert the image, which has been rescaled to 0-1 and had the channels flipped
pred_img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
draw = ImageDraw.Draw(pred_img)

for i in range(len(img_preds["boxes"])):
    if i in keep_idx:
        x1, y1, x2, y2 = img_preds["boxes"][i]
        label = int(img_preds["labels"][i])
        score = float(img_preds["scores"][i])

        # filter out non-confident detections
        if score > det_threshold:
            draw.rectangle(((x1, y1), (x2, y2)), outline="red")
            text = f'{dataset.cat2name[label]}: {score}'
            draw.text((x1+5, y1+5), text)

display(pred_img)

# save model weights
save_path = os.path.join(base_folder, "c1_weights.pth")
torch.save(model.state_dict(), save_path)


### Evaluation on Validation Set ##################################################
with open(val_annotation) as json_file:
    val_data = json.load(json_file)

model.eval()
detections = []
with torch.no_grad():
    for image in val_data["images"]:
        img_name = image["file_name"]
        img_id = image["id"]

        img = Image.open(os.path.join(til_root, 'images', img_name)).convert('RGB')
        img_tensor = transforms.ToTensor()(img)

        preds = model([img_tensor.to(device)])[0]

        for i in range(len(preds["boxes"])):
            x1, y1, x2, y2 = preds["boxes"][i]
            label = int(preds["labels"][i])
            score = float(preds["scores"][i])

            left = int(x1)
            top = int(y1)
            width = int(x2 - x1)
            height = int(y2 - y1)

            detections.append({'image_id':img_id, 'category_id':label, 'bbox':[left, top, width, height], 'score':score})

validation_json = os.path.join(base_folder, "validation_preds.json")
with open(validation_json, 'w') as f:
    json.dump(detections, f)
# Get evaluation score against validation set to make sure your prediction json file is in the correct format
coco_gt = COCO(val_annotation)
coco_dt = coco_gt.loadRes(validation_json)
cocoEval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType='bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

### Generate Predictions on Test Images #################################
# You should already have uploaded the testing images onto your google drive.

testing_images_path = os.path.join(base_folder, "c1_test_release.zip") # change this path to the zip file for the test images
!unzip $testing_images_path

til_test_root = "/content/c1_test_release/" # extracted testing images path
test_img_root = os.path.join(til_test_root, "images")
img_dir = os.scandir(test_img_root)

# load model weights (if not using the current trained model)
model.load_state_dict(torch.load(save_path, map_location=device))
model.to(device)
model.eval()

img = Image.open(next(img_dir).path).convert('RGB')
draw = ImageDraw.Draw(img)
det_threshold = 0.5

# do the prediction
with torch.no_grad():
    img_tensor = transforms.ToTensor()(img)
    img_preds = model([img_tensor.to(device)])[0]

for i in range(len(img_preds["boxes"])):
    x1, y1, x2, y2 = img_preds["boxes"][i]
    label = int(img_preds["labels"][i])
    score = float(img_preds["scores"][i])

    # filter out non-confident detections
    if score > det_threshold:
        draw.rectangle(((x1, y1), (x2, y2)), outline="red")
        text = f'{dataset.cat2name[label]}: {score}'
        draw.text((x1+5, y1+5), text)

display(img)

### Submission of results
# generate detections on the folder of test images (this will be used for submission)
detections = []
with torch.no_grad():
    for image in img_dir:
        img_id = int(image.name.split('.')[0])

        img = Image.open(image.path).convert('RGB')
        img_tensor = transforms.ToTensor()(img)

        preds = model([img_tensor.to(device)])[0]

        for i in range(len(preds["boxes"])):
            x1, y1, x2, y2 = preds["boxes"][i]
            label = int(preds["labels"][i])
            score = float(preds["scores"][i])

            left = int(x1)
            top = int(y1)
            width = int(x2 - x1)
            height = int(y2 - y1)

            detections.append({'image_id':img_id, 'category_id':label, 'bbox':[left, top, width, height], 'score':score})


test_pred_json = os.path.join(base_folder, "test_preds.json")
with open(test_pred_json, 'w') as f:
    json.dump(detections, f)
