import torch
from torch import nn
import time
from sklearn.metrics import accuracy_score
import numpy as np
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def normal_lr_schedule(epoch):
    if epoch < 25:
        lr = 1e-4
    elif epoch < 40:
        lr = 2e-5
    else:
        lr = 1e-6
    return lr

def small_lr_schedule(epoch):
    if epoch < 50:
        lr = 1e-4
    elif epoch < 75:
        lr = 2e-5
    else:
        lr = 1e-6
    return lr

def large_lr_schedule(epoch):
    if epoch < 50:
        lr = 1e-1
    elif epoch < 75:
        lr = 1e-2
    else:
        lr = 1e-3
    return lr

def to_category(label_tensor=None, class_num=3):
    label_tensor = label_tensor.cpu().numpy()
    label_inter = np.zeros((label_tensor.size, class_num))
    for i in range(label_tensor.size):
        label_inter[i, int(label_tensor[i])] = 1
    return label_inter

def acc_scores(model=None, data_loader=None, gpu_device=0, out_mode='single', class_num = 2):
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    test_acc = []
    sum_label = torch.zeros(2).cuda(gpu_device)
    pre_label = torch.zeros(2).cuda(gpu_device)
    for test_img, test_label in data_loader:
        test_img = test_img.cuda(gpu_device)
        test_label = test_label.cuda(gpu_device)
        with torch.no_grad():
            if out_mode == 'triplet':
                _, _, test_pre_y = model(test_img)
            elif out_mode == 'single':
                test_pre_y = model(test_img)
            test_loss = loss_fn(test_pre_y, test_label)
            test_pre_label = torch.argmax(test_pre_y, dim=1)
            test_acc.append(accuracy_score(test_label.detach().cpu().numpy(),
                                           test_pre_label.detach().cpu().numpy()))
            pre_label = torch.cat((pre_label, test_pre_label))
            sum_label = torch.cat((sum_label, test_label))
    pre_label = pre_label[2:]
    sum_label = sum_label[2:]
    print('-----------------------------------------------------------------------')
    print(' test_acc:{:.4}'.format(np.mean(test_acc)))
    print('-----------------------------------------------------------------------')
    print('classification_report:', '\n', classification_report(sum_label.cpu().numpy(), pre_label.cpu().numpy(), digits=4))
    print('-----------------------------------------------------------------------')
    print('AUC:',roc_auc_score(to_category(sum_label, class_num = class_num), to_category(pre_label, class_num = class_num)))
    print('-----------------------------------------------------------------------')

def cls_model_fit(ddai_net=None, train_loader=None, val_loader=None, test_loader=None, lr_fn=None, epoch = 100,
                   gpu_device = 0):
    loss_fn = nn.CrossEntropyLoss()
    # ce_loss = nn.TripletMarginLoss()
    # rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=1e-6, weight_decay=0.0001)
    # scheduler = lr_scheduler.OneCycleLR(rmp_optim, max_lr=2e-5, epochs=500, steps_per_epoch=len(train_loader))
    # scheduler = lr_scheduler.StepLR(optimizer=rmp_optim, step_size=50, gamma=0.1)
    # torch.cuda.empty_cache()
    for i in range(epoch):
        start_time = time.time()
        if lr_fn == 'normal':
            rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=normal_lr_schedule(i))
        elif lr_fn == 'small':
            rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=small_lr_schedule(i))
        elif lr_fn == 'large':
            rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=large_lr_schedule(i))

        ddai_net.train()
        for img_data, img_label in train_loader:
            img_data = img_data.cuda(gpu_device)
            img_label = img_label.cuda(gpu_device)
            pre_y = ddai_net(img_data)
            loss_value = loss_fn(pre_y, img_label)
            # loss_value = loss_dense + loss_vit
            loss_value.backward()
            # scheduler.step()
            rmp_optim.step()
            rmp_optim.zero_grad()

        ddai_net.eval()
        train_acc = []
        for train_img, train_label in train_loader:
            train_img = train_img.cuda(gpu_device)
            train_label = train_label.cuda(gpu_device)
            with torch.no_grad():
                train_pre_y = ddai_net(train_img)
                train_pre_label = torch.argmax(train_pre_y, dim=1)
                train_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                                    train_pre_label.detach().cpu().numpy()))

        val_acc = []
        for val_img, val_label in val_loader:
            val_img = val_img.cuda(gpu_device)
            val_label = val_label.cuda(gpu_device)
            with torch.no_grad():
                val_pre_y = ddai_net(val_img)
                val_loss = loss_fn(val_pre_y, val_label)
                val_pre_label = torch.argmax(val_pre_y, dim=1)
                val_acc.append(accuracy_score(val_label.detach().cpu().numpy(),
                                                  val_pre_label.detach().cpu().numpy()))

        end_time = time.time()
        print('epoch ' + str(i + 1),
              ' Time:{:.3}'.format(end_time - start_time),
              ' train_loss:{:.4}'.format(loss_value.detach().cpu().numpy()),
              ' train_acc:{:.4}'.format(np.mean(train_acc)),
              ' val_loss:{:.4}'.format(val_loss.detach().cpu().numpy()),
              ' val_acc:{:.4}'.format(np.mean(val_acc)))

        # write_1.add_scalar('train_acc',np.mean(train_acc), global_step = i)
        # write_1.add_scalar('train_loss', loss_value.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_loss', val_loss.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_acc', np.mean(val_acc), global_step=i)
        weight_file = ddai_net.state_dict()
        torch.save(weight_file, 'F:\Object_Detection\Weights\Vgg_net.pt')

    ddai_net.eval()
    test_acc = []
    for test_img, test_label in test_loader:
        test_img = test_img.cuda(gpu_device)
        test_label = test_label.cuda(gpu_device)
        with torch.no_grad():
            test_pre_y = ddai_net(test_img)
            test_loss = loss_fn(test_pre_y, test_label)
            test_pre_label = torch.argmax(test_pre_y, dim=1)
            test_acc.append(accuracy_score(test_label.detach().cpu().numpy(),
                                               test_pre_label.detach().cpu().numpy()))

    print('train_acc:{:.4}'.format(np.mean(train_acc)),
          ' val_acc:{:.4}'.format(np.mean(val_acc)),
          ' test_acc:{:.4}'.format(np.mean(test_acc)))
    # write_1.add_graph(ddai_net, input_to_model=train_img[0:2])
    # g = ddai_net.state_dict()
    # torch.save(g, 'E:\DDAI-TCNet\Weights\Dense_SE.pth')

def regression_model_fit(ddai_net=None, train_loader=None, val_loader=None, test_loader=None, lr_fn=None, epoch = 100,
                   gpu_device = 0):
    loss_fn = nn.MSELoss()
    # ce_loss = nn.TripletMarginLoss()
    # rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=1e-6, weight_decay=0.0001)
    # scheduler = lr_scheduler.OneCycleLR(rmp_optim, max_lr=2e-5, epochs=500, steps_per_epoch=len(train_loader))
    # scheduler = lr_scheduler.StepLR(optimizer=rmp_optim, step_size=50, gamma=0.1)
    # torch.cuda.empty_cache()
    for i in range(epoch):
        start_time = time.time()
        if lr_fn == 'normal':
            rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=normal_lr_schedule(i))
        elif lr_fn == 'small':
            rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=small_lr_schedule(i))
        elif lr_fn == 'large':
            rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=large_lr_schedule(i))

        ddai_net.train()
        for img_data, img_label in train_loader:
            img_data = img_data.cuda(gpu_device)
            img_label = img_label.cuda(gpu_device)
            pre_y = ddai_net(img_data)
            loss_value = loss_fn(pre_y, img_label)
            # loss_value = loss_dense + loss_vit
            loss_value.backward()
            # scheduler.step()
            rmp_optim.step()
            rmp_optim.zero_grad()

        ddai_net.eval()
        train_loss = []
        for train_img, train_label in train_loader:
            train_img = train_img.cuda(gpu_device)
            train_label = train_label.cuda(gpu_device)
            with torch.no_grad():
                train_pre_y = ddai_net(train_img)
                train_loss.append(loss_fn(train_pre_y, train_label).cpu().detach().numpy())

        val_loss = []
        for val_img, val_label in val_loader:
            val_img = val_img.cuda(gpu_device)
            val_label = val_label.cuda(gpu_device)
            with torch.no_grad():
                val_pre_y = ddai_net(val_img)
                val_loss.append(loss_fn(val_pre_y, val_label).cpu().detach().numpy())

        end_time = time.time()
        print('epoch ' + str(i + 1),
              ' Time:{:.3}'.format(end_time - start_time),
              ' train_loss:{:.4}'.format(np.sum(np.abs(train_loss))),
              ' val_loss:{:.4}'.format(np.sum(np.abs(val_loss))))

        # write_1.add_scalar('train_acc',np.mean(train_acc), global_step = i)
        # write_1.add_scalar('train_loss', loss_value.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_loss', val_loss.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_acc', np.mean(val_acc), global_step=i)
        weight_file = ddai_net.state_dict()
        torch.save(weight_file, 'F:\Object_Detection\Weights\Regression_net.pt')

    ddai_net.eval()
    test_loss = []
    for test_img, test_label in test_loader:
        test_img = test_img.cuda(gpu_device)
        test_label = test_label.cuda(gpu_device)
        with torch.no_grad():
            test_pre_y = ddai_net(test_img)
            test_loss.append(loss_fn(test_pre_y, test_label).cpu().detach().numpy())

    print('train_loss:{:.4}'.format(np.sum(np.abs(train_loss))),
          ' val_loss:{:.4}'.format(np.sum(np.abs(val_loss))),
          ' test_loss:{:.4}'.format(np.sum(np.abs(test_loss))))
    # write_1.add_graph(ddai_net, input_to_model=train_img[0:2])
    # g = ddai_net.state_dict()
    # torch.save(g, 'E:\DDAI-TCNet\Weights\Dense_SE.pth')