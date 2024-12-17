# prerequisites
import torch
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision import utils
from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
import matplotlib.pyplot as plt

# defining the classifiers
clf_ss = svm.SVC(C=10, gamma='scale', kernel='rbf', probability= True)  # define the classifier for shape
clf_sc = svm.SVC(C=2, gamma='scale', kernel='rbf')  # classify shape map against color labels
clf_cc = svm.SVC(C=2, gamma='scale', kernel='rbf')  # define the classifier for color
clf_cs = svm.SVC(C=10, gamma='scale', kernel='rbf')  # classify color map against shape labels

vals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

#training the shape map on shape labels and color labels
def classifier_shape_train(vae, whichdecode_use, train_dataset):
    vae.eval()
    device = next(vae.parameters()).device
    with torch.no_grad():
        data, labels = next(iter(train_dataset))
        data = data[1]
        train_shapelabels=labels[0].clone()
        train_colorlabels=labels[1].clone()
        print(train_shapelabels[0:10])
        utils.save_image(data[0:10],'train_sample.png')

        data = data.to(device)
        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode_use)
        z_shape = vae.sampling(mu_shape, log_var_shape).to(device)
        print('training shape bottleneck against color labels sc')
        clf_sc.fit(z_shape.cpu().numpy(), train_colorlabels.cpu())

        print('training shape bottleneck against shape labels ss')
        clf_ss.fit(z_shape.cpu().numpy(), train_shapelabels)

#testing the shape classifier (one image at a time)
def classifier_shape_test(vae, whichdecode_use, clf_ss, clf_sc, test_dataset, confusion_mat=0):
    vae.eval()
    device = next(vae.parameters()).device
    with torch.no_grad():
        data, labels = next(iter(test_dataset))
        data=data[1]
        test_shapelabels=labels[0].clone()
        test_colorlabels=labels[1].clone()
        data = data.cuda()
        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode_use)
        z_shape = vae.sampling(mu_shape, log_var_shape).to(device)
        pred_ss = clf_ss.predict(z_shape.cpu())
        pred_sc = clf_sc.predict(z_shape.cpu())

        test_shapelabels = test_shapelabels.cpu().numpy()

        SSreport = accuracy_score(pred_ss,test_shapelabels)#torch.eq(test_shapelabels.cpu(), pred_ss).sum().float() / len(pred_ss)
        SCreport = accuracy_score(pred_sc,test_colorlabels.cpu().numpy())#torch.eq(test_colorlabels.cpu(), pred_sc).sum().float() / len(pred_sc)

        if confusion_mat == 1:
            cm = confusion_matrix(test_shapelabels, pred_ss)
            # Plot the confusion matrix
            plt.imshow(cm, cmap="Greys")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.colorbar()
            for i in range(0,36):
                plt.annotate(f'{vals[i]}', (i,i), fontsize=10)
            plt.show()

    return pred_ss, pred_sc, SSreport, SCreport

#training the color map on shape and color labels
def classifier_color_train(vae, whichdecode_use, train_dataset):
    vae.eval()
    device = next(vae.parameters()).device
    with torch.no_grad():
        data, labels = next(iter(train_dataset))
        data = data[1]
        train_shapelabels=labels[0].clone()
        train_colorlabels=labels[1].clone()
        data = data.cuda()

        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode_use)
        z_color = vae.sampling(mu_color, log_var_color).to(device)
        print('training color bottleneck against color labels cc')
        clf_cc.fit(z_color.cpu().numpy(), train_colorlabels)

        print('training color bottleneck against shape labels cs')
        clf_cs.fit(z_color.cpu().numpy(), train_shapelabels)

#testing the color classifier (one image at a time)
def classifier_color_test(vae, whichdecode_use, clf_cc, clf_cs, test_dataset, verbose=0):
    vae.eval()
    device = next(vae.parameters()).device
    with torch.no_grad():
        data, labels = next(iter(test_dataset))
        data=data[1]
        test_shapelabels=labels[0].clone()
        test_colorlabels=labels[1].clone()
        data = data.cuda()
        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode_use)

        z_color = vae.sampling(mu_color, log_var_color).to(device)
        pred_cc = torch.tensor(clf_cc.predict(z_color.cpu()))
        pred_cs = torch.tensor(clf_cs.predict(z_color.cpu()))

        CCreport = accuracy_score(pred_cc,test_colorlabels.cpu().numpy()) #torch.eq(test_colorlabels.cpu(), pred_cc).sum().float() / len(pred_cc)
        CSreport = accuracy_score(pred_cs,test_shapelabels.cpu().numpy()) #torch.eq(test_shapelabels.cpu(), pred_cs).sum().float() / len(pred_cs)

        if verbose==1:
            print('----**********-------color classification from color map')
            print(confusion_matrix(test_colorlabels, pred_cc))
            print(classification_report(test_colorlabels, pred_cc))

            print('----**********------shape classification from color map')
            print(confusion_matrix(test_shapelabels, pred_cs))
            print(classification_report(test_shapelabels, pred_cs))

    return pred_cc, pred_cs, CCreport, CSreport