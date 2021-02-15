import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import argparse
from dataloader import DataLoader
from utils import progress_report, load_images, load_model
from sklearn.metrics import classification_report


def main():
    print("=== Loading dataset ===")
    categories = [f"class{i}" for i in range(args.num_class)]
    categories_dict = {c: i for i, c in enumerate(categories)}

    train, test = load_images(args.image_folder, categories_dict, args.test_rate,
                              args.train_length, args.test_length, args.seed)
    print(f"train: {len(train[0])}, test: {len(test[0])}")
    data_loader_train = data_utils.DataLoader(DataLoader(train, train=True),
                                              batch_size=args.batch_size, shuffle=True)
    data_loader_test = data_utils.DataLoader(DataLoader(test, train=False),
                                             batch_size=args.batch_size, shuffle=False)

    print("=== Start Training ===")
    model = load_model(args.arch, args.num_class).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(args.device)

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        pred_list = np.empty((0, args.num_class), np.float32)
        gt_list = np.empty(0, np.float32)
        start_time = time.time()

        for step, (images, labels) in enumerate(data_loader_train):
            images = images.requires_grad_().to(args.device)
            labels = labels.long().to(args.device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            pred_list = np.append(pred_list, np.array(F.softmax(preds.data.cpu(), dim=1)), axis=0)
            gt_list = np.append(gt_list, np.array(labels.data.cpu()), axis=0)

            progress_report(step, start_time, args.batch_size, len(train[0]))

        # evaluate, print
        train_acc = np.mean(np.argmax(pred_list, axis=1) == gt_list)
        print(f"\n train epoch: {epoch} acc: {train_acc}, loss: {train_loss}")
        print(classification_report(gt_list, np.argmax(pred_list, axis=1),
                                    target_names=categories))

        if (epoch + 1) % args.test_interval == 0:
            model.eval()
            test_loss = 0
            pred_list = np.empty((0, args.num_class), np.float32)
            gt_list = np.empty(0, np.float32)
            start_time = time.time()

            # evaluate network
            with torch.no_grad():
                for step, (data, labels) in enumerate(data_loader_test):
                    data = data.requires_grad_().to(args.device)
                    labels = labels.long().to(args.device)

                    preds = model(data)
                    test_loss += criterion(preds, labels).data
                    pred_list = np.append(pred_list, np.array(preds.data.cpu()), axis=0)
                    gt_list = np.append(gt_list, np.array(labels.data.cpu()), axis=0)

                    progress_report(step, start_time, args.batch_size, len(test[0]))

                # evaluate, print
                test_acc = np.mean(np.argmax(pred_list, axis=1) == gt_list)
                print(f"\n test epoch: {epoch} acc: {test_acc}, loss: {test_loss}")
                print(classification_report(gt_list, np.argmax(pred_list, axis=1),
                                            target_names=categories))

                # model save
                torch.save(model.state_dict(), f"{args.model_folder}/{args.arch}.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train louvre')
    # parser.add_argument("out")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", "-g", type=int, default=-1)
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0E-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_class", type=int, default=10)
    parser.add_argument("--test_rate", type=float, default=0.2)
    parser.add_argument("--test_interval", type=int, default=1)
    parser.add_argument("--train_length", type=int, default=-1)
    parser.add_argument("--test_length", type=int, default=-1)
    parser.add_argument('--arch', default='vgg', choices=['inception', 'squeezenet', 'vgg', 'resnet', 'mobilenet'])
    parser.add_argument("--image_folder", type=str, default='./img')
    parser.add_argument("--model_folder", type=str, default='./trained_models')
    args = parser.parse_args()

    # torch.manual_seed(args.seed)

    if torch.cuda.is_available() and args.gpu >= 0:
        args.device = torch.device(args.gpu)
    else:
        args.device = torch.device('cpu')

    main()

