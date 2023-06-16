import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader
from IPython.core.display import clear_output


def plot_learning_curves(history, metrics_names, label="", create_plot=True):
    if create_plot:
        fig = plt.figure(figsize=(30, 10))
    # Лосс
    plt.subplot(1, 3, 1)
    plt.title("Лосс")
    plt.plot(history["loss"]["train"], label="train" + label, lw=3, color="royalblue")
    plt.plot(history["loss"]["val"], label="val" + label, lw=3, color="orchid")
    plt.xlabel("Эпоха")
    plt.legend()

    # Первая метрика
    plt.subplot(1, 3, 2)
    plt.title(metrics_names[0])
    plt.plot(
        history[metrics_names[0]]["train"],
        label="train" + label,
        lw=3,
        color="royalblue",
    )
    plt.plot(
        history[metrics_names[0]]["val"], label="val" + label, lw=3, color="orchid"
    )
    plt.xlabel("Эпоха")
    plt.legend()

    # Вторая метрика
    plt.subplot(1, 3, 3)
    plt.title(metrics_names[1])
    plt.plot(
        history[metrics_names[1]]["train"],
        label="train" + label,
        lw=3,
        color="royalblue",
    )
    plt.plot(
        history[metrics_names[1]]["val"], label="val" + label, lw=3, color="orchid"
    )
    plt.xlabel("Эпоха")
    plt.legend()
    plt.show()


class Trainer:
    def __init__(self, metrics, metrics_names):
        self.metrics == metrics
        self.metrics_names = metrics_names

    def train(self, train_loader, model, criterion, optimizer, params):
        stream = train_loader
        model.train()

        full_loss = 0
        first_metric = 0
        second_metric = 0
        cnt = 0

        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(torch.float32).to(params["device"], non_blocking=True)
            target = (
                target.to(torch.float32).to(params["device"], non_blocking=True) / 255
            )

            output = model(images)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = nn.Sigmoid()(output)
            full_loss += loss.detach().cpu().numpy()
            metrics = self.metrics(
                target.detach().cpu().numpy(), output.detach().cpu().numpy()
            )
            first_metric += metrics[0]
            second_metric += metrics[1]
            cnt += 1
        numerator = cnt * params["batch_size"]
        return (
            full_loss / numerator,
            first_metric / numerator,
            second_metric / numerator,
        )

    def eval(self, val_loader, model, criterion, params):
        model.eval()
        stream = val_loader

        full_loss = 0
        first_metric = 0
        second_metric = 0
        cnt = 0

        with torch.no_grad():
            for i, (images, target) in enumerate(stream, start=1):
                images = images.to(torch.float32).to(
                    params["device"], non_blocking=True
                )
                target = (
                    target.to(torch.float32).to(params["device"], non_blocking=True)
                    / 255
                )
                output = model(images)
                loss = criterion(output, target)
                output = nn.Sigmoid()(output)
                full_loss += loss.detach().cpu().numpy()
                metrics = self.metrics(
                    target.detach().cpu().numpy(), output.detach().cpu().numpy()
                )
                first_metric += metrics[0]
                second_metric += metrics[1]
                cnt += 1
        numerator = cnt * params["batch_size"]
        return (
            full_loss / numerator,
            first_metric / numerator,
            second_metric / numerator,
        )

    def train_and_test(self, model, train_dataset, val_dataset, params):
        history = defaultdict(lambda: defaultdict(list))
        train_loader = DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=params["num_workers"],
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=params["batch_size"],
            shuffle=False,
            num_workers=params["num_workers"],
            pin_memory=True,
        )
        criterion = nn.BCEWithLogitsLoss().to(params["device"])
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        best_metric = np.inf
        for epoch in range(1, params["epochs"] + 1):
            train_loss, train_metric_1, train_metric_2 = self.train(
                train_loader, model, criterion, optimizer, params
            )
            history["loss"]["train"].append(train_loss)
            history[self.metrics_names[0]]["train"].append(train_metric_1)
            history[self.metrics_names[1]]["train"].append(train_metric_2)
            val_loss, val_metric_1, val_metric_2 = eval(
                val_loader, model, criterion, params
            )
            history["loss"]["val"].append(val_loss)
            history[self.metrics_names[0]]["val"].append(val_metric_1)
            history[self.metrics_names[1]]["val"].append(val_metric_2)

            clear_output()

            print("epoch:", epoch)
            print("train loss:", history["loss"]["train"][-1])
            print(
                "val ",
                self.metrics_names[0],
                ":",
                history[self.metrics_names[0]]["val"][-1],
            )
            print(
                "val",
                self.metrics_names[1],
                ":",
                history[self.metrics_names[1]]["val"][-1],
            )
            plot_learning_curves(history)
            if val_metric_2 < best_metric:
                best_metric = val_metric_2
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch}, params["path"]
                )

        return model, optimizer, history
